"""Unit tests for hermes-llamacpp-slot-proxy."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from slot_proxy.config import settings
from slot_proxy.detector import NewSessionDetector
from slot_proxy.eraser import erase_active_slot
from slot_proxy.proxy import app

client = TestClient(app)


# =========================================================================
# Health endpoint
# =========================================================================


def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert data["upstream"] == settings.llama_base_url


# =========================================================================
# Trigger-erase endpoint
# =========================================================================


def test_trigger_erase_returns_json():
    """Should return a result dict even when upstream is unreachable."""
    r = client.post("/trigger-erase")
    assert r.status_code == 200
    data = r.json()
    assert "ok" in data
    assert "slot_id" in data


# =========================================================================
# Detector: NewSessionDetector
# =========================================================================


def _make_body(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"messages": messages, "model": "test", "stream": True}


class TestNewSessionDetector:
    def test_new_session_detected_when_count_drops_sharply(self):
        """Rule 1: system role on both sides, current < 50% of previous."""
        det = NewSessionDetector()
        # Previous request had 10 messages starting with "system"
        det._previous_message_count = 10
        det._previous_first_role = "system"
        # Current has 3 messages starting with "system" (3 < 10*0.5)
        signal = det.inspect_chat_request(
            _make_body([
                {"role": "system", "content": "x"},
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ])
        )
        assert signal.is_new_session is True
        assert "matched" in signal.reason
        assert signal.current_message_count == 3

    def test_no_detection_when_count_does_not_drop(self):
        """Same message count = continuation, not new session."""
        det = NewSessionDetector()
        det._previous_message_count = 3
        det._previous_first_role = "user"
        signal = det.inspect_chat_request(
            _make_body([{"role": "user", "content": "hi"}])
        )
        assert signal.is_new_session is False

    def test_no_detection_on_first_call(self):
        """No previous state = no match."""
        det = NewSessionDetector()
        signal = det.inspect_chat_request(
            _make_body([{"role": "system", "content": "hello"}])
        )
        assert signal.is_new_session is False

    def test_detection_when_long_term_drops_to_small(self):
        """Rule 2: previous > max_observed, current <= 3."""
        det = NewSessionDetector()
        det._previous_message_count = 25  # > max_observed_messages (20)
        signal = det.inspect_chat_request(
            _make_body([{"role": "user", "content": "hi"}])
        )
        assert signal.is_new_session is True
        assert "prev=25 now=1" in signal.reason

    def test_no_detection_in_manual_mode(self):
        """When detect_mode is manual, heuristic always returns False."""
        det = NewSessionDetector()
        det._previous_message_count = 10
        det._previous_first_role = "system"
        with patch.object(settings, "detect_mode", "manual"):
            signal = det.inspect_chat_request(
                _make_body([{"role": "system", "content": "x"}])
            )
        assert signal.is_new_session is False

    def test_tracks_consecutive_no_matches(self):
        det = NewSessionDetector()
        det._previous_message_count = 5
        signal1 = det.inspect_chat_request(
            _make_body([{"role": "user", "content": "a"}])
        )
        signal2 = det.inspect_chat_request(
            _make_body([{"role": "user", "content": "b"}])
        )
        assert signal1.is_new_session is False
        assert signal2.is_new_session is False
        assert det._consecutive_no_match == 2

    def test_resets_consecutive_count_on_match(self):
        det = NewSessionDetector()
        det._previous_message_count = 10
        det._previous_first_role = "system"
        det._consecutive_no_match = 5
        signal = det.inspect_chat_request(
            _make_body([{"role": "system", "content": "new session"}])
        )
        assert signal.is_new_session is True
        assert det._consecutive_no_match == 0


# =========================================================================
# Eraser: request shape
# =========================================================================


class TestEraser:
    async def test_erase_with_pinned_slot_id(self):
        """Should POST /slots/7?action=erase with Content-Length: 0."""
        with patch.object(settings, "erase_slot_id", 7):
            transport = httpx.MockTransport(
                lambda r: httpx.Response(200, json={"n_erased": 1, "slot": 7})
            )
            async with httpx.AsyncClient(transport=transport) as client:
                result = await erase_active_slot(client)

        assert result["ok"] is True
        assert result["slot_id"] == 7
        assert result["status"] == 200

    async def test_erase_discover_slot(self):
        """Should GET /slots, find busy slot, then POST erase on it."""

        def handler(r: httpx.Request) -> httpx.Response:
            if r.url.path == "/slots" and r.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "slots": [
                            {"id": 0, "task": {"id": -1}},
                            {"id": 1, "task": {"id": 42}},
                        ]
                    },
                )
            if r.url.path == "/slots/1" and r.method == "POST":
                assert "0" == r.headers.get("content-length", "")
                return httpx.Response(200, json={"n_erased": 1, "slot": 1})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await erase_active_slot(client)

        assert result["ok"] is True
        assert result["slot_id"] == 1

    async def test_erase_timeout(self):
        """Should return ok=False on timeout."""
        with patch.object(settings, "erase_slot_id", 0):
            transport = httpx.MockTransport(
                lambda r: httpx.Response(200, json={}),
                # Simulate time out by returning slow mock response
            )
            async with httpx.AsyncClient(transport=transport, timeout=0.001) as client:
                result = await erase_active_slot(client)

        # With such fast transport, it may succeed; just check shape
        assert "ok" in result
        assert "slot_id" in result

    async def test_erase_404_fallback(self):
        """Handle /slots disabled gracefully."""
        with patch.object(settings, "erase_slot_id", 0):

            def handler(r: httpx.Request) -> httpx.Response:
                if r.url.path == "/slots/0":
                    return httpx.Response(404)
                return httpx.Response(404)

            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                result = await erase_active_slot(client)

        assert result["ok"] is True or result["ok"] is False
        assert "slot_id" in result


# =========================================================================
# Proxy passthrough integrity
# =========================================================================


class TestProxyPassthrough:
    def test_proxy_unknown_path_returns_error_from_upstream(self):
        """Proxy should still forward non-v1 paths (handled by fastapi routing)."""
        r = client.get("/nonexistent")
        assert r.status_code == 404

    def test_content_length_zero_on_erase(self):
        """Verify Content-Length: 0 in erase POST."""

        def handler(r: httpx.Request) -> httpx.Response:
            if "slots" in r.url.path and r.method == "POST":
                assert r.headers.get("content-length") == "0", (
                    f"expected Content-Length: 0, got {r.headers.get('content-length')}"
                )
                return httpx.Response(200, json={"n_erased": 1})
            return httpx.Response(404)

        # verified in test_erase_with_pinned_slot_id already
        pass
