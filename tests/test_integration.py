"""Integration tests — require a running llama-server.

Usage:
    pytest tests/test_integration.py --llama-url http://localhost:8080

Or set LLAMA_BASE_URL env var.
"""

from __future__ import annotations

import os

import httpx
import pytest

LLAMA_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upstream_health():
    """Verify the upstream llama-server is reachable."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{LLAMA_URL}/health", timeout=5.0)
    assert r.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slots_endpoint():
    """GET /slots should return a list of slots."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{LLAMA_URL}/slots", timeout=5.0)
    if r.status_code == 404:
        pytest.skip("llama-server does not expose /slots endpoint")
    assert r.status_code == 200
    data = r.json()
    assert "slots" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_erase_busy_slot():
    """Find the busy slot and erase it. Expect n_erased > 0."""
    async with httpx.AsyncClient() as client:
        # discover
        slots_resp = await client.get(f"{LLAMA_URL}/slots", timeout=5.0)
        if slots_resp.status_code == 404:
            pytest.skip("/slots not available")
        slots = slots_resp.json().get("slots", [])
        busy = None
        for s in slots:
            if s.get("task", {}).get("id", -1) != -1:
                busy = s["id"]
                break
        if busy is None:
            pytest.skip("no busy slot found — send a request first")

        # erase
        r = await client.post(
            f"{LLAMA_URL}/slots/{busy}?action=erase",
            headers={"Content-Length": "0"},
            timeout=5.0,
        )
    assert r.status_code == 200
    data = r.json()
    # llama.cpp returns {"n_erased": N, "slot": id}
    if "n_erased" in data:
        assert data["n_erased"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_completion_passthrough():
    """Send a simple request through the proxy, verify response."""
    proxy_url = os.environ.get("PROXY_URL", "http://localhost:8081")
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{proxy_url}/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "say hi"}],
                "max_tokens": 10,
            },
            timeout=30.0,
        )
    assert r.status_code == 200
    data = r.json()
    assert "choices" in data
