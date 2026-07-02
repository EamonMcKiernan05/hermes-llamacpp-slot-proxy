import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from slot_proxy.config import settings
from slot_proxy.detector import NewSessionDetector
from slot_proxy.eraser import erase_active_slot

logger = logging.getLogger("slot_proxy.proxy")

app = FastAPI(title="llamacpp-slot-proxy", version="0.1.0")
detector = NewSessionDetector()


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    upstream_ok = await _check_upstream()
    return {
        "status": "ok" if upstream_ok else "degraded",
        "upstream": settings.llama_base_url,
        "upstream_reachable": upstream_ok,
        "detect_mode": settings.detect_mode,
        "erase_enabled": settings.erase_enabled,
    }


@app.post("/trigger-erase")
async def trigger_erase() -> dict[str, Any]:
    """Manual erase trigger — use when heuristic is unreliable."""
    async with httpx.AsyncClient() as client:
        result = await erase_active_slot(client)
    return result


# ---------------------------------------------------------------------------
# Proxy: forward all /v1/* requests
# ---------------------------------------------------------------------------


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_v1(request: Request, path: str) -> Response:
    upstream = f"{settings.llama_base_url.rstrip('/')}/v1/{path}"

    body_bytes = await request.body()
    body_json = _try_parse_json(body_bytes)

    # Detection only on POST chat completions
    if settings.erase_enabled and path == "chat/completions" and request.method == "POST":
        if body_json is not None:
            result = detector.inspect_chat_request(body_json)
            if result.is_new_session:
                logger.info("new session detected — scheduling slot erase")
                asyncio.create_task(_background_erase())

    return await _forward_with_streaming(
        method=request.method,
        url=upstream,
        headers=dict(request.headers),
        body=body_bytes,
    )


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------


async def _forward_with_streaming(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> Response | StreamingResponse:
    """Forward request, preserving streaming if the upstream streams."""
    stripped_headers = {k: v for k, v in headers.items() if k.lower() != "host"}

    is_stream_request = (
        headers.get("accept", "") == "text/event-stream"
        or headers.get("x-stream-mode", "") == "chunked"
    )

    if is_stream_request:
        return await _stream_response(method, url, stripped_headers, body)

    return await _buffered_response(method, url, stripped_headers, body)


async def _stream_response(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> Response | StreamingResponse:
    """Proxy a streaming (SSE) response chunk by chunk."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(settings.upstream_timeout_s)) as client:
        req = client.build_request(method, url, headers=headers, content=body)

        try:
            upstream_resp = await client.send(req, stream=True)
        except httpx.ConnectError:
            return Response(
                content='{"error":"upstream unreachable"}',
                status_code=502,
                media_type="application/json",
            )

        async def _stream_chunks() -> AsyncGenerator[bytes, None]:
            async with upstream_resp:  # type: ignore[attr-defined]
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(
            content=_stream_chunks(),
            status_code=upstream_resp.status_code,
            headers={
                k: v
                for k, v in upstream_resp.headers.items()
                if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
            },
            media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
        )


async def _buffered_response(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> Response:
    """Proxy a non-streaming response (buffered)."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(settings.upstream_timeout_s)) as client:
        try:
            upstream_resp = await client.request(method, url, headers=headers, content=body)
        except httpx.ConnectError:
            return Response(
                content='{"error":"upstream unreachable"}',
                status_code=502,
                media_type="application/json",
            )

        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=dict(upstream_resp.headers),
        )


async def _check_upstream() -> bool:
    """Ping the upstream health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            r = await client.get(f"{settings.llama_base_url.rstrip('/')}/health")
            r.raise_for_status()
            return True
    except Exception:
        return False


async def _background_erase() -> None:
    """Fire-and-forget slot erase in the background."""
    try:
        async with httpx.AsyncClient() as client:
            result = await erase_active_slot(client)
        logger.info("background erase completed: %s", result)
    except Exception as exc:
        logger.warning("background erase raised: %s", exc)


def _try_parse_json(body: bytes) -> dict[str, Any] | None:
    if not body:
        return None
    import json

    try:
        result: Any = json.loads(body)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
