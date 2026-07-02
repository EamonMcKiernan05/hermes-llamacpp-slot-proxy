import logging
from typing import Any

import httpx

from slot_proxy.config import settings

logger = logging.getLogger("slot_proxy.eraser")


async def erase_active_slot(client: httpx.AsyncClient) -> dict[str, Any]:
    """Erase the active llama.cpp slot's prompt cache.

    Sends POST /slots/{id}?action=erase with Content-Length: 0 to work
    around a known cpp-httplib bug that hangs on Content-Length mismatch.
    """
    slot_id = settings.erase_slot_id
    if slot_id is None:
        slot_id = await _discover_busy_slot_id(client)

    url = f"{settings.llama_base_url.rstrip('/')}/slots/{slot_id}"
    params = {"action": "erase"}

    try:
        resp = await client.post(
            url,
            params=params,
            headers={"Content-Length": "0"},
            timeout=settings.erase_timeout_s,
        )
        body_text = resp.text
        logger.info(
            "erase slot=%s status=%d body=%.200s",
            slot_id,
            resp.status_code,
            body_text,
        )
        return {"ok": True, "slot_id": slot_id, "status": resp.status_code, "body": body_text}
    except httpx.TimeoutException:
        logger.warning("slot erase timed out after %ss", settings.erase_timeout_s)
        return {"ok": False, "error": "timeout", "slot_id": slot_id}
    except httpx.HTTPStatusError as exc:
        logger.warning("slot erase HTTP error: %s", exc)
        return {"ok": False, "error": str(exc), "slot_id": slot_id}
    except Exception as exc:
        logger.warning("slot erase failed: %s", exc)
        return {"ok": False, "error": str(exc), "slot_id": slot_id}


async def _discover_busy_slot_id(client: httpx.AsyncClient) -> int:
    """Query /slots and return the first slot with an active task."""
    url = f"{settings.llama_base_url.rstrip('/')}/slots"
    try:
        resp = await client.get(url, timeout=settings.upstream_timeout_s)
        resp.raise_for_status()
        data = resp.json()
        for slot in data.get("slots", []):
            # A busy slot has a task with a non-negative id
            task_id = slot.get("task", {}).get("id", -1)
            if task_id != -1:
                return int(slot.get("id", 0))
        logger.warning("no busy slot found in /slots, falling back to slot 0")
        return 0
    except Exception as exc:
        logger.warning("could not discover busy slot (%s), falling back to slot 0", exc)
        return 0
