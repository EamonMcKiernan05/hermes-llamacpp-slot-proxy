import logging

import uvicorn

from slot_proxy.config import settings


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log = logging.getLogger("slot_proxy.main")
    log.info("starting proxy on port %d -> %s/v1", settings.proxy_port, settings.llama_base_url)
    log.info("detect_mode=%s erase_enabled=%s", settings.detect_mode, settings.erase_enabled)
    if settings.erase_slot_id is not None:
        log.info("pinned slot id=%d", settings.erase_slot_id)
    else:
        log.info("slot discovery via /slots endpoint")

    uvicorn.run(
        "slot_proxy.proxy:app",
        host="0.0.0.0",
        port=settings.proxy_port,
        log_level="info",
    )


if __name__ == "__main__":
    run()
