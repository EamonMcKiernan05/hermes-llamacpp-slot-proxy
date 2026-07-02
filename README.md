# hermes-llamacpp-slot-proxy

Tiny HTTP proxy that sits between [Hermes Agent](https://hermes-agent.nousresearch.com)
and a local `llama-server` (llama.cpp). It detects when Hermes starts a logically
new session (`/new` in Hermes TUI) and erases the active llama.cpp slot's prompt
cache via `POST /slots/{id}?action=erase`, so VRAM actually drops between sessions
instead of growing forever.

## Why

When you run `llama-server --parallel 1` (a single slot), llama.cpp keeps that
slot's KV cache alive across logical sessions. Hermes's `/new` command is
Hermes-side only — it resets the message buffer in Hermes but does **not**
instruct llama.cpp to clear its slot. The KV cache grows with each new session
until it hits context limits or runs OOM.

The proxy detects the `/new` pattern (system-prefixed message buffer that drops
by ≥50% between requests) and calls the erase endpoint non-blocking in the
background. The forwarded request is never delayed.

```
   Hermes Agent                :8081                    :8080
   ┌──────────┐    /v1/*    ┌──────────────┐    /v1/*    ┌──────────┐
   │  Hermes  │ ──────────► │  this proxy  │ ──────────► │ llama.cpp│
   │  TUI/API │ ◄────────── │              │ ◄────────── │          │
   └──────────┘   response  │   slot-reset │   response  └──────────┘
                            │              │
                            │ background:  │ POST /slots/{id}?action=erase
                            │ ────────────►│ (Content-Length: 0)
                            └──────────────┘
```

## Quickstart

```bash
# Clone and install
git clone https://github.com/EamonMcKiernan05/hermes-llamacpp-slot-proxy.git
cd hermes-llamacpp-slot-proxy
python -m venv .venv && source .venv/bin/activate
pip install .

# Or via pip (not yet on PyPI):
# pip install hermes-llamacpp-slot-proxy

# Create config
cp .env.example .env
# (Edit .env if your llama-server isn't on :8080)

# Run
slot-proxy
# Listening on :8081, proxying -> http://localhost:8080/v1
```

**Docker:**

```bash
docker run -d \
  --name slot-proxy \
  -p 8081:8081 \
  -e LLAMA_BASE_URL=http://host.docker.internal:8080 \
  ghcr.io/eamonmckiernan05/hermes-llamacpp-slot-proxy:latest
```

## Hermes Config

Change one line in your Hermes `config.yaml`:

```yaml
# Before (direct to llama.cpp):
# provider: ...
#   base_url: http://localhost:8080/v1

# After (via slot-reset proxy):
model:
  provider: ...
  base_url: http://localhost:8081/v1
```

No other changes needed. The proxy speaks the same OpenAI-compatible `/v1/*` API.

If you want to disable the proxy and go back to direct connection, stop the proxy
process and restore the original `base_url`. Alternatively, set `ERASE_ENABLED=false`
in the proxy's `.env` — it becomes a transparent pass-through with no side effects.

## Configuration

All config via environment variables (`.env` file on disk or inline):

| Variable | Default | Description |
|---|---|---|
| `LLAMA_BASE_URL` | `http://localhost:8080` | Upstream llama-server base URL (no `/v1` suffix) |
| `PROXY_PORT` | `8081` | Port the proxy listens on |
| `LLAMA_ERASE_SLOT_ID` | _(auto-discover)_ | Pin to a specific slot (0, 1, ...). If empty, queries `/slots` to find the busy one. |
| `ERASE_TIMEOUT_S` | `5.0` | Timeout in seconds for the erase request itself |
| `UPSTREAM_TIMEOUT_S` | `300.0` | Timeout for forwarding requests to llama.cpp |
| `HERMES_NEW_DETECT_MODE` | `auto` | `auto` = use heuristic; `manual` = disable auto-detect, use `/trigger-erase` endpoint |
| `ERASE_ENABLED` | `true` | Set `false` to run as transparent proxy with no erase logic at all |

## Endpoints

| Path | Method | Description |
|---|---|---|
| `/v1/{path}` | Any | Proxied to upstream llama.cpp (passthrough, streaming-aware) |
| `/health` | GET | Proxy health + upstream reachability check |
| `/trigger-erase` | POST | Manually trigger slot erase (bypasses detection heuristic) |

## Detection Heuristic

The proxy tracks per-client state (in memory): last request's message count and
first message role. It flags a `/new` when:

1. **Rule 1 (sharp drop):** Both the previous and current request start with
   `role: system`, and the current message count is ≤ 50% of the previous count.
   This catches the typical `/new` pattern where Hermes sends only the system
   prompt (1-3 messages) vs a full conversation (8-20+ messages).

2. **Rule 2 (long-session reset):** Previous request had > 20 messages and the
   current request has ≤ 3 messages. Catches edge cases where the first message
   role isn't system.

If neither rule fires, the request passes through untouched. Detection runs
synchronously (trivial dict lookup + integer comparison — ~0.001ms) and a
background task is spawned for the erase — **the forwarded request is never
blocked by the erase call**.

### False positive mitigation

- The detector resets its state only after a match, so rapid oscillation is
  dampened.
- Set `HERMES_NEW_DETECT_MODE=manual` to disable heuristics entirely and use
  `POST /trigger-erase` as a cron job, webhook, or manual command.
- Set `ERASE_ENABLED=false` to run a completely transparent proxy with zero
  side effects.

## Troubleshooting

### Erase hangs

Known `cpp-httplib` bug: requests with `Content-Length` not matching the body
body size can hang indefinitely. The proxy always sends `Content-Length: 0` with
an empty body to work around this. If you still see hangs, check that
`ERASE_TIMEOUT_S` is set (default 5s) — the proxy will break the hang after the
timeout and log a warning.

### "Slot busy" or no n_erased

The erase endpoint is non-blocking from the proxy's side but llama.cpp may
refuse to erase a slot that's actively processing. The erase is a hint —
llama.cpp will eventually free the cache when the slot's current task finishes.
If you need immediate release, restart llama-server.

### Detection false positives (erase fires too often)

- Check `HERMES_NEW_DETECT_MODE=auto` (default). If your workflow sends
  wildly varying message counts, switch to `manual` and trigger erase via a cron
  job or webhook.
- The heuristic is conservative — it only fires on a ≥50% drop with system-role
  prefix. If you're seeing false positives, open an issue with the request
  patterns.

### Detection false negatives (erase never fires)

The most common cause: first message role is not `system`. Hermes always sends
a system prompt on `/new`, but if you've customized your model config to omit
it, the heuristic won't match. Set `HERMES_NEW_DETECT_MODE=manual` and use
`POST /trigger-erase` on a schedule (e.g. every 30 minutes via cron).

### Hermes retry storms

If Hermes retries a failed request, each retry carries the same message count
and won't trigger a new detection (the detector sees the same count as the
previous attempt). No issue.

### OOM still happening

The erase frees only the prompt cache, not the model weights. If VRAM is
growing across sessions despite erase succeeding, you may be hitting a
llama.cpp memory leak or a Hermes-side context accumulation. Try reducing
`--ctx-size` or restarting the server periodically.

### /slots endpoint disabled

If your llama.cpp build doesn't include `LLAMA_SERVER_LLM` or was compiled
without the slots HTTP endpoint, the auto-discovery will fail. Set
`LLAMA_ERASE_SLOT_ID=0` (or your slot number) explicitly in `.env` to skip
discovery.

## Development

```bash
git clone https://github.com/EamonMcKiernan05/hermes-llamacpp-slot-proxy.git
cd hermes-llamacpp-slot-proxy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run all checks
ruff check src/ tests/
mypy src/
pytest -v tests/

# Run integration tests (requires running llama-server)
LLAMA_BASE_URL=http://localhost:8080 pytest -v tests/test_integration.py
```

### Project structure

```
src/slot_proxy/
├── __init__.py       # Package init (empty)
├── config.py         # Settings via pydantic-settings + .env
├── detector.py       # NewSessionDetector heuristic
├── eraser.py         # Slot erase client (discovery + POST)
├── main.py           # CLI entry point (uvicorn)
└── proxy.py          # FastAPI app, routing, streaming passthrough
tests/
├── test_proxy.py     # Unit tests (detector, eraser, proxy passthrough)
└── test_integration.py  # Integration tests (requires llama-server)
```

## License

MIT © Eamon McKiernan
>>>>>>> 25fef66 (Initial commit: llama.cpp slot-reset proxy)
