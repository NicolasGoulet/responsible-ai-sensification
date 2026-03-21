# Bug: Silent Pipeline — No Logs Visible, Hang Undiagnosable

## Bug Description

After the previous fix, the terminal shows even less output than before. The server starts, the WebSocket connects, the model loads (340/340 tqdm bar), then nothing — no log lines, no progress, no error. The pipeline hang remains. Ctrl+C still cannot stop the server. The previous fix replaced bare `print(..., file=sys.stderr)` calls in `transform.py` with `logger.info(...)`, but the Python root logger has no handler and defaults to WARNING level, so those messages are silently discarded. The visibility regression made the hang harder to diagnose.

**Expected:** With a `--verbose` flag, every stage of the pipeline (model load, SAE load, neuronpedia load, cluster map load, generation start, each token, generation end) is printed to stderr, making it possible to identify exactly where the hang occurs.

**Actual:** No app-level log output at any stage. The hang location is unknown.

## Problem Statement

Two independent problems:

1. **Logging is silently discarded.** `logging.getLogger(__name__)` in `stream.py` and `transform.py` sends messages to a logger with no handler (Python root logger is at WARNING by default; uvicorn only configures its own `uvicorn.*` loggers). Nothing is ever printed.

2. **No `--verbose` flag exists.** There is no way to turn on detailed pipeline tracing without editing source code.

## Solution Statement

1. Add `--verbose` to `scripts/start.sh` which sets `VERBOSE=1` env var and passes `--log-level debug` to uvicorn.
2. In `app/server/main.py`, on startup read the `VERBOSE` env var and call `logging.basicConfig(level=logging.DEBUG)` (or `INFO` when not verbose) so the root logger gets a stderr handler and all `logger.*` calls in the app are printed.
3. In `app/server/routers/stream.py`, add `print(..., file=sys.stderr, flush=True)` at every pipeline stage as belt-and-suspenders diagnostics that work regardless of logging configuration.
4. In `app/server/pipeline/extract.py`, add prints inside `load_sae` (before/after `hf_hub_download` and `load_file`) and at the start of each token in `inspect_live`, so we can see whether the hang is in SAE loading or in generation.
5. In `app/server/pipeline/transform.py`, restore `print(..., file=sys.stderr, flush=True)` for the cache-hit path (reverting the silent `logger.info` regression from the prior fix — will be proper once logging is configured, but print is always safe).

## Steps to Reproduce

1. `./scripts/start.sh`
2. Open browser at `http://localhost:8080`
3. Select strategy = `cluster` with a valid cache file
4. Click **Start**
5. Observe: "Loading weights: 340/340" then silence — no log output, no tokens, no error
6. Ctrl+C — server does not stop

## Root Cause Analysis

**Logging not visible:** Python's `logging` module requires a handler to be attached to the root logger before any message is printed. uvicorn's `logging.config.dictConfig(LOGGING_CONFIG)` on startup configures only `uvicorn` and `uvicorn.access` named loggers, leaving the root logger with no handler and level=WARNING. All `logger.info(...)` calls in `app.*` modules produce records that bubble up to the root logger and are silently dropped.

The previous fix in `transform.py` worsened this by replacing unconditional `print(..., file=sys.stderr)` (which always works) with `logger.info(...)` (which is always silently dropped).

**Hang location unknown:** Because no prints or logs appear after "Loading weights: 340/340", the hang could be in:
- `hf_hub_download` inside `load_sae` (makes a HEAD request to HuggingFace to verify cached file — can hang on slow/blocked network in WSL2)
- `download_neuronpedia_explanations` cache read (unlikely, but possible)
- `build_cluster_map` cache load (unlikely)
- The first call to `model(input_ids)` inside `inspect_live` (CUDA forward pass)

Without prints at each stage we cannot distinguish these cases.

**Ctrl+C not working:** uvicorn without `--reload` should handle SIGINT, but if the thread running `asyncio.to_thread(_load_model)` or `asyncio.to_thread(_generate)` is blocked in a C extension (PyTorch CUDA kernel), the Python signal handler is set but cannot preempt the C thread. The event loop is waiting for the thread future to complete and cannot process the cancellation. This is a CUDA/GIL interaction issue; it is out of scope for this plan but documented here for future reference.

## Relevant Files

- **`scripts/start.sh`** — needs `--verbose` flag support; sets `VERBOSE=1` env var.
- **`app/server/main.py`** — needs `logging.basicConfig()` on startup, gated on `VERBOSE` env var.
- **`app/server/routers/stream.py`** — needs `print(..., flush=True)` at every stage.
- **`app/server/pipeline/extract.py`** — needs prints inside `load_sae` and at first token in `inspect_live`.
- **`app/server/pipeline/transform.py`** — needs prints restored for cache-hit path (revert silent regression).

## Step by Step Tasks

### 1. Add `--verbose` support to `scripts/start.sh`

- Parse `$1 == "--verbose"` (or `$VERBOSE == 1`) and set `LOG_LEVEL=debug` and `VERBOSE=1`.
- Pass `--log-level $LOG_LEVEL` to uvicorn.
- Export `VERBOSE` so the FastAPI process inherits it.

```bash
#!/usr/bin/env bash
LOG_LEVEL="info"
if [ "$1" = "--verbose" ] || [ "${VERBOSE:-0}" = "1" ]; then
    LOG_LEVEL="debug"
    export VERBOSE=1
fi
PYTHONPATH=. uv run uvicorn app.server.main:app \
    --host 0.0.0.0 --port 8080 --log-level "$LOG_LEVEL"
```

### 2. Configure Python root logging in `app/server/main.py`

- Import `logging` and `os`.
- At module level (before the app is created), call `logging.basicConfig` with level based on `VERBOSE` env var.
- This ensures all `logger.*` calls in `app.*` modules are printed to stderr.

```python
import logging
import os

_log_level = logging.DEBUG if os.environ.get("VERBOSE") == "1" else logging.INFO
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
```

The `force=True` is needed because uvicorn's `dictConfig` may have already added a NullHandler to the root logger; `force=True` removes existing handlers and replaces them.

### 3. Add `print` diagnostics to `app/server/routers/stream.py`

After every `await` that loads a resource or starts generation, add a `print(..., file=sys.stderr, flush=True)` line so progress is visible even before logging is fully configured. These prints are in addition to existing `logger.*` calls.

Add at the following points:
- Before and after `asyncio.to_thread(_load_model)`: "Loading language model..." / "Language model loaded."
- Before and after `asyncio.to_thread(_load_sae)`: "Loading SAE..." / "SAE loaded."
- Before and after `asyncio.to_thread(_load_neuronpedia)`: "Loading Neuronpedia..." / "Neuronpedia loaded: N features."
- After cluster map is ready (both cache-hit and build paths): "Cluster map ready: N entries."
- Inside `_producer` just before `await asyncio.to_thread(_generate)`: "Generation starting..."
- Inside `_generate` before the first token iteration: "inspect_live started."

### 4. Add `print` diagnostics to `app/server/pipeline/extract.py`

In `load_sae`:
- Before `hf_hub_download`: print "Downloading/locating SAE weights from HuggingFace..."
- After `hf_hub_download`: print "SAE weights file located: {local_path}"
- After `load_file`: print "SAE tensors loaded."
- After `.to(device)`: print "SAE moved to device."

In `inspect_live`:
- Before the `for _ in range(max_new_tokens):` loop: print "inspect_live: starting generation loop (max_new_tokens={max_new_tokens})"
- After `outputs = model(input_ids)` on the first token only (use a counter): print "First forward pass complete."
- After each token yield: print at DEBUG (or every 10 tokens to avoid flooding)

### 5. Restore `print` in `app/server/pipeline/transform.py` cache-hit path

The previous fix replaced `print("Loading cluster map from cache...", file=sys.stderr)` with `logger.info(...)`. Restore it to `print` so it shows in the terminal unconditionally (the `logger.info` call can remain alongside it for when logging is properly configured):

```python
if cache_path.exists():
    print(f"Loading cluster map from cache: {cache_path}", file=sys.stderr, flush=True)
    logger.info("Loading cluster map from cache: %s", cache_path)
    t0 = time.perf_counter()
    with open(cache_path) as f:
        raw = json.load(f)
    result = {int(k): v for k, v in raw.items()}
    elapsed = time.perf_counter() - t0
    print(f"Cluster map loaded: {len(result)} entries in {elapsed:.2f}s", file=sys.stderr, flush=True)
    logger.info("Cluster map loaded: %d entries in %.2fs", len(result), elapsed)
    return result
```

### 6. Validate the fix

Run validation commands listed below.

## Validation Commands

```bash
# Start server in verbose mode and verify log output appears
./scripts/start.sh --verbose

# In a separate shell, trigger a cluster run via the browser at http://localhost:8080
# Observe terminal: should now print every stage:
#   Loading language model...
#   Language model loaded.
#   Loading SAE...
#   Downloading/locating SAE weights from HuggingFace...
#   SAE weights file located: ...
#   SAE loaded.
#   ...
#   Loading cluster map from cache: ...
#   Cluster map loaded: N entries in X.XXs
#   Generation starting...
#   inspect_live: starting generation loop (max_new_tokens=200)
#   First forward pass complete.

# Run server tests
cd app/server && uv run pytest
```

## Notes

- The primary ask is diagnostics, not a final fix. Once the prints show WHERE the hang is, a follow-up spec can address the root cause.
- `force=True` in `logging.basicConfig` is important in Python 3.8+ when uvicorn has already called `dictConfig`. Without it, `basicConfig` is a no-op if any handler exists on the root logger.
- The Ctrl+C issue with CUDA threads is a known Python limitation: `asyncio.to_thread` wraps a thread, and SIGINT sets a Python flag but cannot preempt a C extension (CUDA kernel). Until generation is refactored to use subprocess or has a stop flag checked between tokens, Ctrl+C will not work during active generation. `scripts/stop.sh` remains the reliable kill method.
- All `print(...)` calls added here use `file=sys.stderr, flush=True` to ensure immediate output even when stdout is buffered.
