# Bug: Cluster Map Silent Hang + Ctrl+C Kill Failure

## Bug Description

When using the `cluster` strategy, the server loads the model, then prints "Loading cluster map from cache..." — and then nothing happens. No CPU/memory activity, no tokens generated, no error message. The UI is stuck. Ctrl+C cannot kill the server; only `scripts/stop.sh` (which sends `fuser -k`) works.

Expected: after loading the cluster map, generation starts immediately and tokens are streamed to the browser.
Actual: the pipeline silently hangs forever after "Loading cluster map from cache...".

## Problem Statement

The `_producer` asyncio task in `stream.py` runs `inspect_live` inside a thread via `asyncio.to_thread`. If `inspect_live` (or anything inside `_generate`) raises an exception, the exception is silently dropped because:

1. `producer_task` is created with `asyncio.create_task(_producer())` but **never awaited** — so any exception raised inside it is silently ignored.
2. The sentinel `None` that signals end-of-generation to `_synthesizer` is placed **after** `await asyncio.to_thread(_generate)` — so if `_generate` raises, the sentinel is never put on the queue.
3. `_synthesizer` then blocks forever on `await queue.get()`, causing the pipeline task to hang indefinitely.
4. A hung pipeline task prevents uvicorn from shutting down gracefully on Ctrl+C.

There is also insufficient logging: there is no stderr output after "Loading cluster map from cache..." so it is impossible to know whether generation started, failed, or is still loading.

## Solution Statement

1. Move the sentinel `await queue.put(None)` into a `finally` block inside `_producer`, so it is always sent even if `_generate` raises.
2. After `_synthesizer` completes, `await producer_task` to surface any exception from the producer (it will re-raise, which is caught by the outer `try/except` in `_run_pipeline`).
3. Add structured `logging` (and `print(..., file=sys.stderr, flush=True)` in `transform.py`) at every stage so the operator can follow progress in the uvicorn console output.

## Steps to Reproduce

1. Run `./scripts/start.sh`
2. Open the browser UI at `http://localhost:8080`
3. Select strategy = `cluster` (ensure a cluster cache file already exists under `neuronpedia_cache/`)
4. Click **Start**
5. Observe server logs: "Loading weights..." then "Loading cluster map from cache..." — then nothing forever
6. Try Ctrl+C in the terminal — does not stop the server

## Root Cause Analysis

In `stream.py::_run_pipeline`:

```python
producer_task = asyncio.create_task(_producer())
await _synthesizer()   # blocks forever if _producer fails silently
```

Inside `_producer`:
```python
async def _producer():
    def _generate():
        for token_analysis, elapsed_ms in inspect_live(...):
            asyncio.run_coroutine_threadsafe(queue.put(event), event_loop).result()
    await asyncio.to_thread(_generate)
    await queue.put(None)  # sentinel — NEVER reached if _generate raises
```

If `inspect_live` raises (e.g. a tensor shape mismatch, a CUDA error, or any runtime exception), the exception propagates out of `_generate`, cancels `asyncio.to_thread`, and the `await queue.put(None)` sentinel line is skipped. The `_producer` task fails silently (nobody awaits it), `_synthesizer` blocks on `await queue.get()` indefinitely, and the server process cannot be stopped with Ctrl+C because the event loop is stuck waiting on that queue.

The Ctrl+C issue is a direct consequence: uvicorn receives SIGINT and tries to cancel all tasks, but the hung asyncio task inside `_synthesizer` holds up graceful shutdown when the reloader architecture (`--reload`) is in play.

## Relevant Files

- `app/server/routers/stream.py` — contains `_run_pipeline`, `_producer`, `_synthesizer`; this is where the hang occurs and where logging must be added.
- `app/server/pipeline/transform.py` — contains `build_cluster_map`; adds stderr logging for cache hit/miss and each build stage.
- `app/server/pipeline/extract.py` — contains `inspect_live`; any exception here is the proximate trigger of the silent hang.
- `scripts/start.sh` — invokes uvicorn with `--reload`; signal propagation fix.

## Step by Step Tasks

### 1. Fix `_producer` sentinel — always send it, even on error

In `app/server/routers/stream.py`, refactor `_producer` so the sentinel is in a `finally` block and the exception is stored for re-raising:

- Replace the body of `_producer` with a try/finally that always calls `await queue.put(None)`.
- Store any exception in a local variable, then re-raise after the finally.
- After `await _synthesizer()`, add `await producer_task` so the exception is surfaced and caught by the outer `try/except Exception` block, which will then send `{"type": "error", ...}` to the browser.

```python
async def _producer():
    try:
        def _generate():
            for token_analysis, elapsed_ms in inspect_live(
                params.prompt, model, tokenizer, sae,
                params.layer, neuronpedia,
                max_new_tokens=params.max_tokens,
            ):
                active_features = [f.model_dump() for f in token_analysis.active_features]
                if params.strategy == "cluster":
                    notes = apply_cluster(active_features, cluster_map)
                else:
                    notes = apply_identity(active_features)
                event = {
                    "type": "token",
                    "token": token_analysis.token,
                    "token_id": token_analysis.token_id,
                    "elapsed_ms": elapsed_ms,
                    "notes": notes,
                }
                asyncio.run_coroutine_threadsafe(queue.put(event), event_loop).result()

        await asyncio.to_thread(_generate)
    finally:
        await queue.put(None)  # always signal done, even on error
```

Then after `await _synthesizer()`:
```python
producer_task = asyncio.create_task(_producer())
await _synthesizer()
await producer_task  # re-raises any exception from generation
```

### 2. Add logging throughout `stream.py`

Add `logger.info(...)` calls at every stage in `_run_pipeline`:

- Before and after each loading step (model, SAE, neuronpedia, cluster map).
- When `inspect_live` is about to start (log prompt, max_tokens, strategy).
- Each token generated: log token index and elapsed_ms (at DEBUG level to avoid flooding).
- When generation completes: log total token count.
- On exception: `logger.exception(...)` with context.

Example additions in `_run_pipeline`:
```python
logger.info("Starting pipeline: prompt=%r strategy=%s clusters=%s", params.prompt, params.strategy, params.clusters)
# after cluster map loaded:
logger.info("Cluster map ready: %d entries", len(cluster_map))
# before generation:
logger.info("Starting generation: max_tokens=%d", params.max_tokens)
# inside _generate after each token:
logger.debug("Token %d generated in %dms: %r", token_count, elapsed_ms, token_analysis.token)
# after generation:
logger.info("Generation complete: %d tokens", token_count)
```

### 3. Add logging in `transform.py::build_cluster_map`

Replace bare `print(..., file=sys.stderr)` calls with proper logging, and add more granular progress points:

- Log cache path being checked.
- Log whether cache was found or not.
- Log number of entries loaded from cache.
- Log time taken to load from cache.

```python
import logging
import time
logger = logging.getLogger(__name__)

# at cache hit:
logger.info("Loading cluster map from cache: %s", cache_path)
t0 = time.perf_counter()
with open(cache_path) as f:
    raw = json.load(f)
result = {int(k): v for k, v in raw.items()}
logger.info("Cluster map loaded: %d entries in %.2fs", len(result), time.perf_counter() - t0)
return result
```

### 4. Fix Ctrl+C / graceful shutdown in `scripts/start.sh`

The `--reload` flag causes uvicorn to run a reloader supervisor process that spawns the actual server process. In WSL2, Ctrl+C sends SIGINT to the terminal's process group, but the reloader may not propagate it cleanly to the worker. Remove `--reload` from `start.sh` to run uvicorn directly — this makes Ctrl+C work as expected. Development reloading can be done manually.

Change `scripts/start.sh`:
```bash
#!/usr/bin/env bash
PYTHONPATH=. uv run uvicorn app.server.main:app --host 0.0.0.0 --port 8080
```

### 5. Validate the fix

Run validation commands listed below.

## Validation Commands

```bash
# Start the server (no --reload, Ctrl+C should now work)
./scripts/start.sh

# In a separate shell, verify the server is up
curl -s http://localhost:8080/ | head -5

# Check that the server stops cleanly on Ctrl+C (press Ctrl+C in the start.sh terminal)
# If it stops cleanly, Ctrl+C fix is confirmed.

# Re-start and trigger a cluster strategy run via the browser at http://localhost:8080
# Observe server logs — should now see:
#   INFO ... Starting pipeline: prompt=... strategy=cluster ...
#   INFO ... Cluster map ready: N entries
#   INFO ... Starting generation: max_tokens=200
#   INFO ... Generation complete: N tokens
# (or an error message if inspect_live fails — which is now surfaced instead of silently dropped)

# Run server tests (if any)
cd app/server && uv run pytest
```

## Notes

- The primary bug is the missing `finally` on the sentinel `queue.put(None)`. All other symptoms (hang, Ctrl+C failure, opacity) are downstream of this one omission.
- After this fix, if `inspect_live` raises for any reason (CUDA OOM, tensor shape error, etc.), the error will be logged server-side AND sent to the browser as `{"type": "error", "message": "..."}`.
- The `--reload` removal may break hot-reload during development. If hot-reload is needed, run `./scripts/stop.sh && ./scripts/start.sh` manually after code changes.
- The `event_loop = asyncio.get_event_loop()` line inside the coroutine is fine (it returns the running loop), but consider replacing with `asyncio.get_running_loop()` for clarity and forward compatibility.
