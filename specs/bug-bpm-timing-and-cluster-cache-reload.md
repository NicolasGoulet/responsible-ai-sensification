# Bug: BPM timing missing from initial playback & spurious cluster-cache loading stage

## Bug Description

Two related issues in the web streaming pipeline:

1. **Spurious cluster-cache loading stage**: When a prompt is sent or re-sent and the cluster map is already in `_cluster_cache`, the server still emits a `{"type": "loading", "stage": "cluster map cached"}` WebSocket message. The user sees a "loading" flash on every Start/Send even though the cache hit is instantaneous.

2. **All tokens fired instantaneously, BPM ignored on initial playback**: After generation completes, `_run_pipeline` sends all 200 token events in a tight loop with zero delay. The BPM-based sleep exists only in the loop-replay section, not in the initial pass. With 200 tokens and no loop, all notes arrive at the client within milliseconds and the pipeline ends up frozen at the last frame.

3. **BPM changes only take effect at the start of the next full token replay**: In the loop-replay section, `sleep_s` is computed once per full pass through `collected`, so a live BPM slider change is not picked up until the current full replay finishes.

Expected behaviour: the synthesizer owns the clock — it reads `params.bpm` on every individual token so live changes take effect on the very next note, both during initial playback and during loop replay.

## Problem Statement

`_run_pipeline` mixes three concerns into one flat function:
- **Extraction**: runs `inspect_live`, collects all results before doing anything else
- **Transformation**: applies identity/cluster mapping inline
- **Synthesis / clock**: sends events — but only paces them in the loop-replay section, not the initial pass, and reads BPM once per full pass instead of per token

The synthesizer must own the clock, matching `synthesize.py` in the CLI pipeline (`live_timed` sleeps per token, `live_sustain` holds until the next arrives).

## Solution Statement

Restructure `_run_pipeline` into a **producer + synthesizer** coroutine pair connected by an `asyncio.Queue`:

- **`_producer`** — runs `inspect_live` in a thread, transforms each token as it arrives (identity or cluster), and puts the musical event onto the queue. Uses `asyncio.run_coroutine_threadsafe` to push from the worker thread into the async queue.
- **`_synthesizer`** — reads events from the queue one at a time. For `timed` mode it sleeps `60 / params.bpm` **after each token** (reading `params.bpm` fresh every time so live changes are instant). For `sustain` mode it sends immediately with no sleep. After the queue is drained it handles loop replay with the same per-token clock. Collects events for loop replay.

Both coroutines run concurrently via `asyncio.create_task`. `params` is the shared mutable `PipelineParams` instance so the synthesizer always sees the latest values.

Also remove the `await _send(ws, {"type": "loading", "stage": "cluster map cached"})` line — a cache hit should be silent.

## Steps to Reproduce

1. Start the server: `./scripts/start.sh`
2. Open `http://localhost:8000`
3. Set strategy = **Cluster**, mode = **Timed**, BPM = **120**, max tokens = **200**, loop = **off**
4. Click **Start** and wait for model + cluster map to load
5. Click **Send** (re-use the same prompt)
   - **Bug 1**: Status bar briefly shows "cluster map cached"
6. Watch the waveform canvas after generation finishes:
   - **Bug 2**: All 200 notes flash through in under a second, waveform freezes on the final frame
7. Enable loop, adjust BPM slider mid-replay:
   - **Bug 3**: New BPM does not take effect until the current full token pass finishes

## Root Cause Analysis

**Bug 1** — `stream.py:88` emits a loading message on a cache hit that is instantaneous and needs no status.

**Bug 2 & 3** — `_run_pipeline` collects all results first then fires them with no sleep:
```python
# current — no pacing, all 200 events sent in milliseconds
for token_analysis, elapsed_ms in results:
    await _send(ws, event)

# loop section — BPM sleep exists but sleep_s computed once per full pass
sleep_s = max(60 / params.bpm, 0.1)      # ← read once, stale for the whole pass
for event in collected:
    await _send(ws, {**event, "loop_count": loop_count})
    await asyncio.sleep(sleep_s)
```

The synthesizer does not own the clock; timing is bolted on only in the loop section and reads BPM too infrequently.

## Relevant Files

- **`app/server/routers/stream.py`** — the only file that needs to change. `_run_pipeline` is split into `_producer` + `_synthesizer` coroutines.

## Step by Step Tasks

### 1. Remove the spurious "cluster map cached" loading message

In the `if cluster_key in _cluster_cache:` branch, delete the `await _send(...)` line so a cache hit is completely silent.

### 2. Replace the flat generation + send loop with a producer coroutine

Inside `_run_pipeline`, after the cluster map is ready, define an async `_producer` coroutine:

```python
queue: asyncio.Queue = asyncio.Queue()
collected: list[dict] = []
loop = asyncio.get_event_loop()

async def _producer():
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
            asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()

    await asyncio.to_thread(_generate)
    await queue.put(None)  # sentinel: generation complete
```

### 3. Replace the flat send loop + loop-replay section with a synthesizer coroutine

Define an async `_synthesizer` coroutine that owns the clock:

```python
async def _synthesizer():
    # Initial playback: drain the queue
    while True:
        event = await queue.get()
        if event is None:
            break
        collected.append(event)
        await _send(ws, event)
        if params.mode == "timed":
            await asyncio.sleep(60.0 / params.bpm)  # read params.bpm fresh each token

    await _send(ws, {"type": "done"})

    # Post-generation: loop or idle
    loop_count = 0
    was_looping = False
    while True:
        if params.loop:
            was_looping = True
            loop_count += 1
            for event in collected:
                if not params.loop:
                    break
                await _send(ws, {**event, "loop_count": loop_count})
                if params.mode == "timed":
                    await asyncio.sleep(60.0 / params.bpm)  # live BPM per token
        else:
            if was_looping:
                was_looping = False
                await _send(ws, {"type": "silent"})
            await asyncio.sleep(0.1)
```

### 4. Wire producer and synthesizer together

Replace the existing generation + send code with:

```python
producer_task = asyncio.create_task(_producer())
await _synthesizer()
```

The synthesizer drives execution; the producer fills the queue concurrently. `producer_task` will be done by the time the synthesizer reads the sentinel.

### 5. Validate

Run validation commands below.

## Validation Commands

```bash
# Confirm no syntax errors
cd /home/apprentyr/projects/responsible-ai-sensification && uv run python -c "import app.server.routers.stream; print('OK')"

# Run server tests
cd /home/apprentyr/projects/responsible-ai-sensification/app/server && uv run pytest
```

Manual smoke test:
1. `./scripts/start.sh`
2. Open `http://localhost:8000`, set mode=Timed, BPM=120, max_tokens=10, loop=off, strategy=cluster
3. Click Start → cluster cache hit is **silent** (no "cluster map cached" status)
4. Waveform updates one frame every ~0.5 s for 10 tokens, then stops cleanly
5. Enable loop, wait for second pass, adjust BPM slider mid-pass — pacing changes **on the very next token**
6. Click Send again → same behaviour, no regression

## Notes

- `params` is the shared `PipelineParams` instance mutated by `update_params` messages, so reading `params.bpm` and `params.mode` inside the synthesizer loop always reflects the latest client values.
- `sustain` mode has no sleep — timing is implicit in token arrival rate from the queue.
- No new dependencies required.
