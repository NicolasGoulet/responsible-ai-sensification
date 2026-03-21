# Feature: Web Server + Dynamic Control Frontend

## Feature Description
A FastAPI web server with a browser-based frontend that replaces CLI argument juggling with a live, interactive control panel. Users can type a prompt, pick pipeline options (strategy, loop, layer, clusters, etc.) and hit Start — all while generation is running they can tweak parameters that take effect on the next token without restarting anything. The frontend shows a canvas visualization of per-token notes data and a placeholder image panel. The server orchestrates `extract → transform` as an in-process async pipeline, streaming `MusicalEvent` events to the browser over WebSocket.

**Audio playback is out of scope for this feature** — it is tracked as Feature 1 in `specs/TODO.md`. The `notes` payload is included in every token event so the audio feature can be dropped in without server changes.

## User Story
As a researcher running live SAE sensification sessions,
I want to control all pipeline parameters from a browser UI and adjust them dynamically mid-stream,
So that I don't have to kill and restart command-line processes every time I want to try a different strategy, layer, or cluster count.

## Problem Statement
Every parameter change currently requires killing the running pipeline, editing the command, and restarting it. This is slow and disruptive when experimenting in real time. There is no live visualization of what the synthesizer is doing.

## Solution Statement
A thin FastAPI server exposes a WebSocket endpoint that streams `MusicalEvent` NDJSON to the browser as the pipeline runs. The browser sends parameter-update messages back over the same socket; the server applies them at the start of the next token. A vanilla-JS single-page client renders a live bar-chart canvas (frequency on X, amplitude on Y, one bar per active note) and a control panel matching the wireframe screenshot (prompt, strategy buttons, loop toggle, layer/width/clusters/mode controls, start/stop).

> **Decision log — confirmed:**
> - Audio output: **deferred to TODO Feature 1** (Web Audio API in browser)
> - Frontend stack: **Vanilla HTML + JS** — no build step, easy to tweak in devtools
> - Pipeline execution: **in-process Python modules** — no subprocesses, data flows as Python objects
> - Package layout: **shared root `pyproject.toml`** — monorepo, server deps added there
> - `synthesize.py` batch WAV generation remains available as a CLI but is not called by the server in this feature

## Relevant Files

- `extract.py` — core generation logic; moves to `app/server/pipeline/extract.py`, CLI entry-point preserved
- `transform.py` — transform logic; moves to `app/server/pipeline/transform.py`
- `synthesize.py` — batch WAV synthesis; moves to `app/server/pipeline/synthesize.py` (not used by server in this feature)
- `audio_utils.py` — shared audio helpers; moves to `app/server/pipeline/audio_utils.py`
- `export.py` — JSON export helper; moves to `app/server/pipeline/export.py`
- `pyproject.toml` — add `fastapi`, `uvicorn[standard]`, `websockets` dependencies
- `README.md` — update with new layout and server start instructions

### New Files
- `app/server/__init__.py` — empty package marker
- `app/server/main.py` — FastAPI app factory, mounts routers, serves `app/client/` as static files
- `app/server/pipeline/__init__.py` — empty package marker
- `app/server/session.py` — `PipelineParams` dataclass + `PipelineSession` (params + asyncio task handle)
- `app/server/routers/__init__.py` — empty package marker
- `app/server/routers/stream.py` — WebSocket `/ws/stream`: start/stop/update_params actions, streams token events
- `app/server/routers/config.py` — `GET /api/config/defaults` returns `PipelineParams` defaults as JSON
- `app/client/index.html` — single-page UI: prompt, controls left; canvas + image placeholder right
- `app/client/style.css` — dark theme, two-column flex layout
- `app/client/main.js` — WebSocket client, canvas bar-chart renderer, control panel wiring
- `scripts/start.sh` — `uv run uvicorn app.server.main:app --reload`
- `scripts/stop.sh` — kills uvicorn on port 8000

## Implementation Plan

### Phase 1: Restructure (commit independently)
Move all pipeline scripts into `app/server/pipeline/`, fix internal imports, verify CLI entry-points still work, then commit before touching any server code.

### Phase 2: Server
Build the FastAPI app with `session.py`, config router, and WebSocket stream router. The stream router runs `extract → transform` in-process as an async task, forwarding `MusicalEvent` dicts to the browser. It accepts `start / stop / update_params` messages from the client.

### Phase 3: Client
Vanilla-JS single-page UI with control panel on the left, canvas bar-chart + image placeholder on the right. No audio yet — the canvas encodes frequency on X and amplitude on Y per active note each token.

### Phase 4: Scripts + README
`scripts/start.sh`, `scripts/stop.sh`, update README with new layout and server usage.

## Step by Step Tasks

### Step 1: Move pipeline files into `app/server/pipeline/`
- Create directories: `app/server/pipeline/`, `app/server/routers/`, `app/client/`, `scripts/`
- Move `extract.py` → `app/server/pipeline/extract.py`
- Move `transform.py` → `app/server/pipeline/transform.py`
- Move `synthesize.py` → `app/server/pipeline/synthesize.py`
- Move `audio_utils.py` → `app/server/pipeline/audio_utils.py`
- Move `export.py` → `app/server/pipeline/export.py`
- Create `app/server/pipeline/__init__.py` (empty)
- Create `app/server/__init__.py` (empty)
- Fix imports inside moved files:
  - `extract.py`: `from export import ...` → `from app.server.pipeline.export import ...`
  - `transform.py`: `from audio_utils import ...` → `from app.server.pipeline.audio_utils import ...`
  - `synthesize.py`: `from audio_utils import ...` → `from app.server.pipeline.audio_utils import ...`
- Update root `pyproject.toml` `[tool.uv]` or add a `[tool.uv.sources]` so `uv run python app/server/pipeline/extract.py` resolves imports correctly (add `src` layout or set `pythonpath`)
- Verify: `uv run python app/server/pipeline/extract.py --help` prints usage

### Step 2: Commit the restructure
- Stage all moved files (use `git mv` to preserve history)
- Commit: `refactor: move pipeline scripts into app/server/pipeline/`
- Do NOT add/commit the PNG screenshot

### Step 3: Add server dependencies
- `uv add fastapi "uvicorn[standard]" websockets`
- Verify `pyproject.toml` updated correctly

### Step 4: Implement `app/server/session.py`
- Define `PipelineParams` dataclass:
  ```python
  @dataclass
  class PipelineParams:
      prompt: str = "Hello world"
      model: str = "google/gemma-3-1b-pt"
      layer: int = 22
      width: str = "65k"
      l0: str = "medium"
      max_tokens: int = 200
      strategy: str = "identity"   # "identity" | "cluster"
      clusters: int = 8
      loop: bool = False
      mode: str = "timed"          # "timed" | "sustain"
  ```
- Define `PipelineSession` holding `params: PipelineParams` + `task: asyncio.Task | None`

### Step 5: Implement `app/server/routers/config.py`
- `GET /api/config/defaults` → returns `PipelineParams` defaults as JSON
- `GET /api/config/model-options` → returns hardcoded lists for model IDs, layers, widths, strategies

### Step 6: Implement `app/server/routers/stream.py`
- WebSocket endpoint `WS /ws/stream`
- On connect: send `{"type": "ready", "params": <defaults>}`
- Accept incoming JSON messages:
  - `{"action": "start", "params": {...}}` → cancel any running task, merge params, start new pipeline task
  - `{"action": "stop"}` → cancel running task, send `{"type": "stopped"}`
  - `{"action": "update_params", "params": {...}}` → merge into session params (applied on next token / next loop iteration)
- Pipeline task (async, runs in background):
  1. Load model + SAE + Neuronpedia via `asyncio.to_thread` (they are sync/blocking); send `{"type": "loading", "stage": "..."}` events per stage; skip if module-level cache already holds the right model+layer+width
  2. Call `inspect_live(...)` inside `asyncio.to_thread`; for each `(token_analysis, elapsed_ms)`: apply `transform` (identity or cluster per current params), send result as `{"type": "token", "token": ..., "notes": [...]}` over WS
  3. If `params.loop=True`: replay collected token events indefinitely, sleeping `elapsed_ms` between each, checking for cancellation
  4. Handle `asyncio.CancelledError` cleanly; send `{"type": "stopped"}` on exit
- Module-level dict `_model_cache` keyed by `(model_id, layer, width)` avoids reloading across sessions
- `synthesize.py` is NOT called by the server — audio is deferred to TODO Feature 1

### Step 7: Implement `app/server/main.py`
- Create FastAPI app
- Include routers: `config.py`, `stream.py`
- Mount `app/client/` as static files at `/` (use `StaticFiles`)
- Root `GET /` serves `app/client/index.html`

### Step 8: Implement `app/client/index.html`
Layout matching the wireframe:
```
┌─────────────────────────────────────────┐
│ [Prompt input field]    [Start] [Stop]  │
│                         ┌─────────────┐ │
│ Strategy: [Raw] [Cluster]│  waveform  │ │
│ Layer: [slider 0-28]    │  canvas    │ │
│ Width: [65k ▼]          └─────────────┘ │
│ Clusters: [slider 2-32] ┌─────────────┐ │
│ Mode: [Timed][Sustain]  │  image      │ │
│ [x] Loop                │  placeholder│ │
│ Status: ...             └─────────────┘ │
└─────────────────────────────────────────┘
```
- Use semantic HTML with `<form>` for controls, `<canvas id="waveform">`, `<div id="image-placeholder">`
- Pre-populate all inputs with defaults from `GET /api/config/defaults`

### Step 9: Implement `app/client/style.css`
- Dark theme (`#1a1a2e` background, `#e0e0e0` text)
- Two-column flex layout: controls left (~40%), visuals right (~60%)
- Waveform canvas: black background, green wave line
- Image placeholder: dashed border, centered "Todo: image generation" text
- Buttons: strategy toggle group (active state highlighted), start (green), stop (red)

### Step 10: Implement `app/client/main.js`
- On load: fetch `/api/config/defaults`, populate all form fields with returned values
- `connectWS()`: open `ws://${location.host}/ws/stream`
- On WS message dispatch:
  - `type: "ready"` → populate controls from `msg.params` if not already filled
  - `type: "loading"` → update status bar text with `msg.stage`
  - `type: "token"` → call `drawNotes(msg.notes)`, increment token counter in status bar
  - `type: "stopped"` / `type: "error"` → reset UI to idle state
- `drawNotes(notes)`: clear canvas, draw one vertical bar per note (X = log-scaled freq mapped to canvas width, height = amplitude, colour by instrument or cluster)
- Start button: collect all form values, send `{"action": "start", "params": collectParams()}`
- Stop button: send `{"action": "stop"}`
- Each control's `input`/`change` event while running: send `{"action": "update_params", "params": {key: newValue}}`
- **No audio code in this feature** — canvas visualization only; Web Audio API added in TODO Feature 1

### Step 11: Implement `scripts/start.sh` and `scripts/stop.sh`
```bash
# start.sh
#!/usr/bin/env bash
uv run uvicorn app.server.main:app --host 0.0.0.0 --port 8000 --reload
```
```bash
# stop.sh
#!/usr/bin/env bash
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "Stopped" || echo "Not running"
```
- Make both executable: `chmod +x scripts/*.sh`

### Step 12: Update `README.md`
- Add new directory layout section
- Add "Run the server" section: `./scripts/start.sh` → open `http://localhost:8000`
- Keep existing CLI docs

### Step 13: Run validation commands
- See Validation Commands section below

## Testing Strategy

### Unit Tests
- `tests/test_pipeline_params.py` — validate `PipelineParams` merging/update logic
- `tests/test_config_route.py` — `GET /api/config/defaults` returns expected shape

### Integration Tests
- `tests/test_ws_stream.py` — connect to WS, send `start` with a tiny mock prompt, assert `token` events arrive and `stop` cleanly cancels
- Use `pytest-asyncio` + FastAPI `TestClient` with `websocket_connect()`

### Edge Cases
- Start while already running → old task cancelled, new one starts
- Stop before model finishes loading → cancellation propagates cleanly
- Invalid params in `update_params` → ignored or returned as `{"type": "error"}`
- Browser disconnects mid-stream → task is cancelled, model cache preserved

## Acceptance Criteria
- [ ] `./scripts/start.sh` starts the server; `http://localhost:8000` serves the UI
- [ ] Setting a prompt and clicking Start triggers the pipeline without touching the CLI
- [ ] Strategy toggle (Raw / Cluster), Loop checkbox, and numeric controls are all wired and update the running session
- [ ] Waveform canvas updates on every token event
- [ ] Clicking Stop cleanly cancels the pipeline
- [ ] Existing CLI commands (`uv run python app/server/pipeline/extract.py "prompt" --stream ...`) still work
- [ ] `uv run pytest` passes with zero failures

## Validation Commands
```bash
# 1. Verify restructured CLI still works
uv run python app/server/pipeline/extract.py --help

# 2. Verify server starts
uv run uvicorn app.server.main:app --host 0.0.0.0 --port 8000 &
sleep 2 && curl -s http://localhost:8000/api/config/defaults | python3 -m json.tool
kill %1

# 3. Run tests
uv run pytest tests/ -v
```

## Notes
- **Audio is deferred**: No `sounddevice` / audio playback in this feature. See `specs/TODO.md` Feature 1 for Web Audio API browser-side playback. The `notes` array is already present in every `token` WS event so the audio feature slots in without server changes.
- **Model cache**: Loading Gemma-3 + SAE takes ~30–60 s. The server keeps a module-level `_model_cache` dict keyed by `(model_id, layer, width)` and skips reloading if the key is already present. A `{"type": "loading", "stage": "cached"}` event is sent when the cache is hit.
- **New dependencies**: `fastapi`, `uvicorn[standard]`, `websockets` — add via `uv add fastapi "uvicorn[standard]" websockets`
- **Static file serving**: FastAPI's `StaticFiles` mounts `app/client/` at `/`; no separate HTTP server needed in dev.
- **Import path**: Root `pyproject.toml` is a flat layout project. After moving files to `app/server/pipeline/`, ensure `uv run` resolves `from app.server.pipeline.extract import ...` by adding the repo root to `PYTHONPATH` or by configuring `[tool.uv] pythonpath = ["."]` if supported, else prefix scripts with `PYTHONPATH=. uv run ...`.
- **CLI compatibility**: All moved scripts keep their `if __name__ == "__main__":` blocks. After the restructure, the CLI becomes `uv run python app/server/pipeline/extract.py "prompt" --stream ...`. Update README accordingly.
- **TODO backlog**: `specs/TODO.md` lists the planned follow-up features in order: Web Audio API, live waveform via AnalyserNode, image generation, Neuronpedia progress bar, session history.
