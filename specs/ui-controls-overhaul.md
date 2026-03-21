# Feature: UI Controls Overhaul

## Feature Description
Replace the current static sliders and toggle buttons in the control panel with smarter, context-aware controls. Model, layer, and width become dropdowns populated from the server so that only valid (downloaded) SAE configurations are offered. Strategy becomes a dropdown with descriptive tooltips. Cluster count and max-tokens become integer input fields. Timed mode exposes a note-duration field. A loop counter is displayed near the status bar to track how many full passes have completed in loop mode.

## User Story
As a researcher exploring SAE sensification
I want controls that reflect what is actually available on my machine and provide contextual help
So that I cannot accidentally select an invalid model/layer/width combination, and I understand what each parameter does without reading the docs

## Problem Statement
The current UI has several usability and correctness issues:
- Layer and Width are static HTML (range slider / select) that list all possible values even if most SAE weights are not downloaded — users will get runtime errors if they pick an unavailable combination.
- There is no model selector; the model is hardcoded server-side.
- Strategy, Mode, Cluster count and Max-tokens offer no explanation of what they do.
- The cluster count slider is imprecise for an integer value.
- There is no feedback on how many loop iterations have completed.

## Solution Statement
Expose a `/api/config/model-options` endpoint (already partially implemented) that returns the full catalogue of valid model → layer → width combinations. The frontend fetches this catalogue on load and dynamically populates dropdowns so only valid combinations can be selected. Additional UX improvements (tooltips, integer inputs, conditional fields, loop counter) are added alongside.

## Relevant Files

- **`app/server/routers/config.py`** — extend the existing `/api/config/model-options` endpoint to return per-model layer/width availability and strategy/mode descriptions.
- **`app/server/session.py`** — add `bpm: int` parameter (beats per minute controlling note pace in timed mode).
- **`app/server/routers/stream.py`** — emit `loop_count` field on each loop iteration event; derive note sleep duration from `bpm` (`60 / bpm` seconds) in the loop replay.
- **`app/client/index.html`** — replace slider/toggle controls with dropdowns, int inputs, tooltip markers, and a loop counter display.
- **`app/client/style.css`** — add styles for tooltips, integer inputs, conditional fields, and the loop counter.
- **`app/client/main.js`** — fetch model options, populate dropdowns, wire conditional field visibility (clusters input, bpm input), handle `loop_count` in incoming messages.

## Implementation Plan

### Phase 1: Foundation
- Extend the server config endpoint to carry per-model SAE availability and human-readable descriptions for strategy/mode options.
- Add `bpm` to `PipelineParams` and propagate it through the stream pipeline.
- Add `loop_count` tracking to the loop replay logic in `stream.py` and emit it in each replayed event.

### Phase 2: Core Implementation
- Rebuild `index.html` controls section: model → layer → width dropdowns, strategy dropdown with `[?]` tooltip triggers, clusters int input (conditional), max-tokens int input with tooltip, mode section with tooltip and conditional bpm int input, loop counter badge near status.
- Add all required CSS: tooltip popup, number input styling, `.hidden` utility class, loop counter style.
- Rewrite the relevant sections of `main.js`: fetch `/api/config/model-options`, populate dropdowns on load, cascade layer/width dropdowns when model changes, show/hide clusters input, show/hide bpm input, update loop counter on messages.

### Phase 3: Integration
- Verify the `collectParams` function includes `bpm` and `model`.
- Verify `applyParams` covers all new fields.
- Smoke-test the full flow: change model → layer/width update → start → stop → loop counter increments.

## Step by Step Tasks

### Step 1 — Extend `config.py` with per-model SAE catalogue and descriptions

- Define a `MODEL_CATALOGUE` dict mapping each model ID to its available `layers` (list[int]) and `widths` (list[str]).
  - `google/gemma-3-1b-pt` → layers: `[22]`, widths: `["65k"]`
  - `google/gemma-3-4b-pt` → layers: `[22]`, widths: `["65k"]` (placeholder, same for now)
- Define `STRATEGY_DESCRIPTIONS` and `MODE_DESCRIPTIONS` dicts with short human-readable strings.
- Update `/api/config/model-options` to return:
  ```json
  {
    "models": ["google/gemma-3-1b-pt"],
    "model_catalogue": {
      "google/gemma-3-1b-pt": { "layers": [22], "widths": ["65k"] }
    },
    "strategies": [
      { "value": "identity", "label": "Identity", "description": "Maps each active SAE feature directly to a frequency proportional to its index. Fast, no clustering." },
      { "value": "cluster", "label": "Cluster", "description": "Groups features by semantic similarity (k-means on Neuronpedia embeddings). Each cluster becomes a distinct instrument colour." }
    ],
    "modes": [
      { "value": "timed", "label": "Timed", "description": "Each token's notes play for a fixed duration derived from the BPM setting, then stop." },
      { "value": "sustain", "label": "Sustain", "description": "Notes hold until the next token arrives, creating overlapping, drone-like textures." }
    ]
  }
  ```

### Step 2 — Add `bpm` to `PipelineParams` in `session.py`

- Add field: `bpm: int = 120` (default 120 BPM = 0.5 s/token, matching existing behaviour).
- Ensure `update()` handles int coercion for the field.

### Step 3 — Update `stream.py` to emit loop count and use bpm

- In `_run_pipeline`, add a `loop_count` integer starting at 0.
- On each loop iteration, increment `loop_count` before replaying events.
- Include `loop_count` in every replayed `token` event payload:
  ```json
  { "type": "token", ..., "loop_count": 2 }
  ```
- In the loop replay, use `60 / params.bpm` (instead of hardcoded `elapsed_ms / 1000`) as the sleep interval between replayed events:
  ```python
  await asyncio.sleep(60 / params.bpm)
  ```

### Step 4 — Rebuild `index.html`

Replace the controls `<aside>` content:

- **Model section**: `<select id="model">` populated by JS.
- **Layer section**: `<select id="layer">` populated by JS based on selected model.
- **Width section**: `<select id="width">` populated by JS based on selected model.
- **Strategy section**: `<select id="strategy">` with a `<span class="help-icon" data-tooltip="...">?</span>` tooltip trigger. The descriptions come from the API response.
- **Clusters section**: `<div id="clusters-group" class="hidden">` containing `<input id="clusters" type="number" min="2" max="128" value="8" />`. Shown only when strategy=cluster.
- **Max tokens section**: `<input id="max-tokens" type="number" min="0" value="200" />` + `<span class="help-icon" data-tooltip="Set to 0 for unlimited tokens.">?</span>`.
- **Mode section**: `<select id="mode">` with tooltip trigger. Below it, `<div id="bpm-group" class="hidden">` containing `<input id="bpm" type="number" min="1" value="120" />` labelled "BPM" + `<span class="help-icon" data-tooltip="Beats per minute — controls how fast tokens are paced in timed mode. 60 BPM = 1 s/token, 120 BPM = 0.5 s/token.">?</span>`. Shown only when mode=timed.
- **Loop checkbox**: keep as-is.
- **Status bar**: expand to two lines — token count on top, loop counter `<span id="loop-count-display" class="hidden">Loop: 0</span>` below.

### Step 5 — Update `style.css`

- Add `.hidden { display: none !important; }` utility.
- Add `.help-icon` styles: inline circle badge, cursor help, accent colour.
- Add `.tooltip-wrapper` / `[data-tooltip]` CSS for a hover popup (pure CSS, no JS library):
  ```css
  .help-icon {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: var(--border);
    color: var(--text-muted);
    font-size: 10px;
    cursor: help;
    flex-shrink: 0;
  }
  .help-icon::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    width: 220px;
    padding: 8px 10px;
    background: #222244;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 11px;
    line-height: 1.4;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    white-space: normal;
    z-index: 100;
  }
  .help-icon:hover::after { opacity: 1; }
  ```
- Style `input[type="number"]` consistently with the existing `select` style.
- Add `.loop-counter` style: small muted badge aligned right in the status area.

### Step 6 — Rewrite relevant sections of `main.js`

- **`loadOptions()`** (new async function, replaces `loadDefaults()`):
  1. Fetch `/api/config/model-options` → store `catalogue` and descriptions.
  2. Populate `#model` select from `models` list.
  3. Call `populateLayerWidth(selectedModel)`.
  4. Populate `#strategy` select from `strategies` list; set `data-tooltip` on the help icon from `description`.
  5. Populate `#mode` select from `modes` list; set `data-tooltip` on the help icon.
  6. Fetch `/api/config/defaults` → call `applyParams(d)`.

- **`populateLayerWidth(modelId)`** (new function):
  - Clear and repopulate `#layer` and `#width` selects from `catalogue[modelId].layers` and `.widths`.
  - Select the first option by default.

- **`applyParams(p)`** — add handling for `model`, `bpm`.

- **`collectParams()`** — add `model`, `bpm` (parse as integer from the BPM input value).

- **Conditional visibility wiring**:
  - `#strategy` `change` event → toggle `#clusters-group.hidden` based on value === "cluster".
  - `#mode` `change` event → toggle `#bpm-group.hidden` based on value === "timed".
  - `#model` `change` event → call `populateLayerWidth(this.value)`; send `update_params` if running.

- **`handleMessage`** — in `case "token"`: read `msg.loop_count` if present, update `#loop-count-display` and un-hide it.

- **Initial state**: hide `#clusters-group` and `#bpm-group` if their respective parent control isn't selected.

## Testing Strategy

### Unit Tests
- No new server-side unit tests required (logic is minimal config data + field addition).
- The `/api/config/model-options` response shape can be validated with a simple pytest HTTP test if desired.

### Integration Tests
- Smoke test: start server, connect WebSocket, send `start` with `bpm=60`, verify loop replay sleeps ~1 s between events.
- Loop test: send `start` with `loop=true`, wait for several `token` events, verify `loop_count` increments.

### Edge Cases
- Selecting a model that has only one layer/width: dropdowns should show a single disabled-looking option.
- Setting `max_tokens = 0`: server must not crash (the pipeline should run without a token cap); verify `PipelineParams.update` coerces string "0" to int 0 correctly.
- `bpm` set to an extremely high value (e.g. 9999): loop replay should not spin-lock; consider a maximum cap of 600 BPM server-side (= 0.1 s/token floor).
- Changing model mid-run: layer/width dropdowns repopulate; `update_params` sends new model to server.

## Acceptance Criteria

- [ ] Model, Layer, Width are `<select>` dropdowns populated from `/api/config/model-options`; changing model updates layer/width options.
- [ ] Only one model choice exists currently (`google/gemma-3-1b-pt`) with layer `[22]` and width `["65k"]`.
- [ ] Strategy is a dropdown; hovering the `?` icon shows a plain-English description of the selected strategy.
- [ ] When strategy = "cluster", a numeric input for cluster count is visible; it is hidden otherwise.
- [ ] Max tokens is a numeric input; hovering its `?` shows "Set to 0 for unlimited tokens."
- [ ] Mode is a dropdown with a `?` tooltip describing timed vs sustain.
- [ ] When mode = "timed", a BPM integer input is visible with a tooltip explaining the BPM → duration conversion; hidden otherwise.
- [ ] Loop counter display appears and increments correctly when loop mode is active.
- [ ] `collectParams()` includes all new fields (`model`, `bpm`).
- [ ] No regressions: Start/Stop, live param updates, waveform canvas rendering all work as before.

## Validation Commands

```bash
# 1. Syntax-check the server
cd /home/apprentyr/projects/responsible-ai-sensification
uv run python -c "from app.server.main import app; print('OK')"

# 2. Check the config endpoint returns the new shape
uv run uvicorn app.server.main:app --port 8001 &
sleep 2
curl -s http://localhost:8001/api/config/model-options | python3 -m json.tool
curl -s http://localhost:8001/api/config/defaults | python3 -m json.tool
kill %1

# 3. Validate HTML parses
python3 -c "
from html.parser import HTMLParser
class V(HTMLParser): pass
V().feed(open('app/client/index.html').read())
print('HTML OK')
"

# 4. Validate JS parses (node required)
node --input-type=module < app/client/main.js 2>&1 | head -5 || node -e "require('fs').readFileSync('app/client/main.js','utf8'); console.log('JS OK')"

# 5. Run server tests
uv run pytest
```

## Notes

- The `MODEL_CATALOGUE` in `config.py` is the single source of truth for available SAE configurations. When new SAE weights are downloaded for a new model/layer/width, only this dict needs updating — the frontend adapts automatically.
- `bpm` is an integer in the UI and on the server. The server converts it to a sleep duration as `60 / bpm` seconds. Default 120 BPM preserves the previous 0.5 s/token behaviour.
- The loop counter resets to 0 each time the user clicks Start.
- No new npm/pip packages are required; the tooltip is implemented in pure CSS using `::after` pseudo-elements.
- Future: `MODEL_CATALOGUE` could be generated dynamically by scanning the local HuggingFace cache for downloaded SAE files instead of being hardcoded.
