# Feature: Persist UI Parameters Across Resets

## Feature Description
Save the current web UI parameter values (prompt, model, layer, width, strategy, clusters, max tokens, mode, BPM, loop) to `localStorage` whenever they change. On page load, restore saved values before applying server defaults — so user-set values are never lost on refresh or reconnect.

## User Story
As a user experimenting with the SAE sensification UI,
I want my parameter settings to persist across page reloads and WebSocket resets,
So that I don't have to re-enter my configuration every time I refresh or the server restarts.

## Problem Statement
Every page load resets all controls to server-side defaults. Users who tune parameters (prompt, BPM, strategy, clusters, etc.) lose their settings on any refresh or reconnect.

## Solution Statement
Use the browser `localStorage` API to persist the full params object. Save on every control-change event. Restore saved params at startup, before server defaults are applied — so user values win over defaults. No server changes are needed; no files are written to disk.

Since everything lives in the browser, there is nothing to add to `.gitignore`.

## Relevant Files

- **`app/client/main.js`** — all parameter collection, `applyParams`, and event listeners live here. This is the only file that needs to change.

## Implementation Plan
### Phase 1: Foundation
Define a `localStorage` key constant and two helpers: `saveParams()` (serialise `collectParams()` → JSON → `localStorage`) and `loadSavedParams()` (deserialise and return the object, or `null`).

### Phase 2: Core Implementation
1. Call `saveParams()` inside every control-change/input event listener (after any existing logic).
2. In `loadOptions()`, after options are populated and server defaults are applied, overlay any saved params so user values take precedence.

### Phase 3: Integration
Verify that the priority order is: server dropdown population → server defaults (`applyParams(d)`) → saved params (`applyParams(saved)`). This ensures saved params always win without breaking fresh-install behaviour.

## Step by Step Tasks

### Step 1 — Add storage helpers to main.js
- Add a constant `STORAGE_KEY = "sae_ui_params"` near the top (after the state block).
- Add `function saveParams() { localStorage.setItem(STORAGE_KEY, JSON.stringify(collectParams())); }`.
- Add `function loadSavedParams() { try { const raw = localStorage.getItem(STORAGE_KEY); return raw ? JSON.parse(raw) : null; } catch { return null; } }`.

### Step 2 — Save on every control change
- Append `saveParams()` to the handler of every event listener that calls `sendParamUpdate` or modifies a control:
  - `modelSel` change
  - `layerSel` change
  - `widthSel` change
  - `strategySel` change
  - `clustersIn` input
  - `maxTokensIn` input
  - `modeSel` change
  - `bpmIn` input
  - `loopCb` change
- Also save when the prompt changes: add an `input` listener on `prompt` that calls `saveParams()`.

### Step 3 — Restore saved params at startup
- At the end of `loadOptions()`, after `applyParams(d)` (server defaults), add:
  ```js
  const saved = loadSavedParams();
  if (saved) applyParams(saved);
  ```
  This ensures saved values override server defaults on every load.

### Step 4 — Validate
- Run the validation commands below.

## Testing Strategy
### Unit Tests
No JS unit-test framework is set up; validate manually via the browser console.

### Integration Tests
- Open the UI, change several parameters, refresh — all values should be restored.
- Open a fresh private/incognito window — server defaults should load (no saved params).

### Edge Cases
- `localStorage` unavailable (private browsing in some browsers): `loadSavedParams` returns `null` gracefully; `saveParams` silently fails (wrap in try/catch).
- Saved params contain a model ID that no longer exists in the catalogue: `applyParams` already guards with `if (p.model !== undefined) modelSel.value = p.model` — the select simply won't change if the value isn't in the list.
- Partial save (e.g. old version saved fewer keys): `applyParams` already skips `undefined` fields, so safe.

## Acceptance Criteria
- All control values (prompt, model, layer, width, strategy, clusters, max tokens, mode, BPM, loop) survive a page reload.
- A fresh browser session (no `localStorage`) loads server defaults without errors.
- No new files are created on disk; nothing needs to be gitignored.
- Existing WebSocket and parameter-update behaviour is unchanged.

## Validation Commands
```bash
# Syntax-check the JS file for obvious errors
node --check app/client/main.js

# Confirm localStorage calls are present
grep -n "localStorage" app/client/main.js

# Confirm saveParams is called from every event listener
grep -n "saveParams" app/client/main.js
```

## Notes
- `localStorage` is per-origin and per-browser; nothing is stored on the server or in git.
- If a future feature adds more controls, add `saveParams()` to their event listeners to stay consistent.
