# Feature: Web Audio API — Browser-side Audio Playback

## Feature Description
Replace server-side `sounddevice` playback with in-browser additive synthesis using the Web Audio API. Each `token` WebSocket event already carries a `notes` array (freq + amplitude + instrument); the browser will reconstruct the synthesis logic from `audio_utils.py` using `OscillatorNode`s per partial, `GainNode`s for amplitude, and instrument-specific waveform configurations. Two playback modes are supported: `timed` (fixed duration per token based on BPM) and `sustain` (hold note until the next token arrives). The server-side `sounddevice`/`soundfile` packages are only used in the CLI `synthesize.py` tool and are not in the web pipeline path, so no server changes are required.

## User Story
As a user listening to the sensification in the browser,
I want audio to play directly from the browser synthesizing each token's notes,
So that I can hear the SAE-driven soundscape without needing a server-side audio device.

## Problem Statement
Currently the browser visualizes notes on a canvas but produces no audio. The server-side `synthesize.py` handles audio rendering/playback for the CLI pipeline only. The web frontend needs to replicate the additive synthesis from `audio_utils.py` using the Web Audio API.

## Solution Statement
Add a `WebAudioEngine` class in `main.js` that:
1. Holds a singleton `AudioContext` (created on first user gesture via the Start/Send button to comply with browser autoplay policy).
2. On each `token` message, calls `playNotes(notes, mode, bpm)` which schedules or triggers oscillator networks for each note.
3. Mirrors the per-instrument harmonic structure from `audio_utils._synthesize_note` using `OscillatorNode` + `GainNode` sub-graphs.
4. In `timed` mode: schedules notes to play for `60/bpm` seconds then stop.
5. In `sustain` mode: starts notes immediately and stops the previous set when the next token arrives.
6. Adds a master `GainNode` for global volume and normalizes per-token amplitude across all notes.

## Relevant Files

- **`app/client/main.js`** — Main frontend logic; receives token events and drives canvas rendering. The Web Audio engine and `handleMessage` integration go here.
- **`app/client/index.html`** — Contains the Start/Send buttons (user gestures needed to unlock `AudioContext`) and the waveform canvas. No structural changes needed; possibly add a volume control.
- **`app/client/style.css`** — May need a minor style addition for a volume slider if added.
- **`app/server/pipeline/audio_utils.py`** — Reference for the exact harmonic partial ratios and envelope shapes to replicate in JS.
- **`specs/TODO.md`** — Remove completed item 1, keep item 2 as in-progress context.

### New Files
_(none — all changes go into existing client files)_

## Implementation Plan

### Phase 1: Foundation
- Create a `WebAudioEngine` class that manages an `AudioContext`, a master gain node, and a set of currently active oscillators.
- Expose `resume()` (called on user gesture), `playNotes(notes, durationSec)`, `stopAll()`, and `setVolume(v)` methods.

### Phase 2: Core Implementation
- Implement per-instrument oscillator networks mirroring `audio_utils._synthesize_note`:
  - **piano**: sine(f) + 0.5·sine(2f)
  - **guitar**: sine(f) + 0.3·sine(2f) + 0.2·sine(3f)
  - **bass**: sine(f) + 0.6·sine(f/2)
  - **strings**: sine(f) + sine(f+2 Hz) (chorus-style detuning)
  - **pad**: sine(f) with sin-curve gain envelope over duration
  - **bell**: [sine(f) + 0.4·sine(2f) + 0.2·sine(5f)] × exponential decay
  - **flute**: sine(f) only
  - **brass**: sine(f) + 0.7·sine(2f) + 0.5·sine(3f) + 0.3·sine(4f)
  - **default/unknown**: sine(f)
- Normalize amplitudes across all notes in the token (same as `maxAmp` logic in `drawNotes`).
- Implement `timed` mode: schedule all oscillators to start at `audioCtx.currentTime` and stop at `+durationSec`.
- Implement `sustain` mode: call `stopAll()` first, then start new oscillators with no scheduled stop.

### Phase 3: Integration
- Call `audioCtx.resume()` inside the Start/Send button click handlers (user gesture unlock).
- Call `engine.playNotes(msg.notes, mode, bpm)` inside `handleMessage` for the `token` case.
- Call `engine.stopAll()` on `stopped`, `silent`, and `done` events (for sustain mode).
- Add an optional volume slider to `index.html` wired to `engine.setVolume()`.

## Step by Step Tasks

### Step 1: Update TODO.md
- Remove feature 1 (localStorage) from `specs/TODO.md` since it is implemented.

### Step 2: Implement `WebAudioEngine` in `main.js`
- Add the `WebAudioEngine` class above the existing state block.
- Constructor: create `AudioContext` lazily on first `resume()` call to avoid the autoplay-policy error before any user gesture.
- `resume()`: create `AudioContext` if null, call `.resume()`, create master `GainNode` at gain 1.0 connected to `destination`.
- `_buildNoteGraph(freq, amplitude, instrument, startTime, stopTime)`:
  - Returns an array of `{ osc, gain }` pairs connected to the master gain.
  - Each partial: `createOscillator()` (type `"sine"`) → `createGain()` → masterGain.
  - Set `osc.frequency.value`, `gain.gain.value`, call `osc.start(startTime)`.
  - If `stopTime` is defined, call `osc.stop(stopTime)`.
  - For **pad**: schedule gain from 0 → amplitude → 0 over `[startTime, stopTime]` using `linearRampToValueAtTime`.
  - For **bell**: schedule gain from amplitude → ~0 using `exponentialRampToValueAtTime` over the duration.
  - For **strings**: second oscillator at `freq + 2` Hz.
- `playNotes(notes, mode, bpmOrNull)`:
  - If `!audioCtx` return early (AudioContext not yet unlocked).
  - Calculate `maxAmp = Math.max(...notes.map(n => n.amplitude), 1)`.
  - `durationSec = mode === "timed" ? 60 / bpm : null`.
  - `startTime = audioCtx.currentTime`.
  - `stopTime = durationSec ? startTime + durationSec : null`.
  - If `mode === "sustain"`, call `stopAll()` first.
  - For each note, call `_buildNoteGraph(note.freq, note.amplitude / maxAmp, note.instrument, startTime, stopTime)` and push returned nodes to `_activeNodes`.
- `stopAll()`:
  - For each active oscillator call `osc.stop(audioCtx.currentTime)` wrapped in try/catch.
  - Clear `_activeNodes`.
- `setVolume(v)`: set `masterGain.gain.value = v` if masterGain exists.

### Step 3: Wire engine into event handlers in `main.js`
- Instantiate `const engine = new WebAudioEngine();` in the state block.
- In `startPipeline()`: call `engine.resume()` before sending the WebSocket message.
- In `handleMessage`:
  - `token` case: after `drawNotes(msg.notes)`, add `engine.playNotes(msg.notes ?? [], modeSel.value, parseInt(bpmIn.value))`.
  - `stopped`, `silent` cases: add `engine.stopAll()`.
- In `btnStop` click handler: add `engine.stopAll()`.

### Step 4: Add volume control to `index.html`
- After the loop checkbox section, add a `<section class="control-group">` with a range input (`id="volume"`, min=0, max=1, step=0.01, value=0.7).
- Wire it in `main.js`: `volumeIn.addEventListener("input", () => engine.setVolume(parseFloat(volumeIn.value)))`.
- Initialize engine volume after `engine.resume()` using the slider value.

### Step 5: Add minor CSS for volume control in `style.css`
- Ensure `input[type=range]` has appropriate styling (likely already covered by existing styles; verify).

### Step 6: Validate

## Testing Strategy

### Unit Tests
No automated JS unit tests in this project. Manual testing is the validation path.

### Integration Tests
- Open browser DevTools → Console: no errors on page load.
- Click Start: `AudioContext` state becomes `"running"` (verify in console).
- Token events received: audio plays with correct timing relative to BPM.
- Switching mode mid-run: sustain holds until next token; timed stops after `60/bpm` seconds.
- Stop button: all oscillators stop immediately.

### Edge Cases
- Page load without clicking Start: `AudioContext` must NOT be created (autoplay policy).
- `notes` array empty (silent token): `playNotes([])` should be a no-op with no errors.
- Very high frequency notes (>18 kHz): oscillators may be inaudible but must not error.
- Very low amplitude notes (≈0): gain near zero; must not produce NaN or Infinity.
- `bell`/`pad` envelope with `stopTime = null` (sustain mode): skip envelope automation; play flat amplitude until `stopAll()`.
- `bpm` at extremes (1 BPM → 60 s, 600 BPM → 0.1 s): oscillators scheduled correctly.
- Rapid Start→Stop→Start: prior oscillators cleaned up, no zombie nodes accumulating.

## Acceptance Criteria
- Clicking Start then generating tokens produces audible sound in the browser.
- `timed` mode: each token's notes play for exactly `60/bpm` seconds then stop automatically.
- `sustain` mode: notes from the previous token stop the moment the new token's notes start.
- All eight instrument timbres are audibly distinct (verifiable by sending single-instrument prompts).
- Stop button immediately silences all active oscillators.
- No AudioContext-related console errors on page load or during normal use.
- Volume slider adjusts output loudness in real time.
- Existing canvas waveform visualization continues to work unchanged.

## Validation Commands
- `cd app/server && uv run pytest` — Run server tests to validate zero regressions.
- Open browser DevTools Console after page load: zero errors before clicking Start.
- Click Start, observe `AudioContext.state === "running"` in DevTools → Application → (or log it).
- Run with `mode=timed`, `bpm=60`: each token audibly plays for ~1 second.
- Run with `mode=sustain`: audio changes immediately on each new token, no gap.
- Click Stop mid-run: audio cuts immediately.

## Notes
- The `sounddevice`/`soundfile` packages are only used by the CLI `synthesize.py` tool. The web streaming path in `stream.py` never imports them, so no server dependency changes are needed for this feature.
- The Web Audio API `AudioContext` autoplay policy requires a user gesture before `.resume()` will succeed. Always call `engine.resume()` inside a button click handler, never automatically on page load.
- Web Audio oscillators are extremely lightweight; hundreds can run simultaneously without issue in modern browsers.
- The `strings` instrument uses a 2 Hz detuned second oscillator to create a chorus/beating effect — this is a direct translation of `np.sin(2π(f+2)t)` from `audio_utils.py`.
- The `pad` envelope uses `gain.gain.linearRampToValueAtTime` to approximate the `np.sin(πt/TOKEN_DURATION)` shape. In sustain mode, the envelope is skipped (flat amplitude) since the duration is unknown.
- The `bell` envelope uses `gain.gain.exponentialRampToValueAtTime` to approximate `np.exp(-3t/TOKEN_DURATION)`. In sustain mode, a modest fixed decay can be used instead (e.g., decay to 10% over 2 seconds).
