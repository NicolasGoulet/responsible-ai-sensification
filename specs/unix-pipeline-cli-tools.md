# Feature: Unix Pipeline CLI Tools

## Feature Description
Refactor the existing Python scripts into focused, composable CLI tools following the Unix philosophy: each tool does one thing well and can be chained via pipes or JSON files. `main.py` becomes `extract.py` — a pure data extraction tool that outputs raw JSON. `synthesize.py` gains a `--method` flag for selecting synthesis strategies. This lays the groundwork for a future streaming server pipeline where prompt → extract → transform → synthesize are independent, replaceable stages.

## User Story
As a researcher / artist collaborator
I want to run `uv run python extract.py "my prompt" --layer 22 --width 65k` and pipe the result to `uv run python synthesize.py --method additive`
So that I can iterate on different models, layers, widths, and synthesis methods without reloading the model or coupling stages together

## Problem Statement
Currently `main.py` hardcodes MODEL_ID, LAYER, WIDTH, and the prompt, mixes data extraction with display logic, and calls export as a side-effect. `synthesize.py` has no method selection. The two scripts are not composable — to change a parameter you edit the source, and the synthesis method is invisible/implicit. This makes experimentation slow and blocks the path toward a streaming multi-stage pipeline.

## Solution Statement
- Rename `main.py` → `extract.py`. All model/SAE parameters become CLI flags. The prompt is a positional argument. Output is raw JSON only (to a file or stdout). No display/print logic in the extraction path — it runs silently unless `--verbose` is passed.
- `synthesize.py` gains `--method <name>` to select the synthesis strategy. The current sine-wave additive approach becomes `--method additive` (the default). New methods can be registered without touching the CLI wiring.
- `export.py` is kept as-is (pure library utility used by `extract.py`).
- A note clarifies the future split: today `synthesize.py` conflates *transformation* (feature index → musical parameter) with *synthesis* (musical parameter → audio samples). A future `transform.py` will sit between them.

## Relevant Files

- `main.py` — source file to be renamed/replaced by `extract.py`. Contains model loading, SAE loading, Neuronpedia download, and `inspect_live()`. The `__main__` block becomes a proper argparse CLI.
- `export.py` — unchanged library used by `extract.py` to serialise `GenerationAnalysis` to JSON. Import path changes because the caller file is renamed.
- `synthesize.py` — gains `--method` flag and a dispatch table so new methods can be added. Current logic moves into a named strategy function `synthesize_additive()`.

### New Files
- `extract.py` — replaces `main.py`. Pure extraction CLI: loads model + SAE + Neuronpedia, runs `inspect_live()`, writes raw JSON. No audio, no display beyond optional progress lines.

## Implementation Plan

### Phase 1: Foundation
Ensure the internal logic (model loading, SAE, Neuronpedia, generation) is cleanly separated from the `__main__` entry point so it can be imported without side-effects. Move all Pydantic models and helper functions to a shared module if needed (not strictly required for this phase — keeping everything in `extract.py` is fine for now).

### Phase 2: Core Implementation
- Create `extract.py` with argparse CLI (positional `prompt`, flags for `--model`, `--layer`, `--width`, `--l0`, `--max-tokens`, `--output`, `--verbose`).
- Update `synthesize.py` to dispatch on `--method`. Move existing logic into `synthesize_additive()`. Add a `METHODS` registry dict.

### Phase 3: Integration
- Delete `main.py` (its logic now lives in `extract.py`).
- Update `export.py` import (it imports from `main` — change to `extract` after rename).
- Verify the full pipeline: `extract.py "prompt" | synthesize.py` (via file or stdin).

## Step by Step Tasks

### Step 1: Create `extract.py` from `main.py`
- Copy all content of `main.py` into a new file `extract.py`.
- In `extract.py`, replace the hardcoded `__main__` block with an `argparse` CLI:
  - Positional argument: `prompt` (str)
  - `--model` (default: `google/gemma-3-1b-pt`)
  - `--layer` (int, default: `22`)
  - `--width` (str, default: `65k`)
  - `--l0` (str, default: `medium`)
  - `--max-tokens` (int, default: `200`)
  - `--output` (Path, default: `runs/analysis.json`)
  - `--verbose` (flag, suppresses prints when absent)
- Replace the hardcoded `model_id` string inside `inspect_live()` with the passed `model_id` argument (currently hardcoded to `"google/gemma-3-1b-pt"` at line 256).
- The script should only print progress lines when `--verbose` is set; otherwise it runs silently and writes the JSON.

### Step 2: Fix the `model_id` hardcode in `inspect_live`
- In `extract.py`, update `inspect_live()` signature to accept `model_id: str` as a parameter.
- Pass `model_id` into the `GenerationAnalysis` constructor instead of the hardcoded string.

### Step 3: Update `export.py` imports
- `export.py` currently imports `from main import GenerationAnalysis`.
- After creating `extract.py`, update that import to `from extract import GenerationAnalysis`.

### Step 4: Update `synthesize.py` with method dispatch
- Rename the existing synthesis logic block into a function `synthesize_additive(input_path, output_dir)`.
- Add a `METHODS` dict: `{"additive": synthesize_additive}`.
- Add `--method` flag to argparse (default: `additive`, choices from `METHODS.keys()`).
- Dispatch `METHODS[args.method](args.input, args.output_dir)` in `main()`.

### Step 5: Delete `main.py`
- Remove `main.py` now that all its logic lives in `extract.py`.
- Verify no other files import from `main` (only `export.py` did, already fixed in Step 3).

### Step 6: Update `README.md`
- Document the two tools and their flags.
- Show the basic pipeline example: `uv run python extract.py "prompt" && uv run python synthesize.py runs/analysis.json`.

### Step 7: Validate
- Run the validation commands below.

## Testing Strategy

### Unit Tests
No formal test suite exists yet. Manual smoke tests are the validation path for now.

### Integration Tests
End-to-end pipeline test: `extract.py` → JSON file → `synthesize.py` → WAV file. Verify JSON schema matches what `synthesize.py` expects.

### Edge Cases
- Prompt with special characters/quotes from shell.
- `--layer` or `--width` value not available in the SAE repo (will raise a download error — acceptable, no special handling needed now).
- `--max-tokens 1` (single token generation).
- `--output -` (stdout) — not required in this phase, but worth noting as a future extension.

## Acceptance Criteria
- `uv run python extract.py "The law of conservation" --layer 22 --width 65k` runs without error and writes a valid JSON file to `runs/analysis.json`.
- `uv run python extract.py "hello" --model google/gemma-3-1b-pt --layer 22 --width 65k --output runs/test.json` writes JSON to the specified path.
- `uv run python synthesize.py runs/analysis.json --method additive` produces a `.wav` file and is equivalent to the previous behaviour.
- `uv run python synthesize.py runs/analysis.json` (no `--method`) uses `additive` as the default.
- `main.py` no longer exists in the repo root.
- `export.py` imports from `extract`, not `main`.
- `--help` on both scripts shows all flags with descriptions.

## Validation Commands

```bash
# 1. Verify extract CLI help
uv run python extract.py --help

# 2. Run extraction with explicit flags (requires model download — skip in CI)
uv run python extract.py "The law of conservation of energy" --layer 22 --width 65k --output runs/test_cli.json --verbose

# 3. Verify JSON output has expected keys
python -c "import json; d=json.load(open('runs/test_cli.json')); assert 'generated_tokens' in d and 'model_id' in d, 'missing keys'"

# 4. Synthesize using explicit method flag
uv run python synthesize.py runs/test_cli.json --method additive --output-dir audio

# 5. Verify WAV was produced
ls audio/test_cli.wav

# 6. Synthesize using default (no --method) — should behave identically
uv run python synthesize.py runs/test_cli.json --output-dir audio

# 7. Confirm main.py is gone
test ! -f main.py && echo "main.py correctly removed"

# 8. Confirm export.py imports from extract
grep "from extract import" export.py
```

## Notes

**On the transform/synthesize split:** `synthesize.py` currently does two things at once — it *transforms* raw SAE feature data into musical parameters (feature index → frequency via log-scale mapping) and *synthesizes* audio from those parameters (sine wave generation). These are two distinct operations. A future `transform.py` tool will sit between `extract.py` and `synthesize.py` and output a standardised intermediate format (e.g. a list of timed `{frequency, amplitude}` events, or OSC/MIDI messages). `synthesize.py` will then only need to read that format and render audio. For this task, the split is not implemented — only the groundwork (method dispatch) is laid so that adding methods later is trivial.

**On standard formats for digital musical creation:** The natural targets for the transformation output are:
- **OSC (Open Sound Control)**: real-time UDP messages widely supported by SuperCollider, Max/MSP, Pure Data, Ableton + M4L — ideal for the streaming use case.
- **MIDI**: universally understood but limited resolution (128 pitches, 128 velocities). Useful for DAW compatibility.
- **JSON event stream** (custom): simplest for now, easy to consume in any environment.
The streaming server vision maps cleanly onto OSC: server generates tokens, each token fires a burst of OSC messages (one per active feature), artists receive the stream in their environment of choice.

**No new library needed for this phase.** `argparse` is stdlib. OSC/MIDI libraries (`python-osc`, `mido`) will be needed in the transform phase.
