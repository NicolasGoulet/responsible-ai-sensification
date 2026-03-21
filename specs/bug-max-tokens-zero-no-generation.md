# Bug: max_tokens=0 Generates Zero Tokens Instead of Unlimited

## Bug Description

The UI has a tooltip on the "Max tokens" field that reads: *"Set to 0 for unlimited tokens."* When the user sets max_tokens to 0 (as advertised), nothing is generated. The server logs confirm:

```
[pipeline] Generation starting (max_tokens=0)...
[inspect_live] Starting generation loop (max_new_tokens=0, input_len=4)
[pipeline] Generation complete: 0 tokens.
```

`range(0)` is an empty iterator — no tokens are produced, the pipeline immediately completes with "done", and the browser shows nothing.

The 0 value is also persisted to localStorage, so every subsequent run (even after page reload) uses max_tokens=0 unless the user manually changes the field.

**Expected:** max_tokens=0 means "generate until EOS (unlimited)".
**Actual:** max_tokens=0 means `range(0)` → empty loop → 0 tokens generated.

## Problem Statement

`inspect_live` in `extract.py` uses `range(max_new_tokens)` as the generation loop bound. When `max_new_tokens=0`, the range is empty and the loop body never executes. The tooltip in `index.html` documents 0 as the "unlimited" sentinel, but the implementation never handles this case.

## Solution Statement

In `inspect_live`, treat `max_new_tokens=0` as unlimited by replacing `range(max_new_tokens)` with a conditional: use `range(max_new_tokens)` when `max_new_tokens > 0`, and `itertools.count()` (an infinite iterator) when `max_new_tokens == 0`. The EOS-token break inside the loop already handles termination in the unlimited case.

This is a single-line change in one function. No other files need to change.

## Steps to Reproduce

1. Open the browser at `http://localhost:8080`
2. Set "Max tokens" to `0` (or note it is already 0 from a prior save)
3. Click **Start**
4. Observe: immediately shows "Done (0 tokens)" — no audio, no visualisation

## Root Cause Analysis

In `extract.py::inspect_live`:

```python
for step in range(max_new_tokens):   # range(0) == empty
    ...
    if next_token_id == tokenizer.eos_token_id:
        break
```

`range(0)` produces no iterations. The function returns immediately, yielding nothing.

The tooltip in `index.html` (`data-tooltip="Set to 0 for unlimited tokens."`) was written with the intent that 0 would be handled as a sentinel for "no limit", but the implementation in `inspect_live` was never updated to match.

The value 0 is also written to `localStorage` via `saveParams()` whenever `max_tokens=0` is collected from the input, so the bad value persists across page reloads.

## Relevant Files

- **`app/server/pipeline/extract.py`** — contains `inspect_live`; the only file that needs changing. Replace `range(max_new_tokens)` with `itertools.count()` when `max_new_tokens == 0`.

## Step by Step Tasks

### 1. Fix `inspect_live` in `app/server/pipeline/extract.py`

- Add `import itertools` at the top of the file (alongside existing standard library imports).
- In `inspect_live`, replace:
  ```python
  for step in range(max_new_tokens):
  ```
  with:
  ```python
  _steps = range(max_new_tokens) if max_new_tokens > 0 else itertools.count()
  for step in _steps:
  ```
- No other change needed. The `if next_token_id == tokenizer.eos_token_id: break` already handles termination in the unlimited case.

### 2. Validate the fix

Run validation commands listed below.

## Validation Commands

```bash
# Start the server
./scripts/start.sh --verbose

# Reproduce the bug scenario: open http://localhost:8080, set max_tokens=0, click Start
# Expected: generation runs until EOS and tokens stream in
# Confirm in server logs:
#   [pipeline] Generation starting (max_tokens=0)...
#   [inspect_live] Starting generation loop (max_new_tokens=0, input_len=N)
#   [inspect_live] First forward pass complete.
#   ... tokens streaming ...
#   [pipeline] Generation complete: N tokens.   ← N > 0

# Also confirm max_tokens>0 still works correctly: set to 50, click Start
# Expected: exactly 50 tokens generated (or fewer if EOS encountered first)

# Run server tests
cd app/server && uv run pytest
```

## Notes

- `itertools.count()` is a builtin from the Python standard library — no new dependency.
- The EOS break `if next_token_id == tokenizer.eos_token_id: break` ensures the unlimited loop terminates naturally when the model generates an end-of-sequence token.
- The localStorage value of 0 will remain, but it will now work correctly. Users who previously hit this bug will see generation start working on their next click without needing to clear localStorage.
- The tooltip `"Set to 0 for unlimited tokens."` in `index.html` is already correct and does not need to change.
