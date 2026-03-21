# Feature: Live SAE Feature Inspection

## Feature Description
Add an `inspect_live` function to `main.py` that runs a text prompt through a Gemma 3 model and a Gemma Scope 2 JumpReLU SAE, extracts live feature activations at a target residual stream layer, and prints a minimal CLI report showing which sparse features are most active and where they fire across the token sequence.

This enables mechanistic interpretability research directly from the terminal without relying on static datasets, external visualization libraries, or a Jupyter environment.

## User Story
As an AI interpretability researcher,
I want to pass a text prompt and immediately see which SAE features activate and where across the token sequence,
So that I can investigate model behavior at a mechanistic level using live activations on any arbitrary input.

## Problem Statement
The current `main.py` is a rough work-in-progress that only loads models and runs text generation to verify dependencies. There is no way to inspect internal activations or identify which SAE features fire for a given input.

## Solution Statement
Replace the existing `main.py` content with a clean inspection pipeline:
1. A `JumpReluSAE` module implementing the JumpReLU encoding formula.
2. A `load_sae()` helper downloading and instantiating the SAE from `google/gemma-scope-2-1b-pt`.
3. A `get_residual_stream()` helper capturing activations at a target layer via a one-shot forward hook.
4. An `inspect_live(prompt, model, tokenizer, sae, layer, top_k=10)` function running the full pipeline and printing a minimal CLI report.
5. A `__main__` driver block with hardcoded defaults.

## Relevant Files

- **`main.py`** — The only source file. Fully rewritten with the inspection pipeline. Old generation code is removed entirely.
- **`pyproject.toml`** — Dependency manifest. No new dependencies needed (all required libs already present: `transformers`, `huggingface-hub`, `safetensors`, `torch`, `numpy`).
- **`Tutorial_Gemma_Scope_2.ipynb`** — Reference for SAE loading conventions, JumpReLU encoding formula, and confirmed checkpoint path defaults (layer=22, width="65k", l0="medium").

### New Files
- **`specs/live-sae-inspection.md`** — This plan file (already created).

## Implementation Plan

### Phase 1: Foundation
- Rewrite `main.py` imports to only include what is needed: `transformers`, `huggingface_hub`, `safetensors.torch`, `torch`, `torch.nn`, `numpy`.
- Add `device = "cuda" if torch.cuda.is_available() else "cpu"` immediately after imports.
- Keep `torch.set_grad_enabled(False)` for memory efficiency.

### Phase 2: Core Implementation
- Define `JumpReluSAE` with the JumpReLU `encode` method.
- Write `load_sae()` downloading from `google/gemma-scope-2-1b-pt`.
- Write `get_residual_stream()` using a one-shot forward hook.
- Write `inspect_live()` with pure CLI print output.

### Phase 3: Integration
- Add a `__main__` driver block loading model, tokenizer, and SAE with hardcoded defaults, then calling `inspect_live` on a hardcoded sample prompt.

## Step by Step Tasks

### Step 1: Rewrite main.py imports and device setup
- Replace the entire import block with only what is needed:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from huggingface_hub import hf_hub_download
  from safetensors.torch import load_file
  import torch
  import torch.nn as nn
  import numpy as np
  ```
- Add immediately after:
  ```python
  torch.set_grad_enabled(False)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```

### Step 2: Define JumpReluSAE module
- Define `class JumpReluSAE(nn.Module)` with:
  - `__init__(self, w_enc, b_enc, threshold, w_dec, b_dec)` storing tensors as `nn.Parameter`.
  - `encode(self, x)` implementing:
    ```python
    pre_acts = x @ self.w_enc + self.b_enc
    mask = pre_acts > self.threshold
    acts = mask * torch.relu(pre_acts)
    return acts
    ```

### Step 3: Write load_sae helper
- Implement `load_sae(layer=22, width="65k", l0="medium", category="resid_post", device=device) -> JumpReluSAE`:
  - Construct checkpoint path: `f"{category}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"`.
  - Call `hf_hub_download(repo_id="google/gemma-scope-2-1b-pt", filename=path)` to get local path.
  - Load with `load_file(local_path)` from `safetensors.torch`.
  - Instantiate `JumpReluSAE` from the loaded tensors, move to `device`, set to `eval()`.

### Step 4: Write get_residual_stream helper
- Implement `get_residual_stream(model, layer_idx, input_ids) -> torch.Tensor`:
  - Use a closure with a list `captured = []`.
  - Register hook on `model.model.layers[layer_idx]` that appends `output[0].detach()` and removes itself.
  - Call `model(input_ids)`.
  - Return `captured[0].squeeze(0)` — shape `(seq_len, d_model)`.

### Step 5: Write inspect_live function
- Implement `inspect_live(prompt, model, tokenizer, sae, layer, top_k=10)`:

  **Tokenize:**
  ```python
  inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  input_ids = inputs["input_ids"].to(device)
  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  ```

  **Extract residual stream and encode:**
  ```python
  residual = get_residual_stream(model, layer, input_ids)   # (seq_len, d_model)
  sae_acts = sae.encode(residual.float())                   # (seq_len, d_sae)
  ```

  **Skip BOS (index 0):**
  ```python
  tokens_nobos = tokens[1:]
  acts_nobos = sae_acts[1:]                                 # (seq_len-1, d_sae)
  ```

  **Top-K features by mean activation:**
  ```python
  mean_acts = acts_nobos.mean(dim=0)                        # (d_sae,)
  topk_indices = mean_acts.topk(top_k).indices              # (top_k,)
  ```

  **Print top-K feature summary:**
  ```
  === Top-10 Active Features (layer 22) ===
    #1  Feature  12345  mean_act=3.1420
    #2  Feature   8901  mean_act=2.7183
    ...
  ```

  **Per-token activations for top-K features (top 10% tokens only):**
  - For each top-K feature, compute per-token activations and select the top 10% tokens by activation value (i.e. `ceil(0.1 * seq_len)` tokens with highest activation, minimum 1).
  - Print only those tokens with their activation value:
    ```
    Feature 12345:
        ▸ "energy"   : 8.4521
        ▸ "conserv"  : 6.1203
    ```
  - Skip printing the feature block entirely if all its activations are 0.

  **L0 per token:**
  ```python
  l0_per_token = (acts_nobos > 0).sum(dim=1).tolist()
  ```
  Print as:
  ```
  === L0 per token (excluding BOS) ===
      "The"       : 42
      "law"       : 37
      ...
  ```

  **Summary:**
  ```
  Average L0 (non-BOS): 39.50
  ```

### Step 6: Write __main__ driver block
- Under `if __name__ == "__main__":`:
  - Load `model` and `tokenizer` from `"google/gemma-3-1b-pt"` with `device_map="auto"`.
  - Call `load_sae(layer=22, width="65k", l0="medium", device=device)` to get `sae`.
  - Hardcode a sample prompt:
    ```python
    prompt = "The law of conservation of energy states that energy cannot be created or destroyed."
    ```
  - Call `inspect_live(prompt, model, tokenizer, sae, layer=22, top_k=10)`.

### Step 7: Validate
- Run the validation commands listed below to confirm correctness.

## Testing Strategy

### Unit Tests
No formal test suite. Manual smoke test via `python main.py`.

### Integration Tests
- Run `inspect_live` with the hardcoded prompt and confirm:
  - Top-K feature summary prints without error.
  - Per-token block prints only for features with non-zero activations.
  - L0 list has length equal to `seq_len - 1`.
  - Average L0 is a positive float.

### Edge Cases
- **Single-token prompt:** After BOS removal only 1 token remains — `ceil(0.1 * 1) = 1`, so 1 token is printed; no indexing errors.
- **All-zero SAE activations for a feature:** Skip printing that feature's block entirely.
- **CPU-only machine:** `device = "cpu"` — no `.cuda()` calls anywhere.
- **`top_k` larger than number of active features:** `mean_acts.topk(top_k)` returns `top_k` indices including zero-activation ones; zero-activation feature blocks are silently skipped in output.

## Acceptance Criteria
- `uv run python main.py` runs end-to-end without error.
- Prints exactly `top_k` lines in the feature summary.
- Prints per-token activations (top 10% tokens) for each top-K feature that has non-zero activations.
- Prints L0 count for every non-BOS token.
- Prints average L0 as a float.
- No hardcoded `"cuda"` references — a `device` variable controls all device placement.
- No external visualization libraries (plotly, matplotlib, IPython) used anywhere.
- No old generation code remains.

## Validation Commands
- `cd /home/apprentyr/projects/responsible-ai-sensification && uv run python -c "import main; print('imports OK')"` — Confirm module imports without error
- `cd /home/apprentyr/projects/responsible-ai-sensification && uv run python -c "from main import JumpReluSAE, load_sae, get_residual_stream, inspect_live; print('symbols OK')"` — Confirm all symbols are importable
- `cd /home/apprentyr/projects/responsible-ai-sensification && uv run python main.py` — Run the full pipeline end-to-end

## Notes
- **Gemma vs Gemma Scope versioning:** `google/gemma-scope-2-1b-pt` (Gemma Scope **2**) is the correct SAE repo for `google/gemma-3-1b-pt`. The "2" in Gemma Scope 2 refers to the second SAE release, not the Gemma model generation. The SAEs were trained on Gemma 3 residual streams.
- **`l0="medium"` is a literal string** in the checkpoint path — confirmed from the tutorial notebook. It is not a numeric placeholder.
- **Layer default is 22**, not 20 — confirmed from the tutorial notebook (`resid_post/layer_22_width_65k_l0_medium`).
- **`width="65k"` is a string** in the checkpoint path, not an integer.
- No new dependencies are required. `ipython` is not needed since output is plain CLI.
- The `BitsAndBytesConfig`, `einops`, `plotly`, `pandas`, `partial`, `dataclasses`, `gc`, `Literal`, and `textwrap` imports from the original `main.py` are all removed.
