# Feature: Token-by-Token SAE Inspection with Neuronpedia Enrichment

## Feature Description
Redesign the inspection pipeline so that instead of analyzing a static prompt in one shot, the model **generates tokens autoregressively** and, at each generation step, the residual stream at the target layer is captured, passed through the SAE, and all active features are recorded along with their Neuronpedia human-readable descriptions.

The output is a structured Pydantic object tree representing the full generation: which token was generated at each step, which SAE features fired (all of them, not a top-K), the activation value of each, and the L0 (total count of active features) per token.

A companion `explain_feature()` function renders any feature in human-readable text form.

Neuronpedia explanations for the loaded scope are downloaded automatically at startup and cached locally. An `list_available_scopes()` exploration helper lets callers discover what model/layer/width combinations exist on Neuronpedia before downloading.

## User Story
As an AI interpretability researcher or artist,
I want to run a prompt through the model, generate tokens, and receive a structured object with the full SAE feature activation data per generated token — enriched with human-readable concept labels from Neuronpedia —
So that I can feed that data into any downstream processing (music, images, analysis) without being constrained by a fixed visualization format.

## Problem Statement
The current `inspect_live` function mixes analysis and display logic, operates on a static prompt (no generation), selects only a top-K subset of features, and produces no structured output — just printed text. There is no link to Neuronpedia labels, no concept of time (token sequence during generation), and no reusable data structure.

## Solution Statement
Replace `inspect_live` with a generation-time inspection pipeline:

1. A set of **Pydantic models** capturing the full data structure.
2. A **Neuronpedia download layer**: `list_available_scopes()` for discovery, `download_neuronpedia_explanations()` for bulk download of all feature descriptions, with local `.jsonl` cache.
3. A rewritten **`inspect_live()`** that generates tokens one by one, hooks the residual stream at each step, encodes via SAE, and builds a `GenerationAnalysis` Pydantic object.
4. An **`explain_feature()`** function that takes an `ActiveFeature` and returns a human-readable string.

## Relevant Files

- **`main.py`** — The only source file. All new code goes here. The existing `inspect_live` and `get_residual_stream` functions are replaced; `JumpReluSAE` and `load_sae` are kept and extended.
- **`pyproject.toml`** — Dependency manifest. One new dependency needed: `requests` for S3 HTTP calls (see Notes).
- **`specs/live-sae-inspection.md`** — Previous spec, superseded by this plan.

### New Files
- **`specs/token-generation-sae-inspection.md`** — This plan file.
- **`neuronpedia_cache/`** — Directory created at runtime to store downloaded `.jsonl` explanation files (one per scope, e.g. `gemma-3-1b_22_65k.jsonl`).

## Implementation Plan

### Phase 1: Foundation — Pydantic models and Neuronpedia download layer
Define the data model first so every subsequent function has a clear return type contract. Then build the Neuronpedia download infrastructure so labels are available before any generation happens.

### Phase 2: Core Implementation — generation loop and SAE hook
Rewrite `inspect_live` as a generation loop that: registers a residual stream hook, does one forward pass per step to get both the next-token logits and the hooked activations, runs the SAE encoder, and assembles a `TokenAnalysis` per generated token.

### Phase 3: Integration — `explain_feature` and `__main__` driver
Wire everything together: `explain_feature` for human-readable output of any feature, and an updated `__main__` block that demonstrates the full pipeline.

## Step by Step Tasks

### Step 1: Add `requests` dependency
- Run `uv add requests` to add the HTTP library used for S3 discovery and downloads.

### Step 2: Update imports in `main.py`
- Add to the import block:
  ```python
  import json
  import gzip
  import os
  import requests
  from pathlib import Path
  from pydantic import BaseModel
  ```
- Keep all existing imports (`transformers`, `huggingface_hub`, `safetensors`, `torch`, `torch.nn`, `math`).
- Remove `math` if it becomes unused after the old `inspect_live` is deleted (it was only used for `math.ceil`).

### Step 3: Define Pydantic models
Add the following classes after imports, before any functions:

```python
class ActiveFeature(BaseModel):
    index: int
    activation: float
    description: str | None  # None if not found in Neuronpedia


class TokenAnalysis(BaseModel):
    token_id: int
    token: str
    l0: int                          # total number of active features (activation > 0)
    active_features: list[ActiveFeature]


class GenerationAnalysis(BaseModel):
    prompt: str
    model_id: str
    layer: int
    sae_width: str                   # e.g. "65k"
    generated_tokens: list[TokenAnalysis]
    full_generated_text: str         # decoded full generation (all tokens joined)


class NeuronpediaScope(BaseModel):
    model_id: str
    layer: int
    width: str                       # e.g. "65k"
    explanations: dict[int, str]     # feature_index -> description string
```

### Step 4: Write `list_available_scopes()`
Implement a function that queries the Neuronpedia S3 bucket and returns the available scopes for a given model:

```python
NEURONPEDIA_S3 = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"

def list_available_scopes(model_id: str) -> list[str]:
    """Return list of scope IDs available on Neuronpedia for model_id.

    Example return value: ['7-gemmascope-2-res-16k', '22-gemmascope-2-res-65k', ...]
    """
    url = (
        f"{NEURONPEDIA_S3}/?list-type=2"
        f"&prefix=v1/{model_id}/&delimiter=/"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    # Parse CommonPrefixes from XML: v1/{model_id}/{scope_id}/
    import re
    prefixes = re.findall(r"<Prefix>v1/[^/]+/([^/]+)/</Prefix>", resp.text)
    return [p for p in prefixes if p != model_id]
```

### Step 5: Write `download_neuronpedia_explanations()`
Implement a function that downloads all explanation batch files for a scope and returns a `NeuronpediaScope`. Results are cached locally in `neuronpedia_cache/`:

```python
CACHE_DIR = Path("neuronpedia_cache")

def download_neuronpedia_explanations(
    model_id: str,
    layer: int,
    width: str,
) -> NeuronpediaScope:
    """Download all feature explanation descriptions for the given scope.

    Uses a local cache at neuronpedia_cache/{model_id}_{layer}_{width}.jsonl.
    Returns a NeuronpediaScope with explanations dict populated.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{model_id}_{layer}_{width}.jsonl"

    explanations: dict[int, str] = {}

    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                entry = json.loads(line)
                explanations[entry["index"]] = entry["description"]
        return NeuronpediaScope(
            model_id=model_id, layer=layer, width=width,
            explanations=explanations,
        )

    # Discover batch count
    scope_id = f"{layer}-gemmascope-2-res-{width}"
    prefix = f"v1/{model_id}/{scope_id}/explanations/"
    list_url = (
        f"{NEURONPEDIA_S3}/?list-type=2&prefix={prefix}&delimiter=/"
    )
    resp = requests.get(list_url)
    resp.raise_for_status()
    import re
    batch_keys = re.findall(r"<Key>(" + re.escape(prefix) + r"batch-\d+\.jsonl\.gz)</Key>", resp.text)

    with open(cache_file, "w") as out:
        for key in sorted(batch_keys, key=lambda k: int(k.split("batch-")[1].split(".")[0])):
            url = f"{NEURONPEDIA_S3}/{key}"
            data = requests.get(url).content
            for line in gzip.decompress(data).decode().splitlines():
                entry = json.loads(line)
                idx = int(entry["index"])
                desc = entry["description"]
                explanations[idx] = desc
                out.write(json.dumps({"index": idx, "description": desc}) + "\n")

    return NeuronpediaScope(
        model_id=model_id, layer=layer, width=width,
        explanations=explanations,
    )
```

### Step 6: Keep `JumpReluSAE` and `load_sae` unchanged
These are correct and do not need modification.

### Step 7: Replace `get_residual_stream` and `inspect_live` with the new generation-loop version
Delete the old `get_residual_stream` and `inspect_live` functions entirely. Add:

```python
def inspect_live(
    prompt: str,
    model,
    tokenizer,
    sae: JumpReluSAE,
    layer: int,
    neuronpedia: NeuronpediaScope,
    max_new_tokens: int = 200,
) -> GenerationAnalysis:
    """Generate tokens autoregressively and capture SAE feature activations at each step.

    For each generated token (excluding the prompt), records:
    - The token string and id
    - All active SAE features (activation > 0) with their Neuronpedia description
    - L0: total number of active features for that token position

    Stops at EOS or when max_new_tokens is reached.
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)

    token_analyses: list[TokenAnalysis] = []

    for _ in range(max_new_tokens):
        # Capture residual stream at the last token position via a one-shot hook
        captured: list[torch.Tensor] = []

        def hook_fn(_module, _input, output):
            captured.append(output[0].detach())

        hook = model.model.layers[layer].register_forward_hook(hook_fn)
        outputs = model(input_ids)
        hook.remove()

        # residual at the last (newly appended) position: shape (d_model,)
        residual_last = captured[0][0, -1, :]  # (d_model,)

        # Next token: greedy argmax
        next_token_id = int(outputs.logits[0, -1].argmax().item())

        # SAE encode: shape (d_sae,)
        sae_acts = sae.encode(residual_last.float().unsqueeze(0)).squeeze(0)

        # All active features
        active_indices = (sae_acts > 0).nonzero(as_tuple=True)[0].tolist()
        l0 = len(active_indices)
        active_features = [
            ActiveFeature(
                index=i,
                activation=sae_acts[i].item(),
                description=neuronpedia.explanations.get(i),
            )
            for i in active_indices
        ]

        token_str = tokenizer.decode([next_token_id])
        token_analyses.append(TokenAnalysis(
            token_id=next_token_id,
            token=token_str,
            l0=l0,
            active_features=active_features,
        ))

        # Stop at EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token_id]], device=device)],
            dim=1,
        )

    full_text = tokenizer.decode(
        [t.token_id for t in token_analyses],
        skip_special_tokens=True,
    )

    return GenerationAnalysis(
        prompt=prompt,
        model_id="google/gemma-3-1b-pt",
        layer=layer,
        sae_width=neuronpedia.width,
        generated_tokens=token_analyses,
        full_generated_text=full_text,
    )
```

### Step 8: Write `explain_feature()`
Add after `inspect_live`:

```python
def explain_feature(feature: ActiveFeature) -> str:
    """Return a human-readable string describing a single active feature."""
    desc = feature.description or "no description available"
    return (
        f"Feature {feature.index}\n"
        f"  Concept : {desc}\n"
        f"  Activation : {feature.activation:.4f}"
    )
```

### Step 9: Update `__main__` driver block
Replace the existing `__main__` block:

```python
if __name__ == "__main__":
    MODEL_ID = "google/gemma-3-1b-pt"
    LAYER = 22
    WIDTH = "65k"

    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading SAE (layer={LAYER}, width={WIDTH})...")
    sae = load_sae(layer=LAYER, width=WIDTH, l0="medium", device=device)

    print(f"Downloading Neuronpedia explanations for layer {LAYER} {WIDTH}...")
    print("  Available scopes:", list_available_scopes("gemma-3-1b"))
    neuronpedia = download_neuronpedia_explanations("gemma-3-1b", LAYER, WIDTH)
    print(f"  Loaded {len(neuronpedia.explanations)} feature descriptions.")

    prompt = "The law of conservation of energy states that energy cannot be created or destroyed."
    print(f"\nRunning generation inspection for prompt: {prompt!r}\n")

    result = inspect_live(prompt, model, tokenizer, sae, LAYER, neuronpedia)

    print(f"Generated text: {result.full_generated_text!r}\n")
    for step, tok in enumerate(result.generated_tokens, start=1):
        print(f"Step {step:>3} | token={tok.token!r:<15} | L0={tok.l0}")
        for feat in tok.active_features[:3]:   # print first 3 active features as sample
            print(f"           {explain_feature(feat)}")
        if len(tok.active_features) > 3:
            print(f"           ... and {len(tok.active_features) - 3} more features")
```

### Step 10: Run validation commands
- Execute all commands listed in **Validation Commands** below and confirm zero errors.

## Testing Strategy

### Unit Tests
No formal test suite. Manual smoke test via `uv run python main.py`.

### Integration Tests
- Confirm `list_available_scopes("gemma-3-1b")` returns a non-empty list containing `"22-gemmascope-2-res-65k"`.
- Confirm `download_neuronpedia_explanations("gemma-3-1b", 22, "65k")` returns a `NeuronpediaScope` with `len(explanations) == 65536`.
- Confirm `inspect_live(...)` returns a `GenerationAnalysis` with `len(generated_tokens) >= 1`.
- Confirm each `TokenAnalysis` has `l0 == len(active_features)`.
- Confirm `explain_feature(feat)` returns a non-empty string for any `ActiveFeature`.

### Edge Cases
- **EOS on first token**: `generated_tokens` has length 1; no crash.
- **Feature with no Neuronpedia description**: `description=None`; `explain_feature` returns `"no description available"`.
- **`max_new_tokens=0`**: Loop body never executes; returns `GenerationAnalysis` with empty `generated_tokens` and `full_generated_text=""`.
- **Cache already exists**: Second call reads from local file, no HTTP requests made.
- **CPU-only machine**: `device="cpu"`, no `.cuda()` calls; all tensor operations work on CPU.

## Acceptance Criteria
- `uv run python main.py` runs end-to-end without error.
- `result` is a `GenerationAnalysis` Pydantic object with at least one `TokenAnalysis` entry.
- Each `TokenAnalysis.l0` equals `len(TokenAnalysis.active_features)`.
- Each `ActiveFeature.description` is either a non-empty string or `None` (never raises KeyError).
- `explain_feature(feat)` returns a multi-line human-readable string for any `ActiveFeature`.
- `neuronpedia_cache/gemma-3-1b_22_65k.jsonl` is created on first run and reused on subsequent runs.
- Generation stops at EOS token or `max_new_tokens`, whichever comes first.
- The `GenerationAnalysis` object is JSON-serializable via `.model_dump_json()` (Pydantic v2).

## Validation Commands
- `uv add requests` — Install new dependency
- `uv run python -c "from main import ActiveFeature, TokenAnalysis, GenerationAnalysis, NeuronpediaScope; print('Pydantic models OK')"` — Confirm models importable
- `uv run python -c "from main import list_available_scopes; s = list_available_scopes('gemma-3-1b'); print('scopes:', s); assert '22-gemmascope-2-res-65k' in s"` — Confirm S3 discovery works
- `uv run python -c "from main import download_neuronpedia_explanations; np = download_neuronpedia_explanations('gemma-3-1b', 22, '65k'); print('features loaded:', len(np.explanations))"` — Confirm explanation download and cache
- `uv run python main.py` — Run full pipeline end-to-end

## Notes
- **`requests` vs `urllib`**: `requests` is added as a dependency for clean HTTP handling. `urllib` would work but is more verbose for this use case.
- **Greedy decoding**: The generation loop uses `argmax` (greedy) for simplicity. Beam search or sampling can be added later by swapping the next-token selection line.
- **Memory**: Storing all 65k activation values per token is unnecessary — we store only non-zero indices and their values (sparse). For a 200-token generation, this is at most 200 × ~100 active features = 20,000 `ActiveFeature` objects, which is lightweight.
- **`input_ids` growth**: At each step `input_ids` grows by 1 token. KV-cache is not explicitly managed here; HuggingFace handles it internally when `use_cache=True` (default). However, since we register a new hook at every step, the hook fires correctly each time.
- **Neuronpedia scope naming convention**: The S3 prefix uses `{layer}-gemmascope-2-res-{width}` regardless of the actual Gemma model generation. The "2" refers to the second SAE release, not the model.
- **Available layers for gemma-3-1b**: 7, 13, 17, 22 — all with 16k, 65k, and 262k widths. `list_available_scopes()` will show all of them.
- **No new Pydantic dependency needed**: `pydantic>=2.12.5` is already in `pyproject.toml`.
- **Future**: Text snippets from `activations/` batches can be added to `ActiveFeature` later by downloading and indexing the `activations/` directory — same S3 pattern, much larger dataset (~330MB per scope).
