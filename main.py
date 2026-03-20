import gzip
import json
import re
from pathlib import Path

import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from export import export_to_json

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ActiveFeature(BaseModel):
    index: int
    activation: float
    description: str | None  # None if not found in Neuronpedia


class TokenAnalysis(BaseModel):
    token_id: int
    token: str
    l0: int  # total number of active features (activation > 0)
    active_features: list[ActiveFeature]


class GenerationAnalysis(BaseModel):
    prompt: str
    model_id: str
    layer: int
    sae_width: str  # e.g. "65k"
    generated_tokens: list[TokenAnalysis]
    full_generated_text: str  # decoded full generation (all tokens joined)


class NeuronpediaScope(BaseModel):
    model_id: str
    layer: int
    width: str  # e.g. "65k"
    explanations: dict[int, str]  # feature_index -> description string


# ---------------------------------------------------------------------------
# Neuronpedia download layer
# ---------------------------------------------------------------------------

NEURONPEDIA_S3 = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
CACHE_DIR = Path("neuronpedia_cache")


def list_available_scopes(model_id: str) -> list[str]:
    """Return list of scope IDs available on Neuronpedia for model_id.

    Example return value: ['7-gemmascope-2-res-16k', '22-gemmascope-2-res-65k', ...]
    """
    url = f"{NEURONPEDIA_S3}/?list-type=2&prefix=v1/{model_id}/&delimiter=/"
    resp = requests.get(url)
    resp.raise_for_status()
    prefixes = re.findall(r"<Prefix>v1/[^/]+/([^/]+)/</Prefix>", resp.text)
    return [p for p in prefixes if p != model_id]


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
            model_id=model_id,
            layer=layer,
            width=width,
            explanations=explanations,
        )

    # Discover batch count
    scope_id = f"{layer}-gemmascope-2-res-{width}"
    prefix = f"v1/{model_id}/{scope_id}/explanations/"
    list_url = f"{NEURONPEDIA_S3}/?list-type=2&prefix={prefix}&delimiter=/"
    resp = requests.get(list_url)
    resp.raise_for_status()
    batch_keys = re.findall(
        r"<Key>(" + re.escape(prefix) + r"batch-\d+\.jsonl\.gz)</Key>", resp.text
    )

    with open(cache_file, "w") as out:
        for key in sorted(
            batch_keys, key=lambda k: int(k.split("batch-")[1].split(".")[0])
        ):
            url = f"{NEURONPEDIA_S3}/{key}"
            data = requests.get(url).content
            for line in gzip.decompress(data).decode().splitlines():
                entry = json.loads(line)
                idx = int(entry["index"])
                desc = entry["description"]
                explanations[idx] = desc
                out.write(json.dumps({"index": idx, "description": desc}) + "\n")

    return NeuronpediaScope(
        model_id=model_id,
        layer=layer,
        width=width,
        explanations=explanations,
    )


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------


class JumpReluSAE(nn.Module):
    def __init__(self, w_enc, b_enc, threshold, w_dec, b_dec):
        super().__init__()
        self.w_enc = nn.Parameter(w_enc)
        self.b_enc = nn.Parameter(b_enc)
        self.threshold = nn.Parameter(threshold)
        self.w_dec = nn.Parameter(w_dec)
        self.b_dec = nn.Parameter(b_dec)

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.relu(pre_acts)
        return acts


def load_sae(
    layer=22, width="65k", l0="medium", category="resid_post", device=device
) -> JumpReluSAE:
    path = f"{category}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
    local_path = hf_hub_download(repo_id="google/gemma-scope-2-1b-pt", filename=path)
    tensors = load_file(local_path)
    sae = JumpReluSAE(
        w_enc=tensors["w_enc"],
        b_enc=tensors["b_enc"],
        threshold=tensors["threshold"],
        w_dec=tensors["w_dec"],
        b_dec=tensors["b_dec"],
    )
    return sae.to(device).eval()


# ---------------------------------------------------------------------------
# Generation-time inspection
# ---------------------------------------------------------------------------


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
        hidden = captured[0]
        residual_last = hidden.squeeze(0)[-1, :]  # (d_model,)

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
        token_analyses.append(
            TokenAnalysis(
                token_id=next_token_id,
                token=token_str,
                l0=l0,
                active_features=active_features,
            )
        )

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


def explain_feature(feature: ActiveFeature) -> str:
    """Return a human-readable string describing a single active feature."""
    desc = feature.description or "no description available"
    return (
        f"Feature {feature.index}\n"
        f"  Concept : {desc}\n"
        f"  Activation : {feature.activation:.4f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
        for feat in tok.active_features[:3]:  # print first 3 active features as sample
            print(f"           {explain_feature(feat)}")
        if len(tok.active_features) > 3:
            print(f"           ... and {len(tok.active_features) - 3} more features")

    output_json = Path("runs") / "analysis.json"
    export_to_json(result, output_json)
    print(f"\nExported analysis to {output_json}")
