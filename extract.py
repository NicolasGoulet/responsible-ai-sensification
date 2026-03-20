"""extract.py: Extract SAE feature activations from a language model generation.

Usage:
    uv run python extract.py "your prompt" [--model MODEL] [--layer LAYER]
        [--width WIDTH] [--l0 L0] [--max-tokens N] [--output PATH] [--verbose]
        [--stream] [--loop]
"""
import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Generator

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
    """Return list of scope IDs available on Neuronpedia for model_id."""
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
) -> Generator[tuple[TokenAnalysis, int], None, None]:
    """Generate tokens autoregressively and capture SAE feature activations.

    Yields (TokenAnalysis, elapsed_ms) per token. elapsed_ms covers the forward
    pass + SAE encode + TokenAnalysis construction time.
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)

    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        captured: list[torch.Tensor] = []

        def hook_fn(_module, _input, output):
            captured.append(output[0].detach())

        hook = model.model.layers[layer].register_forward_hook(hook_fn)
        outputs = model(input_ids)
        hook.remove()

        hidden = captured[0]
        residual_last = hidden.squeeze(0)[-1, :]  # (d_model,)

        next_token_id = int(outputs.logits[0, -1].argmax().item())

        sae_acts = sae.encode(residual_last.float().unsqueeze(0)).squeeze(0)

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
        token_analysis = TokenAnalysis(
            token_id=next_token_id,
            token=token_str,
            l0=l0,
            active_features=active_features,
        )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        yield token_analysis, elapsed_ms

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token_id]], device=device)],
            dim=1,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SAE feature activations from a language model generation."
    )
    parser.add_argument("prompt", type=str, help="Prompt to generate from")
    parser.add_argument(
        "--model", default="google/gemma-3-1b-pt", help="HuggingFace model ID"
    )
    parser.add_argument("--layer", type=int, default=22, help="Transformer layer index")
    parser.add_argument("--width", default="65k", help="SAE width (e.g. 65k)")
    parser.add_argument("--l0", default="medium", help="SAE L0 target (e.g. medium)")
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/analysis.json"),
        help="Output JSON path",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    parser.add_argument(
        "--stream", action="store_true", help="Emit NDJSON per token to stdout"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Replay recorded tokens indefinitely after generation (Ctrl+C to stop)",
    )
    args = parser.parse_args()

    def log(msg: str) -> None:
        if args.verbose:
            print(msg, file=sys.stderr)

    log(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    log(f"Loading SAE (layer={args.layer}, width={args.width}, l0={args.l0})...")
    sae = load_sae(layer=args.layer, width=args.width, l0=args.l0, device=device)

    log(f"Downloading Neuronpedia explanations for layer {args.layer} {args.width}...")
    neuronpedia = download_neuronpedia_explanations("gemma-3-1b", args.layer, args.width)
    log(f"  Loaded {len(neuronpedia.explanations)} feature descriptions.")

    if args.stream:
        meta = {
            "type": "meta",
            "model_id": args.model,
            "layer": args.layer,
            "sae_width": args.width,
        }
        print(json.dumps(meta), flush=True)

    log(f"Running generation for prompt: {args.prompt!r}")

    results: list[tuple[TokenAnalysis, int]] = []
    try:
        for token_analysis, elapsed_ms in inspect_live(
            args.prompt,
            model,
            tokenizer,
            sae,
            args.layer,
            neuronpedia,
            max_new_tokens=args.max_tokens,
        ):
            results.append((token_analysis, elapsed_ms))

            is_eos = token_analysis.token_id == tokenizer.eos_token_id
            if is_eos:
                print(f"[extract] EOS token received after {len(results)} tokens", file=sys.stderr, flush=True)

            if args.stream:
                event = {
                    "type": "token",
                    "token_id": token_analysis.token_id,
                    "token": token_analysis.token,
                    "l0": token_analysis.l0,
                    "active_features": [f.model_dump() for f in token_analysis.active_features],
                    "elapsed_ms": elapsed_ms,
                }
                print(json.dumps(event), flush=True)
    except KeyboardInterrupt:
        sys.exit(0)

    token_analyses = [ta for ta, _ in results]
    full_text = tokenizer.decode(
        [t.token_id for t in token_analyses], skip_special_tokens=True
    )
    result = GenerationAnalysis(
        prompt=args.prompt,
        model_id=args.model,
        layer=args.layer,
        sae_width=args.width,
        generated_tokens=token_analyses,
        full_generated_text=full_text,
    )

    export_to_json(result, args.output)
    log(f"Exported analysis to {args.output}")

    if args.loop:
        token_events = [
            {
                "type": "token",
                "token_id": ta.token_id,
                "token": ta.token,
                "l0": ta.l0,
                "active_features": [f.model_dump() for f in ta.active_features],
                "elapsed_ms": elapsed_ms,
            }
            for ta, elapsed_ms in results
        ]
        try:
            loop_count = 0
            while True:
                loop_count += 1
                print(f"[loop] starting replay iteration {loop_count} ({len(token_events)} tokens)", file=sys.stderr, flush=True)
                for event in token_events:
                    print(json.dumps(event), flush=True)
                    time.sleep(event["elapsed_ms"] / 1000)
        except KeyboardInterrupt:
            pass
