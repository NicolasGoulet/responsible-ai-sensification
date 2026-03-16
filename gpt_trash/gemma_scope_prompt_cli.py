#!/usr/bin/env python3
import argparse
import datetime as dt
import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_grad_enabled(False)

NEURONPEDIA_S3 = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
CACHE_DIR = Path("neuronpedia_cache")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ActiveFeature(BaseModel):
    index: int
    activation: float
    description: Optional[str] = None


class TokenAnalysis(BaseModel):
    position: int
    token_id: int
    token: str
    l0: int
    active_features: list[ActiveFeature]


class FeatureAggregate(BaseModel):
    index: int
    description: Optional[str] = None
    sum_activation: float
    max_activation: float
    mean_activation_when_active: float
    active_token_count: int
    active_token_fraction: float


class PromptAnalysis(BaseModel):
    prompt: str
    hf_model_id: str
    neuronpedia_model_id: str
    layer: int
    category: str
    sae_width: str
    sae_l0: str
    token_count: int
    aggregate_metric: str
    top_features: list[FeatureAggregate]
    analyzed_tokens: list[TokenAnalysis]
    created_at_utc: str


class NeuronpediaScope(BaseModel):
    model_id: str
    layer: int
    width: str
    explanations: dict[int, str]


# ---------------------------------------------------------------------------
# Neuronpedia helpers
# ---------------------------------------------------------------------------

def list_available_scopes(model_id: str) -> list[str]:
    url = (
        f"{NEURONPEDIA_S3}/?list-type=2"
        f"&prefix=v1/{model_id}/&delimiter=/"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    prefixes = re.findall(r"<Prefix>v1/[^/]+/([^/]+)/</Prefix>", resp.text)
    return [p for p in prefixes if p != model_id]


def download_neuronpedia_explanations(
    model_id: str,
    layer: int,
    width: str,
) -> NeuronpediaScope:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{model_id}_{layer}_{width}.jsonl"

    explanations: dict[int, str] = {}
    if cache_file.exists():
        with cache_file.open() as f:
            for line in f:
                entry = json.loads(line)
                explanations[int(entry["index"])] = entry["description"]
        return NeuronpediaScope(model_id=model_id, layer=layer, width=width, explanations=explanations)

    scope_id = f"{layer}-gemmascope-2-res-{width}"
    prefix = f"v1/{model_id}/{scope_id}/explanations/"
    list_url = f"{NEURONPEDIA_S3}/?list-type=2&prefix={prefix}&delimiter=/"
    resp = requests.get(list_url, timeout=60)
    resp.raise_for_status()
    batch_keys = re.findall(r"<Key>(" + re.escape(prefix) + r"batch-\d+\.jsonl\.gz)</Key>", resp.text)

    with cache_file.open("w") as out:
        for key in sorted(batch_keys, key=lambda k: int(k.split("batch-")[1].split(".")[0])):
            url = f"{NEURONPEDIA_S3}/{key}"
            data = requests.get(url, timeout=120).content
            for line in gzip.decompress(data).decode().splitlines():
                entry = json.loads(line)
                idx = int(entry["index"])
                desc = entry["description"]
                explanations[idx] = desc
                out.write(json.dumps({"index": idx, "description": desc}) + "\n")

    return NeuronpediaScope(model_id=model_id, layer=layer, width=width, explanations=explanations)


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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = x @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.relu(pre_acts)
        return acts


def load_sae(
    layer: int = 22,
    width: str = "65k",
    l0: str = "medium",
    category: str = "resid_post",
    repo_id: str = "google/gemma-scope-2-1b-pt",
    hf_token: Optional[str] = None,
    device: str = "cpu",
) -> JumpReluSAE:
    path = f"{category}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
    local_path = hf_hub_download(repo_id=repo_id, filename=path, token=hf_token)
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
# Model helpers
# ---------------------------------------------------------------------------

def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def choose_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def get_layer_module(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "layers"):
        return model.layers[layer_idx]
    raise AttributeError("Could not find transformer layers on this model object.")


def load_model_and_tokenizer(model_id: str, hf_token: Optional[str], device: str):
    dtype = choose_dtype(device)
    model_kwargs = {
        "torch_dtype": dtype,
        "token": hf_token,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["attn_implementation"] = "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device == "cpu":
        model = model.to(device)
    return model.eval(), tokenizer


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def pretty_token(tokenizer, token_id: int) -> str:
    raw = tokenizer.convert_ids_to_tokens([token_id])[0]
    raw = raw.replace("▁", "␠")
    raw = raw.replace("Ġ", "␠")
    raw = raw.replace("\n", "↩")
    return raw


def capture_layer_residuals(model, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook_fn(_module, _input, output):
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        captured.append(tensor.detach())

    hook = get_layer_module(model, layer).register_forward_hook(hook_fn)
    with torch.inference_mode():
        _ = model(input_ids=input_ids)
    hook.remove()

    if not captured:
        raise RuntimeError("Forward hook did not capture any residual activations.")
    return captured[0][0]  # (seq_len, d_model)


def encode_residuals_batched(sae: JumpReluSAE, residuals: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    outputs = []
    for start in range(0, residuals.shape[0], batch_size):
        chunk = residuals[start : start + batch_size].float()
        acts = sae.encode(chunk)
        outputs.append(acts.detach().cpu())
    return torch.cat(outputs, dim=0)


def analyze_prompt(
    prompt: str,
    model,
    tokenizer,
    sae: JumpReluSAE,
    layer: int,
    neuronpedia: NeuronpediaScope,
    category: str = "resid_post",
    sae_l0: str = "medium",
    aggregate_metric: str = "sum_activation",
    top_k: int = 12,
    token_batch_size: int = 64,
) -> tuple[PromptAnalysis, torch.Tensor, list[str]]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    residuals = capture_layer_residuals(model, input_ids, layer)
    sae_acts = encode_residuals_batched(sae, residuals, batch_size=token_batch_size)  # (seq_len, d_sae)

    token_ids = input_ids[0].detach().cpu().tolist()
    token_labels = [pretty_token(tokenizer, tid) for tid in token_ids]

    # Filter special tokens from downstream reporting.
    keep_positions = [
        i for i, tid in enumerate(token_ids)
        if tid not in set(tokenizer.all_special_ids)
    ]
    if not keep_positions:
        raise ValueError("After removing special tokens, there are no tokens left to analyze.")

    sae_acts = sae_acts[keep_positions]
    token_ids = [token_ids[i] for i in keep_positions]
    token_labels = [token_labels[i] for i in keep_positions]

    token_analyses: list[TokenAnalysis] = []
    for pos, (token_id, token_str, acts) in enumerate(zip(token_ids, token_labels, sae_acts), start=0):
        active_indices = (acts > 0).nonzero(as_tuple=True)[0].tolist()
        active_features = [
            ActiveFeature(
                index=int(i),
                activation=float(acts[i].item()),
                description=neuronpedia.explanations.get(int(i)),
            )
            for i in active_indices
        ]
        token_analyses.append(
            TokenAnalysis(
                position=pos,
                token_id=int(token_id),
                token=token_str,
                l0=len(active_indices),
                active_features=active_features,
            )
        )

    active_mask = sae_acts > 0
    sum_activation = sae_acts.sum(dim=0).numpy()
    max_activation = sae_acts.max(dim=0).values.numpy()
    active_token_count = active_mask.sum(dim=0).numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_activation_when_active = np.divide(
            sum_activation,
            active_token_count,
            out=np.zeros_like(sum_activation),
            where=active_token_count > 0,
        )
    active_token_fraction = active_token_count / sae_acts.shape[0]

    metric_map = {
        "sum_activation": sum_activation,
        "count": active_token_count,
        "max_activation": max_activation,
        "mean_activation_when_active": mean_activation_when_active,
    }
    if aggregate_metric not in metric_map:
        raise ValueError(f"Unknown aggregate metric: {aggregate_metric}")

    ranking = np.argsort(metric_map[aggregate_metric])[::-1]
    ranking = [int(i) for i in ranking if metric_map[aggregate_metric][i] > 0][:top_k]

    top_features = [
        FeatureAggregate(
            index=i,
            description=neuronpedia.explanations.get(i),
            sum_activation=float(sum_activation[i]),
            max_activation=float(max_activation[i]),
            mean_activation_when_active=float(mean_activation_when_active[i]),
            active_token_count=int(active_token_count[i]),
            active_token_fraction=float(active_token_fraction[i]),
        )
        for i in ranking
    ]

    analysis = PromptAnalysis(
        prompt=prompt,
        hf_model_id=getattr(model.config, "_name_or_path", "unknown"),
        neuronpedia_model_id=neuronpedia.model_id,
        layer=layer,
        category=category,
        sae_width=neuronpedia.width,
        sae_l0=sae_l0,
        token_count=len(token_ids),
        aggregate_metric=aggregate_metric,
        top_features=top_features,
        analyzed_tokens=token_analyses,
        created_at_utc=dt.datetime.utcnow().isoformat() + "Z",
    )
    return analysis, sae_acts, token_labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def truncate(text: Optional[str], limit: int = 72) -> str:
    if not text:
        return "(no description)"
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_report(
    analysis: PromptAnalysis,
    sae_acts: torch.Tensor,
    token_labels: list[str],
    output_png: Path,
    top_k_heatmap: int = 10,
):
    top_features = analysis.top_features[:top_k_heatmap]
    if not top_features:
        raise ValueError("No active features found; cannot render report.")

    feature_indices = [f.index for f in top_features]
    heatmap = sae_acts[:, feature_indices].numpy().T  # (features, tokens)

    bar_values = [f.sum_activation if analysis.aggregate_metric == "sum_activation"
                  else f.active_token_count if analysis.aggregate_metric == "count"
                  else f.max_activation if analysis.aggregate_metric == "max_activation"
                  else f.mean_activation_when_active
                  for f in top_features]

    bar_labels = [f"{f.index} | {truncate(f.description, 60)}" for f in top_features]
    l0_values = [tok.l0 for tok in analysis.analyzed_tokens]

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.3])

    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(bar_labels))
    ax1.barh(y, bar_values)
    ax1.set_yticks(y)
    ax1.set_yticklabels(bar_labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_title(f"Top features by {analysis.aggregate_metric}")
    ax1.set_xlabel(analysis.aggregate_metric)

    ax2 = fig.add_subplot(gs[:, 1])
    im = ax2.imshow(heatmap, aspect="auto")
    ax2.set_title("Token × feature activation heatmap")
    ax2.set_xlabel("Prompt token position")
    ax2.set_ylabel("Feature index")
    ax2.set_yticks(np.arange(len(feature_indices)))
    ax2.set_yticklabels([str(i) for i in feature_indices], fontsize=9)

    max_xticks = min(len(token_labels), 40)
    if len(token_labels) <= 40:
        xticks = np.arange(len(token_labels))
    else:
        xticks = np.linspace(0, len(token_labels) - 1, max_xticks, dtype=int)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([token_labels[i] for i in xticks], rotation=70, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(np.arange(len(l0_values)), l0_values, marker="o", linewidth=1)
    ax3.set_title("L0 sparsity by token")
    ax3.set_xlabel("Token position")
    ax3.set_ylabel("# active features")
    ax3.set_xticks(xticks if len(token_labels) > 1 else [0])
    ax3.set_xticklabels([str(i) for i in (xticks if len(token_labels) > 1 else [0])], rotation=0, fontsize=8)

    title = (
        f"Gemma Scope prompt report | layer={analysis.layer} | width={analysis.sae_width} | "
        f"tokens={analysis.token_count} | top={len(top_features)}"
    )
    fig.suptitle(title, fontsize=14)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Gemma Scope SAE activations for a prompt and export a PNG report.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt string to analyze.")
    parser.add_argument("--model-id", type=str, default="google/gemma-3-1b-pt")
    parser.add_argument("--scope-repo-id", type=str, default="google/gemma-scope-2-1b-pt")
    parser.add_argument("--neuronpedia-model-id", type=str, default="gemma-3-1b")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--width", type=str, default="65k")
    parser.add_argument("--l0", type=str, default="medium")
    parser.add_argument("--category", type=str, default="resid_post")
    parser.add_argument("--aggregate", type=str, choices=["sum_activation", "count", "max_activation", "mean_activation_when_active"], default="sum_activation")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--top-k-heatmap", type=int, default=10)
    parser.add_argument("--token-batch-size", type=int, default=64)
    parser.add_argument("--outdir", type=Path, default=Path("runs/gemma_scope_prompt_report"))
    parser.add_argument("--hf-token", type=str, default=None, help="Optional HF token. Defaults to HF_TOKEN env var.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    device = choose_device()

    print(f"[1/5] Loading tokenizer and model on {device}...")
    model, tokenizer = load_model_and_tokenizer(args.model_id, hf_token=hf_token, device=device)

    print(f"[2/5] Loading SAE from {args.scope_repo_id}...")
    sae = load_sae(
        layer=args.layer,
        width=args.width,
        l0=args.l0,
        category=args.category,
        repo_id=args.scope_repo_id,
        hf_token=hf_token,
        device=device,
    )

    print(f"[3/5] Downloading / reading Neuronpedia explanations...")
    neuronpedia = download_neuronpedia_explanations(
        model_id=args.neuronpedia_model_id,
        layer=args.layer,
        width=args.width,
    )

    print(f"[4/5] Running prompt analysis...")
    analysis, sae_acts, token_labels = analyze_prompt(
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        layer=args.layer,
        neuronpedia=neuronpedia,
        category=args.category,
        sae_l0=args.l0,
        aggregate_metric=args.aggregate,
        top_k=args.top_k,
        token_batch_size=args.token_batch_size,
    )

    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", args.prompt[:40]).strip("_") or "prompt"
    output_png = args.outdir / f"{stem}_layer{args.layer}_{args.width}.png"
    output_json = args.outdir / f"{stem}_layer{args.layer}_{args.width}.json"

    print(f"[5/5] Rendering PNG report...")
    render_report(
        analysis=analysis,
        sae_acts=sae_acts,
        token_labels=token_labels,
        output_png=output_png,
        top_k_heatmap=args.top_k_heatmap,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(analysis.model_dump_json(indent=2))

    print("\nDone.")
    print(f"PNG report : {output_png}")
    print(f"JSON report: {output_json}")
    print("\nTop features:")
    for feat in analysis.top_features[: min(8, len(analysis.top_features))]:
        print(
            f"  - feature {feat.index:<7d} | count={feat.active_token_count:<4d} | "
            f"sum={feat.sum_activation:>9.4f} | {truncate(feat.description, 90)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
