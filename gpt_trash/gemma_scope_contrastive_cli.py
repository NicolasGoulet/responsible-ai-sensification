#!/usr/bin/env python3
"""Contrastive Gemma Scope CLI.

This tool compares raw SAE activations across either:
1) a prompt pair (--prompt-a / --prompt-b), or
2) a CSV of prompts grouped by a label column.

It preserves the raw sparse SAE outputs per prompt and also produces
contrastive summaries to help identify which layer/site/features separate
prompt groups the most.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_grad_enabled(False)

NEURONPEDIA_S3 = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
CACHE_DIR = Path("neuronpedia_cache")
CATEGORY_TO_SCOPE_TAG = {
    "resid_post": "res",
    "attn_out": "att",
    "mlp_out": "mlp",
}


@dataclass
class PromptRecord:
    prompt_id: str
    group: str
    text: str


@dataclass
class ComboResult:
    layer: int
    category: str
    aggregate_metric: str
    site_score: float
    top_features: list[dict]
    prompt_vectors: np.ndarray  # (num_prompts, width)
    prompt_ids: list[str]
    prompt_groups: list[str]
    raw_paths: list[str]
    token_counts: list[int]


# ---------------------------------------------------------------------------
# Neuronpedia helpers
# ---------------------------------------------------------------------------


def scope_source_name(layer: int, category: str, width: str) -> str:
    tag = CATEGORY_TO_SCOPE_TAG.get(category)
    if tag is None:
        raise ValueError(f"Unsupported category for Neuronpedia source naming: {category}")
    return f"{layer}-gemmascope-2-{tag}-{width}"


class NeuronpediaScope:
    def __init__(self, model_id: str, layer: int, width: str, category: str, explanations: dict[int, str]):
        self.model_id = model_id
        self.layer = layer
        self.width = width
        self.category = category
        self.explanations = explanations



def download_neuronpedia_explanations(
    model_id: str,
    layer: int,
    width: str,
    category: str,
) -> NeuronpediaScope:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{model_id}_{layer}_{category}_{width}.jsonl"
    explanations: dict[int, str] = {}

    if cache_file.exists():
        with cache_file.open() as f:
            for line in f:
                entry = json.loads(line)
                explanations[int(entry["index"])] = entry["description"]
        return NeuronpediaScope(model_id, layer, width, category, explanations)

    source = scope_source_name(layer=layer, category=category, width=width)
    prefix = f"v1/{model_id}/{source}/explanations/"
    list_url = f"{NEURONPEDIA_S3}/?list-type=2&prefix={prefix}&delimiter=/"
    resp = requests.get(list_url, timeout=60)
    resp.raise_for_status()
    batch_keys = re.findall(r"<Key>(" + re.escape(prefix) + r"batch-\d+\.jsonl\.gz)</Key>", resp.text)

    if not batch_keys:
        return NeuronpediaScope(model_id, layer, width, category, explanations)

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

    return NeuronpediaScope(model_id, layer, width, category, explanations)


# ---------------------------------------------------------------------------
# SAE + model helpers
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
        return mask * torch.relu(pre_acts)



def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"



def choose_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32



def load_model_and_tokenizer(model_id: str, hf_token: str | None, device: str):
    dtype = choose_dtype(device)
    model_kwargs = {"torch_dtype": dtype, "token": hf_token}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["attn_implementation"] = "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device == "cpu":
        model = model.to(device)
    return model.eval(), tokenizer



def load_sae(
    layer: int,
    width: str,
    l0: str,
    category: str,
    repo_id: str,
    hf_token: str | None,
    device: str,
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



def get_layer_module(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "layers"):
        return model.layers[layer_idx]
    raise AttributeError("Could not find transformer layers on this model object.")



def get_site_module(model, layer_idx: int, category: str):
    layer = get_layer_module(model, layer_idx)
    if category == "resid_post":
        return layer

    if category == "attn_out":
        for name in ["self_attn", "attn", "attention"]:
            if hasattr(layer, name):
                return getattr(layer, name)
        raise AttributeError(f"Could not find attention module for layer {layer_idx}.")

    if category == "mlp_out":
        for name in ["mlp", "feed_forward", "ffn"]:
            if hasattr(layer, name):
                return getattr(layer, name)
        raise AttributeError(f"Could not find MLP module for layer {layer_idx}.")

    raise ValueError(f"Unsupported category: {category}")



def _extract_first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
    return None



def capture_site_activations(model, input_ids: torch.Tensor, layer: int, category: str) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    module = get_site_module(model, layer_idx=layer, category=category)

    def hook_fn(_module, _inputs, output):
        tensor = _extract_first_tensor(output)
        if tensor is None:
            raise RuntimeError(f"Hook for layer {layer} / {category} returned no tensor output.")
        captured.append(tensor.detach())

    hook = module.register_forward_hook(hook_fn)
    with torch.inference_mode():
        _ = model(input_ids=input_ids)
    hook.remove()

    if not captured:
        raise RuntimeError(f"Forward hook did not capture activations for layer {layer} / {category}.")

    result = captured[0]
    if result.ndim != 3:
        raise RuntimeError(
            f"Expected captured activations to have shape (batch, seq, hidden); got {tuple(result.shape)}."
        )
    return result[0]  # (seq_len, hidden)



def pretty_token(tokenizer, token_id: int) -> str:
    raw = tokenizer.convert_ids_to_tokens([token_id])[0]
    raw = raw.replace("▁", "␠")
    raw = raw.replace("Ġ", "␠")
    raw = raw.replace("\n", "↩")
    return raw



def encode_hidden_states_batched(sae: JumpReluSAE, hidden_states: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    outputs = []
    for start in range(0, hidden_states.shape[0], batch_size):
        chunk = hidden_states[start : start + batch_size].float()
        acts = sae.encode(chunk)
        outputs.append(acts.detach().cpu())
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------



def sanitize_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")[:80] or "prompt"



def load_pair_prompts(prompt_a: str, prompt_b: str, label_a: str, label_b: str) -> list[PromptRecord]:
    return [
        PromptRecord(prompt_id=sanitize_id(label_a), group=label_a, text=prompt_a),
        PromptRecord(prompt_id=sanitize_id(label_b), group=label_b, text=prompt_b),
    ]



def load_csv_prompts(
    csv_path: Path,
    text_column: str,
    group_column: str,
    id_column: str | None,
    group_a: str | None,
    group_b: str | None,
) -> tuple[list[PromptRecord], str, str]:
    rows: list[PromptRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (row.get(text_column) or "").strip()
            group = (row.get(group_column) or "").strip()
            if not text or not group:
                continue
            prompt_id = (row.get(id_column) or "").strip() if id_column else ""
            if not prompt_id:
                prompt_id = f"row_{i:04d}_{sanitize_id(group)}"
            rows.append(PromptRecord(prompt_id=sanitize_id(prompt_id), group=group, text=text))

    if not rows:
        raise ValueError("No usable prompts found in CSV.")

    groups = sorted({row.group for row in rows})
    if group_a is None or group_b is None:
        if len(groups) != 2:
            raise ValueError(
                "CSV mode needs either exactly 2 groups in the file or explicit --group-a / --group-b."
            )
        group_a, group_b = groups[0], groups[1]

    filtered = [row for row in rows if row.group in {group_a, group_b}]
    if not filtered:
        raise ValueError("After filtering to the requested groups, no prompts remain.")

    if not any(row.group == group_a for row in filtered) or not any(row.group == group_b for row in filtered):
        raise ValueError("Both comparison groups must have at least one prompt.")

    return filtered, group_a, group_b


# ---------------------------------------------------------------------------
# Raw prompt extraction
# ---------------------------------------------------------------------------



def extract_prompt_raw_outputs(
    prompt: str,
    model,
    tokenizer,
    sae: JumpReluSAE,
    layer: int,
    category: str,
    token_batch_size: int,
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    hidden_states = capture_site_activations(model, input_ids, layer=layer, category=category)
    sae_acts = encode_hidden_states_batched(sae, hidden_states, batch_size=token_batch_size)

    token_ids = input_ids[0].detach().cpu().tolist()
    token_labels = [pretty_token(tokenizer, tid) for tid in token_ids]

    keep_positions = [
        i for i, tid in enumerate(token_ids)
        if tid not in set(tokenizer.all_special_ids)
    ]
    if not keep_positions:
        raise ValueError("After removing special tokens, there are no tokens left to analyze.")

    sae_acts = sae_acts[keep_positions]
    token_ids = [token_ids[i] for i in keep_positions]
    token_labels = [token_labels[i] for i in keep_positions]
    l0 = (sae_acts > 0).sum(dim=1).numpy()

    return {
        "token_ids": np.asarray(token_ids, dtype=np.int64),
        "token_labels": np.asarray(token_labels),
        "sae_acts": sae_acts.numpy().astype(np.float32, copy=False),
        "l0": l0.astype(np.int32, copy=False),
    }



def aggregate_prompt_vector(sae_acts: np.ndarray, metric: str) -> np.ndarray:
    active_mask = sae_acts > 0
    if metric == "sum_activation":
        return sae_acts.sum(axis=0)
    if metric == "count":
        return active_mask.sum(axis=0).astype(np.float32)
    if metric == "max_activation":
        return sae_acts.max(axis=0)
    if metric == "mean_activation_when_active":
        counts = active_mask.sum(axis=0)
        sums = sae_acts.sum(axis=0)
        out = np.zeros_like(sums, dtype=np.float32)
        np.divide(sums, counts, out=out, where=counts > 0)
        return out
    raise ValueError(f"Unsupported aggregate metric: {metric}")



def save_sparse_raw_output(
    output_path: Path,
    prompt_record: PromptRecord,
    layer: int,
    category: str,
    width: str,
    l0_name: str,
    raw: dict,
):
    acts = raw["sae_acts"]
    rows, cols = np.nonzero(acts > 0)
    values = acts[rows, cols].astype(np.float32, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        shape=np.asarray(acts.shape, dtype=np.int64),
        token_ids=raw["token_ids"],
        token_labels=raw["token_labels"],
        l0=raw["l0"],
        active_token_indices=rows.astype(np.int32),
        active_feature_indices=cols.astype(np.int32),
        active_values=values,
        prompt_id=np.asarray([prompt_record.prompt_id]),
        group=np.asarray([prompt_record.group]),
        prompt_text=np.asarray([prompt_record.text]),
        layer=np.asarray([layer], dtype=np.int32),
        category=np.asarray([category]),
        sae_width=np.asarray([width]),
        sae_l0=np.asarray([l0_name]),
    )


# ---------------------------------------------------------------------------
# Contrastive analysis
# ---------------------------------------------------------------------------



def compute_top_features(
    group_a_mean: np.ndarray,
    group_b_mean: np.ndarray,
    descriptions: dict[int, str],
    top_k: int,
) -> list[dict]:
    diff = group_a_mean - group_b_mean
    abs_diff = np.abs(diff)
    ranking = np.argsort(abs_diff)[::-1]
    ranking = [int(i) for i in ranking if abs_diff[i] > 0][:top_k]

    out = []
    for idx in ranking:
        out.append(
            {
                "index": idx,
                "description": descriptions.get(idx),
                "group_a_mean": float(group_a_mean[idx]),
                "group_b_mean": float(group_b_mean[idx]),
                "difference": float(diff[idx]),
                "abs_difference": float(abs_diff[idx]),
            }
        )
    return out



def summarize_combo(
    prompt_vectors: np.ndarray,
    prompt_ids: list[str],
    prompt_groups: list[str],
    group_a: str,
    group_b: str,
    descriptions: dict[int, str],
    top_k: int,
    aggregate_metric: str,
    layer: int,
    category: str,
    raw_paths: list[str],
    token_counts: list[int],
) -> ComboResult:
    groups = np.asarray(prompt_groups)
    a_mask = groups == group_a
    b_mask = groups == group_b
    if not a_mask.any() or not b_mask.any():
        raise ValueError(f"Both groups must be present for combo layer={layer} category={category}.")

    group_a_mean = prompt_vectors[a_mask].mean(axis=0)
    group_b_mean = prompt_vectors[b_mask].mean(axis=0)
    top_features = compute_top_features(group_a_mean, group_b_mean, descriptions, top_k=top_k)
    site_score = 0.0
    if top_features:
        top_abs = np.asarray([feat["abs_difference"] for feat in top_features], dtype=np.float32)
        site_score = float(top_abs.mean())

    return ComboResult(
        layer=layer,
        category=category,
        aggregate_metric=aggregate_metric,
        site_score=site_score,
        top_features=top_features,
        prompt_vectors=prompt_vectors,
        prompt_ids=prompt_ids,
        prompt_groups=prompt_groups,
        raw_paths=raw_paths,
        token_counts=token_counts,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------



def truncate(text: str | None, limit: int = 72) -> str:
    if not text:
        return "(no description)"
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"



def render_overview_png(results: list[ComboResult], layers: list[int], categories: list[str], output_path: Path):
    layer_to_idx = {layer: i for i, layer in enumerate(layers)}
    category_to_idx = {cat: i for i, cat in enumerate(categories)}
    heatmap = np.full((len(categories), len(layers)), np.nan, dtype=np.float32)
    for result in results:
        heatmap[category_to_idx[result.category], layer_to_idx[result.layer]] = result.site_score

    fig, ax = plt.subplots(figsize=(max(6, 1.7 * len(layers)), max(4, 1.3 * len(categories))))
    im = ax.imshow(heatmap, aspect="auto")
    ax.set_title("Contrastive separation score by layer / site")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Category")
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories)
    for i in range(len(categories)):
        for j in range(len(layers)):
            val = heatmap[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def render_detail_png(result: ComboResult, group_a: str, group_b: str, descriptions: dict[int, str], output_path: Path):
    top_features = result.top_features
    if not top_features:
        raise ValueError("No differential features found; cannot render detail PNG.")

    feature_indices = [feat["index"] for feat in top_features]
    feature_labels = [f"{idx} | {truncate(descriptions.get(idx), 50)}" for idx in feature_indices]

    matrix = result.prompt_vectors[:, feature_indices].T  # (features, prompts)
    signed_diff = np.asarray([feat["difference"] for feat in top_features], dtype=np.float32)

    prompt_labels = [f"{group}:{pid}" for group, pid in zip(result.prompt_groups, result.prompt_ids)]
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45])

    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(feature_labels))
    ax1.barh(y, signed_diff)
    ax1.set_yticks(y)
    ax1.set_yticklabels(feature_labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel(f"{group_a} - {group_b} ({result.aggregate_metric})")
    ax1.set_title(f"Top differential features | layer={result.layer} | {result.category}")

    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(matrix, aspect="auto")
    ax2.set_title("Prompt × feature contrast heatmap")
    ax2.set_xlabel("Prompts")
    ax2.set_ylabel("Feature index")
    ax2.set_yticks(np.arange(len(feature_indices)))
    ax2.set_yticklabels([str(idx) for idx in feature_indices], fontsize=9)
    ax2.set_xticks(np.arange(len(prompt_labels)))
    ax2.set_xticklabels(prompt_labels, rotation=65, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def parse_csv_list(value: str, cast=str) -> list:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [cast(item) for item in items]



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare raw Gemma Scope SAE activations across a prompt pair or grouped prompt CSV."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt-a", type=str, help="First prompt for pair mode.")
    mode.add_argument("--csv", type=Path, help="CSV file containing prompts for grouped comparison mode.")

    parser.add_argument("--prompt-b", type=str, help="Second prompt for pair mode.")
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")

    parser.add_argument("--text-column", type=str, default="prompt")
    parser.add_argument("--group-column", type=str, default="group")
    parser.add_argument("--id-column", type=str, default="prompt_id")
    parser.add_argument("--group-a", type=str, default=None)
    parser.add_argument("--group-b", type=str, default=None)

    parser.add_argument("--model-id", type=str, default="google/gemma-3-1b-pt")
    parser.add_argument("--scope-repo-id", type=str, default="google/gemma-scope-2-1b-pt")
    parser.add_argument("--neuronpedia-model-id", type=str, default="gemma-3-1b")
    parser.add_argument("--layers", type=str, default="22", help="Comma-separated layer list, e.g. 8,12,16,20,22,24")
    parser.add_argument("--categories", type=str, default="resid_post", help="Comma-separated categories, e.g. resid_post,attn_out,mlp_out")
    parser.add_argument("--width", type=str, default="65k")
    parser.add_argument("--l0", type=str, default="medium")
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["sum_activation", "count", "max_activation", "mean_activation_when_active"],
        default="sum_activation",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--token-batch-size", type=int, default=64)
    parser.add_argument("--outdir", type=Path, default=Path("runs/gemma_scope_contrastive"))
    parser.add_argument("--hf-token", type=str, default=None)
    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.prompt_a and not args.prompt_b:
        parser.error("Pair mode requires both --prompt-a and --prompt-b.")

    layers = parse_csv_list(args.layers, cast=int)
    categories = parse_csv_list(args.categories, cast=str)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    device = choose_device()

    if args.prompt_a:
        prompts = load_pair_prompts(args.prompt_a, args.prompt_b, args.label_a, args.label_b)
        group_a, group_b = args.label_a, args.label_b
    else:
        prompts, group_a, group_b = load_csv_prompts(
            csv_path=args.csv,
            text_column=args.text_column,
            group_column=args.group_column,
            id_column=args.id_column,
            group_a=args.group_a,
            group_b=args.group_b,
        )

    print(f"[1/6] Loading tokenizer and model on {device}...")
    model, tokenizer = load_model_and_tokenizer(args.model_id, hf_token=hf_token, device=device)

    raw_dir = args.outdir / "raw"
    results: list[ComboResult] = []

    total_combos = len(layers) * len(categories)
    combo_counter = 0
    for layer in layers:
        for category in categories:
            combo_counter += 1
            print(f"[2/6] Loading SAE {combo_counter}/{total_combos}: layer={layer} category={category}...")
            sae = load_sae(
                layer=layer,
                width=args.width,
                l0=args.l0,
                category=category,
                repo_id=args.scope_repo_id,
                hf_token=hf_token,
                device=device,
            )

            print(f"[3/6] Loading Neuronpedia explanations for layer={layer} category={category}...")
            neuronpedia = download_neuronpedia_explanations(
                model_id=args.neuronpedia_model_id,
                layer=layer,
                width=args.width,
                category=category,
            )

            print(f"[4/6] Extracting raw SAE outputs for {len(prompts)} prompts...")
            prompt_vectors = []
            prompt_ids = []
            prompt_groups = []
            raw_paths = []
            token_counts = []

            for idx, prompt_record in enumerate(prompts, start=1):
                print(f"       - {idx:>3}/{len(prompts)} | {prompt_record.group}:{prompt_record.prompt_id}")
                raw = extract_prompt_raw_outputs(
                    prompt=prompt_record.text,
                    model=model,
                    tokenizer=tokenizer,
                    sae=sae,
                    layer=layer,
                    category=category,
                    token_batch_size=args.token_batch_size,
                )
                vector = aggregate_prompt_vector(raw["sae_acts"], metric=args.aggregate)
                prompt_vectors.append(vector)
                prompt_ids.append(prompt_record.prompt_id)
                prompt_groups.append(prompt_record.group)
                token_counts.append(int(raw["sae_acts"].shape[0]))

                raw_path = raw_dir / f"layer_{layer}" / category / f"{prompt_record.group}__{prompt_record.prompt_id}.npz"
                save_sparse_raw_output(
                    output_path=raw_path,
                    prompt_record=prompt_record,
                    layer=layer,
                    category=category,
                    width=args.width,
                    l0_name=args.l0,
                    raw=raw,
                )
                raw_paths.append(str(raw_path))

            prompt_vectors_np = np.stack(prompt_vectors, axis=0).astype(np.float32)
            result = summarize_combo(
                prompt_vectors=prompt_vectors_np,
                prompt_ids=prompt_ids,
                prompt_groups=prompt_groups,
                group_a=group_a,
                group_b=group_b,
                descriptions=neuronpedia.explanations,
                top_k=args.top_k,
                aggregate_metric=args.aggregate,
                layer=layer,
                category=category,
                raw_paths=raw_paths,
                token_counts=token_counts,
            )
            results.append(result)

    if not results:
        raise ValueError("No results were produced.")

    results.sort(key=lambda x: x.site_score, reverse=True)
    best = results[0]

    print("[5/6] Writing summary JSON...")
    summary = {
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "hf_model_id": args.model_id,
        "scope_repo_id": args.scope_repo_id,
        "neuronpedia_model_id": args.neuronpedia_model_id,
        "groups": {"a": group_a, "b": group_b},
        "aggregate_metric": args.aggregate,
        "width": args.width,
        "l0": args.l0,
        "num_prompts": len(prompts),
        "best_combo": {
            "layer": best.layer,
            "category": best.category,
            "site_score": best.site_score,
            "top_features": best.top_features,
        },
        "all_combos": [
            {
                "layer": result.layer,
                "category": result.category,
                "site_score": result.site_score,
                "top_features": result.top_features,
                "prompt_ids": result.prompt_ids,
                "prompt_groups": result.prompt_groups,
                "raw_paths": result.raw_paths,
                "token_counts": result.token_counts,
            }
            for result in results
        ],
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_json = args.outdir / "contrastive_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    print("[6/6] Rendering PNGs...")
    overview_png = args.outdir / "overview_scores.png"
    detail_png = args.outdir / "best_combo_detail.png"
    render_overview_png(results=results, layers=layers, categories=categories, output_path=overview_png)

    best_descriptions = download_neuronpedia_explanations(
        model_id=args.neuronpedia_model_id,
        layer=best.layer,
        width=args.width,
        category=best.category,
    ).explanations
    render_detail_png(best, group_a=group_a, group_b=group_b, descriptions=best_descriptions, output_path=detail_png)

    print("\nDone.")
    print(f"Summary JSON : {summary_json}")
    print(f"Overview PNG : {overview_png}")
    print(f"Detail PNG   : {detail_png}")
    print(f"Best combo   : layer={best.layer} | category={best.category} | score={best.site_score:.4f}")
    print("Top differential features:")
    for feat in best.top_features[: min(8, len(best.top_features))]:
        print(
            f"  - feature {feat['index']:<7d} | diff={feat['difference']:>10.4f} | "
            f"{truncate(feat['description'], 90)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
