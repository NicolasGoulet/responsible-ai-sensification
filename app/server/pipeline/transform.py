"""transform.py: Transform TokenStream NDJSON to MusicalEvent NDJSON.

Usage:
    # Pipe mode (stdin → stdout):
    uv run python extract.py ... --stream | uv run python transform.py [--strategy identity|cluster]

    # Standalone mode (batch JSON → stdout):
    uv run python transform.py runs/analysis.json [--strategy identity|cluster]
"""
import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import torch

from app.server.pipeline.audio_utils import feature_to_frequency

INSTRUMENT_LIST = ["piano", "guitar", "bass", "strings", "pad", "bell", "flute", "brass"]
CACHE_DIR = Path("neuronpedia_cache")


def build_cluster_map(
    model_id: str,
    layer: int,
    sae_width: str,
    n_clusters: int,
    embed_model: str,
) -> dict[int, dict]:
    """Return {feature_index: {cluster_id, instrument}} mapping."""
    safe_model_id = model_id.replace("/", "_")
    cache_path = CACHE_DIR / f"{safe_model_id}_{layer}_{sae_width}_clusters_{n_clusters}.json"

    if cache_path.exists():
        print("Loading cluster map from cache...", file=sys.stderr)
        with open(cache_path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    np_cache = CACHE_DIR / f"{model_id}_{layer}_{sae_width}.jsonl"
    if not np_cache.exists():
        # model_id in the NDJSON meta may differ from the Neuronpedia cache key
        # (e.g. "google/gemma-3-1b-pt" vs "gemma-3-1b") — search by layer+width
        matches = [
            p for p in CACHE_DIR.glob(f"*_{layer}_{sae_width}.jsonl")
            if "clusters" not in p.name
        ]
        if matches:
            np_cache = matches[0]
            print(f"Using Neuronpedia cache: {np_cache}", file=sys.stderr)
        else:
            print(f"Neuronpedia cache not found for layer={layer} width={sae_width}", file=sys.stderr)
            return {}

    indices = []
    descriptions = []
    with open(np_cache) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("description"):
                indices.append(entry["index"])
                descriptions.append(entry["description"])

    if not descriptions:
        print("No descriptions found, cluster map empty.", file=sys.stderr)
        return {}

    n_clusters = min(n_clusters, len(descriptions))

    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 512 if embed_device == "cuda" else 64
    print(f"Embedding {len(descriptions)} descriptions on {embed_device}...", file=sys.stderr)

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embed_model, device=embed_device)
    embeddings = embedder.encode(descriptions, batch_size=batch_size, show_progress_bar=True)

    print(f"Running MiniBatchKMeans with {n_clusters} clusters...", file=sys.stderr)
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    instruments = [inst for inst, _ in zip(cycle(INSTRUMENT_LIST), range(n_clusters))]

    cluster_map = {}
    for idx, label in zip(indices, labels):
        cluster_map[idx] = {"cluster_id": int(label), "instrument": instruments[label]}

    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in cluster_map.items()}, f)
    print(f"Cluster map saved to {cache_path}", file=sys.stderr)

    return cluster_map


def apply_identity(active_features: list[dict]) -> list[dict]:
    return [
        {"freq": feature_to_frequency(f["index"]), "amplitude": f["activation"], "cluster": None}
        for f in active_features
    ]


def apply_cluster(active_features: list[dict], cluster_map: dict) -> list[dict]:
    notes = []
    for f in active_features:
        freq = feature_to_frequency(f["index"])
        amplitude = f["activation"]
        info = cluster_map.get(f["index"], {"cluster_id": 0, "instrument": INSTRUMENT_LIST[0]})
        notes.append(
            {
                "freq": freq,
                "amplitude": amplitude,
                "cluster": info["cluster_id"],
                "instrument": info["instrument"],
            }
        )
    return notes


def events_from_batch_json(path: Path):
    """Yield (type, data) events from a batch JSON file."""
    with open(path) as f:
        data = json.load(f)
    yield "meta", {
        "type": "meta",
        "model_id": data.get("model_id", ""),
        "layer": data.get("layer", 0),
        "sae_width": data.get("sae_width", ""),
    }
    for tok in data.get("generated_tokens", []):
        yield "token", {
            "type": "token",
            "token_id": tok.get("token_id", 0),
            "token": tok.get("token", ""),
            "l0": tok.get("l0", 0),
            "active_features": tok.get("active_features", []),
            "elapsed_ms": 0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Transform TokenStream NDJSON to MusicalEvent NDJSON"
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="Batch JSON file (omit to read from stdin)"
    )
    parser.add_argument(
        "--strategy", default="identity", choices=["identity", "cluster"]
    )
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters")
    parser.add_argument(
        "--embed-model", default="all-MiniLM-L6-v2", help="Sentence transformer model"
    )
    args = parser.parse_args()

    events: list[tuple[str, dict]] = []
    if args.input:
        events = list(events_from_batch_json(args.input))
        meta_data = next(d for t, d in events if t == "meta")
    else:
        first_line = sys.stdin.readline().strip()
        if not first_line:
            return
        meta_data = json.loads(first_line)
        if meta_data.get("type") != "meta":
            print("Expected meta line first", file=sys.stderr)
            return

    model_id = meta_data.get("model_id", "")
    layer = meta_data.get("layer", 0)
    sae_width = meta_data.get("sae_width", "")

    cluster_map: dict[int, dict] = {}
    if args.strategy == "cluster":
        cluster_map = build_cluster_map(model_id, layer, sae_width, args.clusters, args.embed_model)

    out_meta = {**meta_data, "strategy": args.strategy}
    if args.strategy == "cluster":
        out_meta["n_clusters"] = args.clusters
    print(json.dumps(out_meta), flush=True)

    if args.input:
        token_events = ((t, d) for t, d in events if t == "token")
    else:
        def _stdin_tokens():
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("type") == "token":
                    yield "token", data
        token_events = _stdin_tokens()

    for _, data in token_events:
        active_features = data.get("active_features", [])
        if args.strategy == "cluster":
            notes = apply_cluster(active_features, cluster_map)
        else:
            notes = apply_identity(active_features)

        musical_event = {
            "type": "token",
            "token": data.get("token", ""),
            "token_id": data.get("token_id", 0),
            "elapsed_ms": data.get("elapsed_ms", 0),
            "notes": notes,
        }
        print(json.dumps(musical_event), flush=True)


if __name__ == "__main__":
    main()
