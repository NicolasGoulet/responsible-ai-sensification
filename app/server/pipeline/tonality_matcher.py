"""tonality_matcher.py: Match a prompt embedding against precomputed tonality embeddings.

This is the shared, runtime-facing prompt-to-tonality matcher. It stays agnostic to the
CLI and browser entry paths: those callers only need to provide the prompt text and a
precomputed cache of key-description embeddings.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from app.server.pipeline.tonality import DEFAULT_EMBED_MODEL


@dataclass(frozen=True)
class CachedTonalityEmbedding:
    key: str
    description: str
    embedding: list[float]


@dataclass(frozen=True)
class CachedTonalityEmbeddings:
    name: str
    source: str
    description: str
    embed_model: str
    dimensions: int
    content_hash: str
    keys: list[CachedTonalityEmbedding]


@dataclass(frozen=True)
class TonalityMatch:
    key: str
    score: float
    description: str


@dataclass(frozen=True)
class TonalityMatchResult:
    prompt: str
    embed_model: str
    top_k: int
    matches: list[TonalityMatch]


def load_cached_tonalities(path: str | Path) -> CachedTonalityEmbeddings:
    """Load a precomputed tonality embedding cache from JSON.

    This loader intentionally lives in the matcher module so the runtime browser path does
    not depend on helper functions being present in `tonality.py` on every branch.
    """
    with open(path) as f:
        raw = json.load(f)

    keys_raw = raw.get("keys")
    if not isinstance(keys_raw, list) or not keys_raw:
        raise ValueError("Tonality cache must contain a non-empty 'keys' list")

    keys = [
        CachedTonalityEmbedding(
            key=str(entry["key"]),
            description=str(entry["description"]),
            embedding=[float(x) for x in entry["embedding"]],
        )
        for entry in keys_raw
    ]
    return CachedTonalityEmbeddings(
        name=str(raw.get("name") or ""),
        source=str(raw.get("source") or ""),
        description=str(raw.get("description") or ""),
        embed_model=str(raw.get("embed_model") or DEFAULT_EMBED_MODEL),
        dimensions=int(raw.get("dimensions") or (len(keys[0].embedding) if keys else 0)),
        content_hash=str(raw.get("content_hash") or ""),
        keys=keys,
    )


def embed_prompt(prompt: str, embed_model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    """Embed the prompt into the same semantic space as the tonal-description cache."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embed_model)
    vector = embedder.encode([prompt], show_progress_bar=False)[0]
    return [float(x) for x in vector]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_tonalities(
    prompt_embedding: Sequence[float],
    cache: CachedTonalityEmbeddings,
    top_k: int = 3,
) -> list[TonalityMatch]:
    """Return the top-k closest tonalities for a prompt embedding.

    Simple initial strategy:
    - cosine similarity against every precomputed key-description embedding
    - sort high to low
    - return the k strongest matches
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    ranked = [
        TonalityMatch(
            key=entry.key,
            score=_cosine_similarity(prompt_embedding, entry.embedding),
            description=entry.description,
        )
        for entry in cache.keys
    ]
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[: min(top_k, len(ranked))]


def match_prompt_to_tonalities(
    prompt: str,
    cache_path: str | Path,
    top_k: int = 3,
    embed_model: str | None = None,
) -> TonalityMatchResult:
    """Embed the prompt and return the k most similar tonalities."""
    cache = load_cached_tonalities(cache_path)
    model_name = embed_model or cache.embed_model or DEFAULT_EMBED_MODEL
    prompt_embedding = embed_prompt(prompt, embed_model=model_name)
    matches = rank_tonalities(prompt_embedding, cache, top_k=top_k)
    return TonalityMatchResult(
        prompt=prompt,
        embed_model=model_name,
        top_k=top_k,
        matches=matches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Match a prompt to the closest precomputed tonalities")
    parser.add_argument("prompt", help="Prompt text to embed and compare against tonal descriptions")
    parser.add_argument("--cache", type=Path, required=True, help="Path to a precomputed tonality embedding cache")
    parser.add_argument("--top-k", type=int, default=3, help="Number of strongest tonal matches to return")
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Override the embedding model stored in the cache (defaults to the cache model)",
    )
    args = parser.parse_args()

    result = match_prompt_to_tonalities(
        args.prompt,
        cache_path=args.cache,
        top_k=args.top_k,
        embed_model=args.embed_model,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

