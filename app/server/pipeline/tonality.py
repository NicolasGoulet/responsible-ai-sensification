"""tonality.py: Load key-description sets and optionally precompute semantic embeddings.

This module is intentionally parallel to transform.py's feature-description embedding flow.
It gives the project a default, editable set of key-character descriptions and a small CLI
for converting those descriptions into a reusable embedding cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_DIR = Path(__file__).with_name("tonality_data")
DEFAULT_KEYSET_PATH = DATA_DIR / "schubart_keys.json"
DEFAULT_CACHE_DIR = Path("tonality_cache")


@dataclass(frozen=True)
class KeyDescription:
    key: str
    description: str


@dataclass(frozen=True)
class KeyDescriptionSet:
    name: str
    source: str
    description: str
    keys: list[KeyDescription]

    def description_map(self) -> dict[str, str]:
        return {entry.key: entry.description for entry in self.keys}


@dataclass(frozen=True)
class KeyEmbeddingEntry:
    key: str
    description: str
    embedding: list[float]


@dataclass(frozen=True)
class KeyEmbeddingCache:
    name: str
    source: str
    description: str
    embed_model: str
    dimensions: int
    content_hash: str
    keys: list[KeyEmbeddingEntry]


def _coerce_description_set(raw: dict[str, Any]) -> KeyDescriptionSet:
    if not isinstance(raw, dict):
        raise ValueError("Key-description file must be a JSON object")

    name = str(raw.get("name") or "custom_keyset")
    source = str(raw.get("source") or "")
    description = str(raw.get("description") or "")
    keys_raw = raw.get("keys")
    if not isinstance(keys_raw, dict) or not keys_raw:
        raise ValueError("Key-description file must contain a non-empty 'keys' object")

    keys: list[KeyDescription] = []
    seen: set[str] = set()
    for key_name, text in keys_raw.items():
        if not isinstance(key_name, str) or not key_name.strip():
            raise ValueError("Every key name must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Description for key {key_name!r} must be a non-empty string")
        normalized = key_name.strip()
        if normalized in seen:
            raise ValueError(f"Duplicate key name in description set: {normalized}")
        seen.add(normalized)
        keys.append(KeyDescription(key=normalized, description=text.strip()))

    keys.sort(key=lambda entry: entry.key)
    return KeyDescriptionSet(name=name, source=source, description=description, keys=keys)


def load_key_descriptions(path: str | Path | None = None) -> KeyDescriptionSet:
    """Load a key-description set from JSON.

    Schema:
        {
          "name": "schubart_default",
          "source": "...",
          "description": "...",
          "keys": {
            "C major": "...",
            "C minor": "..."
          }
        }
    """
    target = Path(path) if path is not None else DEFAULT_KEYSET_PATH
    with open(target) as f:
        raw = json.load(f)
    return _coerce_description_set(raw)


def load_key_embedding_cache(path: str | Path) -> KeyEmbeddingCache:
    """Load a precomputed key-embedding cache from JSON."""
    with open(path) as f:
        raw = json.load(f)

    keys_raw = raw.get("keys")
    if not isinstance(keys_raw, list) or not keys_raw:
        raise ValueError("Embedding cache must contain a non-empty 'keys' list")

    keys = [
        KeyEmbeddingEntry(
            key=str(entry["key"]),
            description=str(entry["description"]),
            embedding=[float(x) for x in entry["embedding"]],
        )
        for entry in keys_raw
    ]

    return KeyEmbeddingCache(
        name=str(raw.get("name") or ""),
        source=str(raw.get("source") or ""),
        description=str(raw.get("description") or ""),
        embed_model=str(raw.get("embed_model") or DEFAULT_EMBED_MODEL),
        dimensions=int(raw.get("dimensions") or (len(keys[0].embedding) if keys else 0)),
        content_hash=str(raw.get("content_hash") or ""),
        keys=keys,
    )


def _description_set_hash(key_set: KeyDescriptionSet) -> str:
    payload = json.dumps(
        {
            "name": key_set.name,
            "source": key_set.source,
            "description": key_set.description,
            "keys": [[entry.key, entry.description] for entry in key_set.keys],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_key_embedding_cache(
    key_set: KeyDescriptionSet,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> KeyEmbeddingCache:
    """Embed all key descriptions and return a serializable cache object."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embed_model)
    descriptions = [entry.description for entry in key_set.keys]
    embeddings = embedder.encode(descriptions, show_progress_bar=True)
    dimensions = int(len(embeddings[0])) if len(embeddings) else 0

    entries = [
        KeyEmbeddingEntry(
            key=entry.key,
            description=entry.description,
            embedding=[float(x) for x in vec],
        )
        for entry, vec in zip(key_set.keys, embeddings, strict=True)
    ]
    return KeyEmbeddingCache(
        name=key_set.name,
        source=key_set.source,
        description=key_set.description,
        embed_model=embed_model,
        dimensions=dimensions,
        content_hash=_description_set_hash(key_set),
        keys=entries,
    )


def save_key_embedding_cache(
    cache: KeyEmbeddingCache,
    output_path: str | Path | None = None,
) -> Path:
    DEFAULT_CACHE_DIR.mkdir(exist_ok=True)
    target = Path(output_path) if output_path is not None else DEFAULT_CACHE_DIR / f"{cache.name}_{cache.embed_model.replace('/', '_')}.json"
    with open(target, "w") as f:
        json.dump(asdict(cache), f, ensure_ascii=False, indent=2)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Load key descriptions and precompute semantic embeddings")
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=DEFAULT_KEYSET_PATH,
        help="Path to a JSON file describing tonalities and their text descriptions",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="SentenceTransformer model used to embed the descriptions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the embedding cache (defaults to tonality_cache/<name>_<model>.json)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Validate and print the normalized description set without computing embeddings",
    )
    args = parser.parse_args()

    key_set = load_key_descriptions(args.descriptions)
    if args.print_only:
        print(json.dumps({
            "name": key_set.name,
            "source": key_set.source,
            "description": key_set.description,
            "keys": key_set.description_map(),
        }, ensure_ascii=False, indent=2))
        return

    cache = build_key_embedding_cache(key_set, embed_model=args.embed_model)
    path = save_key_embedding_cache(cache, args.output)
    print(path)


if __name__ == "__main__":
    main()
