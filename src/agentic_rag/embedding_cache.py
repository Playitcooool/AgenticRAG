"""On-disk cache for corpus/chunk embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from agentic_rag.types import DocumentChunk


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


def embedding_cache_key(records: list[DocumentChunk], embedder_config: dict) -> str:
    payload = {
        "embedder": embedder_config,
        "records": [
            {
                "record_id": record.record_id,
                "source": record.source,
                "text_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
            }
            for record in records
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_or_compute_embeddings(
    records: list[DocumentChunk],
    embedder: Embedder,
    embedder_config: dict,
    cache_dir: Path,
    require_cached: bool = False,
) -> list[list[float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = embedding_cache_key(records, embedder_config)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"# using cached chunk embeddings: {cache_path}", flush=True)
        return payload["vectors"]
    if require_cached:
        raise FileNotFoundError(f"Required chunk embedding cache is missing: {cache_path}")

    vectors = embedder.encode([record.text for record in records])
    payload = {
        "cache_key": cache_key,
        "embedder": embedder_config,
        "records": [
            {
                "record_id": record.record_id,
                "source": record.source,
            }
            for record in records
        ],
        "vectors": vectors,
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"# wrote chunk embeddings cache: {cache_path}", flush=True)
    return vectors
