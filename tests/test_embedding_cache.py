from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_rag.embedding_cache import load_or_compute_embeddings
from agentic_rag.types import DocumentChunk


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


class EmbeddingCacheTest(unittest.TestCase):
    def test_reuses_cached_record_embeddings(self) -> None:
        records = [
            DocumentChunk("1", "source", "alpha"),
            DocumentChunk("2", "source", "beta"),
        ]
        embedder = CountingEmbedder()
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            first = load_or_compute_embeddings(records, embedder, {"provider": "test"}, cache_dir)
            second = load_or_compute_embeddings(records, embedder, {"provider": "test"}, cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(1, embedder.calls)


if __name__ == "__main__":
    unittest.main()
