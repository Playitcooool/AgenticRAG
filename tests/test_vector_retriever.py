from __future__ import annotations

import unittest

from agentic_rag.embeddings import HashingEmbedder
from agentic_rag.types import DocumentChunk
from agentic_rag.vector_retriever import VectorRetriever


class FakeIndex:
    name = "fake"

    def __init__(self) -> None:
        self.vectors = []

    def add(self, vectors: list[list[float]]) -> None:
        self.vectors.extend(vectors)

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        scores = []
        for index, vector in enumerate(self.vectors):
            score = sum(left * right for left, right in zip(query, vector, strict=True))
            scores.append((index, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:k]


class VectorRetrieverTest(unittest.TestCase):
    def test_searches_query_fanouts_through_index_backend(self) -> None:
        retriever = VectorRetriever(
            [
                DocumentChunk("1", "Pharmacy", "Discharge medications include metformin."),
                DocumentChunk("2", "Nutrition", "Low sodium diet recommended."),
            ],
            HashingEmbedder(dim=64),
            FakeIndex(),
        )

        results = retriever.search(["metformin", "sodium"], top_k=1)

        self.assertEqual({"1", "2"}, {result.record.record_id for result in results})


if __name__ == "__main__":
    unittest.main()
