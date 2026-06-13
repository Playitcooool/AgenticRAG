"""Dense-vector retriever adapter for the agentic RAG pipeline."""

from __future__ import annotations

from collections.abc import Iterable

from agentic_rag.embeddings import HashingEmbedder
from agentic_rag.types import DocumentChunk, SearchResult
from agentic_rag.vector_backends import VectorIndex


class VectorRetriever:
    """Retriever that delegates nearest-neighbor search to a vector index."""

    def __init__(self, records: Iterable[DocumentChunk], embedder: HashingEmbedder, index: VectorIndex):
        self.records = list(records)
        self.embedder = embedder
        self.index = index
        self.index.add(self.embedder.encode([record.text for record in self.records]))

    def search(self, queries: Iterable[str], top_k: int = 4) -> list[SearchResult]:
        best_by_record: dict[str, SearchResult] = {}
        for query in queries:
            query_vector = self.embedder.encode_one(query)
            for record_index, score in self.index.search(query_vector, top_k):
                record = self.records[record_index]
                result = SearchResult(query=query, record=record, score=score)
                current = best_by_record.get(record.record_id)
                if current is None or result.score > current.score:
                    best_by_record[record.record_id] = result
        return sorted(best_by_record.values(), key=lambda result: result.score, reverse=True)
