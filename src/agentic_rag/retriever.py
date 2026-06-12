"""Deterministic lexical retriever used by the RAG agent."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from agentic_rag.types import PatientRecord, SearchResult

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into normalized lexical terms."""

    return TOKEN_RE.findall(text.lower())


class LexicalRetriever:
    """Small BM25-like retriever without external dependencies."""

    def __init__(self, records: Iterable[PatientRecord]):
        self.records = list(records)
        self._term_counts = [Counter(tokenize(record.text)) for record in self.records]
        self._doc_freq = Counter()
        for counts in self._term_counts:
            self._doc_freq.update(counts.keys())
        self._avg_len = (
            sum(sum(counts.values()) for counts in self._term_counts) / len(self._term_counts)
            if self._term_counts
            else 1.0
        )

    @classmethod
    def from_json_chunks(cls, path: Path, source: str | None = None) -> "LexicalRetriever":
        """Load records from the benchmark `chunks.json` id:text format."""

        raw_chunks = json.loads(path.read_text(encoding="utf-8"))
        records = []
        for raw_chunk in raw_chunks:
            record_id, text = raw_chunk.split(":", 1)
            records.append(
                PatientRecord(
                    record_id=record_id.strip(),
                    source=source or path.parent.name,
                    text=text.strip(),
                )
            )
        return cls(records)

    def search(self, queries: Iterable[str], top_k: int = 4) -> list[SearchResult]:
        """Retrieve a deduplicated set of snippets for all query fanouts."""

        best_by_record: dict[str, SearchResult] = {}
        for query in queries:
            for result in self._search_one(query, top_k=top_k):
                current = best_by_record.get(result.record.record_id)
                if current is None or result.score > current.score:
                    best_by_record[result.record.record_id] = result
        return sorted(best_by_record.values(), key=lambda result: result.score, reverse=True)

    def _search_one(self, query: str, top_k: int) -> list[SearchResult]:
        query_terms = tokenize(query)
        scored = []
        for record, counts in zip(self.records, self._term_counts, strict=True):
            score = self._score(query_terms, counts)
            if score > 0:
                scored.append(SearchResult(query=query, record=record, score=score))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

    def _score(self, query_terms: list[str], counts: Counter[str]) -> float:
        doc_len = sum(counts.values()) or 1
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (len(self.records) - self._doc_freq[term] + 0.5) / (self._doc_freq[term] + 0.5))
            norm = freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * doc_len / self._avg_len))
            score += idf * norm
        return score
