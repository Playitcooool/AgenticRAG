"""Deterministic local embeddings for backend comparison experiments."""

from __future__ import annotations

import hashlib
import math

from agentic_rag.retriever import tokenize


class HashingEmbedder:
    """A lightweight signed hashing vectorizer.

    This is not meant to beat neural embeddings. It keeps the benchmark
    reproducible and dependency-light while isolating FAISS vs turbovec index
    behavior inside the same agentic RAG architecture.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.encode_one(text) for text in texts]

    def encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        terms = tokenize(text)
        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
