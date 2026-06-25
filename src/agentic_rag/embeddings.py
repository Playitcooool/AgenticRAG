"""Deterministic local embeddings for backend comparison experiments."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

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


class LocalSentenceTransformerEmbedder:
    """Embedding model loaded from a local SentenceTransformers-compatible path."""

    def __init__(self, model_path: str | Path, batch_size: int = 32):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local embedding model path does not exist: {self.model_path}. "
                "Download or place EmbeddingGemma there before running vector benchmarks."
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install the bench extra to use local embedding models: uv sync --extra bench") from exc

        self.model = SentenceTransformer(str(self.model_path), local_files_only=True, device="cpu")
        self.batch_size = batch_size
        self.dim: int | None = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = [list(map(float, vector)) for vector in vectors]
        if vectors and self.dim is None:
            self.dim = len(vectors[0])
        return vectors

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
