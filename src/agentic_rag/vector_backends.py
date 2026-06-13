"""Vector index backends used by the benchmark experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class BackendUnavailable(RuntimeError):
    """Raised when an optional native vector backend is not installed."""


class VectorIndex(Protocol):
    """Minimal index surface shared by FAISS, turbovec, and test backends."""

    name: str

    def add(self, vectors: list[list[float]]) -> None:
        ...

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        ...


@dataclass
class ExactNumpyIndex:
    """Exact cosine/IP search baseline implemented with NumPy."""

    dim: int
    name: str = "exact-numpy"

    def __post_init__(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise BackendUnavailable("Install numpy to use the exact-numpy backend.") from exc

        self._np = np
        self._vectors = np.empty((0, self.dim), dtype="float32")

    def add(self, vectors: list[list[float]]) -> None:
        matrix = self._np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"expected vectors with shape (n, {self.dim})")
        self._vectors = self._np.vstack([self._vectors, matrix])

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        if len(self._vectors) == 0:
            return []
        query_vector = self._np.asarray(query, dtype="float32")
        scores = self._vectors @ query_vector
        top = self._np.argsort(-scores)[:k]
        return [(int(index), float(scores[index])) for index in top]


@dataclass
class FaissFlatIndex:
    """FAISS exact inner-product index."""

    dim: int
    name: str = "faiss-flat"

    def __post_init__(self) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise BackendUnavailable("Install faiss-cpu and numpy to use the faiss-flat backend.") from exc

        self._faiss = faiss
        self._np = np
        self._index = faiss.IndexFlatIP(self.dim)

    def add(self, vectors: list[list[float]]) -> None:
        matrix = self._np.asarray(vectors, dtype="float32")
        self._index.add(matrix)

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        query_matrix = self._np.asarray([query], dtype="float32")
        scores, indices = self._index.search(query_matrix, k)
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if index >= 0
        ]


@dataclass
class FaissPQIndex:
    """FAISS product-quantized inner-product index."""

    dim: int
    bit_width: int = 4
    name: str = "faiss-pq"

    def __post_init__(self) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise BackendUnavailable("Install faiss-cpu and numpy to use the faiss-pq backend.") from exc

        self._faiss = faiss
        self._np = np
        self._index = faiss.IndexPQ(self.dim, self.dim, self.bit_width, faiss.METRIC_INNER_PRODUCT)

    def add(self, vectors: list[list[float]]) -> None:
        matrix = self._np.asarray(vectors, dtype="float32")
        if len(matrix) < 2**self.bit_width:
            raise BackendUnavailable(
                f"faiss-pq needs at least {2**self.bit_width} vectors to train with bit_width={self.bit_width}."
            )
        self._index.train(matrix)
        self._index.add(matrix)

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        query_matrix = self._np.asarray([query], dtype="float32")
        scores, indices = self._index.search(query_matrix, k)
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if index >= 0
        ]


@dataclass
class TurboVecIndex:
    """turbovec TurboQuant index wrapper."""

    dim: int
    bit_width: int = 4
    name: str = "turbovec"

    def __post_init__(self) -> None:
        try:
            import numpy as np
            from turbovec import TurboQuantIndex
        except ImportError as exc:
            raise BackendUnavailable("Install turbovec and numpy to use the turbovec backend.") from exc

        self._np = np
        self._index = TurboQuantIndex(dim=self.dim, bit_width=self.bit_width)

    def add(self, vectors: list[list[float]]) -> None:
        matrix = self._np.asarray(vectors, dtype="float32")
        self._index.add(matrix)

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        query_vector = self._np.asarray(query, dtype="float32")
        scores, indices = self._index.search(query_vector, k=k)
        return [
            (int(index), float(score))
            for index, score in zip(indices, scores, strict=True)
            if index >= 0
        ]


def build_vector_index(name: str, dim: int, bit_width: int = 4) -> VectorIndex:
    normalized = name.lower().replace("_", "-")
    if normalized == "exact-numpy":
        return ExactNumpyIndex(dim=dim)
    if normalized == "faiss-flat":
        return FaissFlatIndex(dim=dim)
    if normalized == "faiss-pq":
        return FaissPQIndex(dim=dim, bit_width=bit_width)
    if normalized == "turbovec":
        return TurboVecIndex(dim=dim, bit_width=bit_width)
    raise ValueError(f"unknown vector backend: {name}")
