from collections.abc import Sequence
from typing import Protocol

import numpy as np
from anyio import to_thread


class Embedder(Protocol):
    """Turns text into L2-normalized vectors. Implementations are synchronous and CPU-bound."""

    @property
    def dim(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """Local ONNX embedding model via fastembed: no network calls at query time, no API keys."""

    def __init__(
        self, model_name: str, cache_dir: str | None = None, threads: int | None = None
    ) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name, cache_dir=cache_dir, threads=threads)
        self._dim = len(self.embed_queries(["dimension probe"])[0])

    @property
    def dim(self) -> int:
        return self._dim

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return _normalize(self._model.passage_embed(list(texts)))

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return _normalize(self._model.query_embed(list(texts)))


def _normalize(vectors: object) -> list[list[float]]:
    matrix = np.asarray(list(vectors), dtype=np.float32)  # type: ignore[call-overload]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized: list[list[float]] = (matrix / np.maximum(norms, 1e-12)).tolist()
    return normalized


async def embed_passages(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    return await to_thread.run_sync(embedder.embed_passages, texts)


async def embed_queries(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    return await to_thread.run_sync(embedder.embed_queries, texts)
