"""Vector index over the local memory ledger (OpenBMB embeddings)."""

from __future__ import annotations

from typing import Any, Protocol

from ..store.db import Store
from ..util.logging import get_logger

log = get_logger("retrieval.index")


class EmbedderLike(Protocol):
    dimension: int | None

    def encode_corpus(self, texts: list[str]) -> Any: ...
    def encode_queries(self, texts: list[str]) -> Any: ...


class VectorIndex:
    def __init__(
        self,
        store: Store,
        embedder: EmbedderLike,
        *,
        model: str,
        batch_size: int = 8,
    ):
        self.store = store
        self.embedder = embedder
        self.model = model
        self.batch_size = max(1, batch_size)

    def sync(self) -> int:
        """Embed active memories that do not have a vector yet."""
        import numpy as np

        active = self.store.active_memories()
        existing = self.store.vector_ids(model=self.model)
        missing = [row for row in active if row["id"] not in existing]
        pruned = self.store.prune_vectors(model=self.model)
        if pruned:
            log.info("pruned %s stale vectors", pruned)
        if not missing:
            return 0
        added = 0
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            texts = [str(row["text"]) for row in batch]
            vectors = self.embedder.encode_corpus(texts)
            for row, vector in zip(batch, vectors, strict=False):
                arr = np.asarray(vector, dtype=np.float32)
                if arr.ndim != 1 or arr.size == 0:
                    continue
                self.store.upsert_vector(
                    str(row["id"]),
                    model=self.model,
                    vector=arr.tobytes(),
                    dim=int(arr.size),
                )
                added += 1
        log.info("indexed %s memories with %s", added, self.model)
        return added

    def search(self, query_vector: Any, limit: int = 24) -> list[tuple[str, float]]:
        import numpy as np

        rows = self.store.vectors(model=self.model)
        if not rows:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        ids: list[str] = []
        vectors: list[Any] = []
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            if vector.size != query.size:
                continue
            ids.append(str(row["memory_id"]))
            vectors.append(vector)
        if not vectors:
            return []
        matrix = np.vstack(vectors)
        scores = matrix @ query
        order = np.argsort(-scores)[: max(1, limit)]
        return [(ids[int(i)], float(scores[int(i)])) for i in order]

    def count(self) -> int:
        return len(self.store.vector_ids(model=self.model))
