"""Hybrid retrieval: FTS5 + OpenBMB dense vectors + optional reranker."""

from __future__ import annotations

import math
from typing import Any

from ..extract.validate import Candidate
from ..store.db import Store
from ..util.logging import get_logger
from ..util.timeutil import human_age, parse_timestamp, utc_now

log = get_logger("consolidate.retrieve")


class Retriever:
    def __init__(
        self,
        store: Store,
        *,
        top_k: int = 6,
        embedder: Any = None,
        index: Any = None,
        reranker: Any = None,
        rrf_k: int = 60,
        rerank_top_n: int = 6,
        fts: bool = True,
    ):
        self.store = store
        self.top_k = top_k
        self.embedder = embedder
        self.index = index
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n
        self.fts = fts

    @property
    def hybrid(self) -> bool:
        return self.embedder is not None and self.index is not None

    def sync(self) -> int:
        if self.index is None:
            return 0
        try:
            return int(self.index.sync())
        except Exception as exc:  # noqa: BLE001
            log.warning("vector index sync failed: %s", exc)
            return 0

    def close(self) -> None:
        for component in (self.reranker, self.embedder):
            if component is not None and hasattr(component, "close"):
                try:
                    component.close()
                except Exception:  # noqa: BLE001
                    pass

    def query(self, candidate: Candidate) -> str:
        parts = [candidate.text, candidate.subject, " ".join(candidate.entities)]
        return " ".join(part for part in parts if part)

    def retrieve(self, candidate: Candidate) -> list[dict[str, Any]]:
        query = self.query(candidate)
        fused: dict[str, float] = {}
        rows: dict[str, dict[str, Any]] = {}
        limit = max(self.top_k * 4, 12)

        if self.fts:
            for rank, row in enumerate(self.store.search(query, limit=limit)):
                mid = str(row["id"])
                rows[mid] = row
                fused[mid] = fused.get(mid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        if self.hybrid:
            try:
                query_vector = self.embedder.encode_queries([query])[0]
                for rank, (mid, _score) in enumerate(self.index.search(query_vector, limit=limit)):
                    memory_row = self.store.get_memory(mid)
                    if memory_row is None or memory_row.get("status") != "active":
                        continue
                    rows[mid] = memory_row
                    fused[mid] = fused.get(mid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            except Exception as exc:  # noqa: BLE001
                log.warning("dense retrieval failed for one candidate: %s", exc)

        if not fused:
            return []

        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        if self.reranker is not None and self.rerank_top_n > 0:
            head = ordered[: max(1, self.rerank_top_n)]
            docs = [str(rows[mid].get("text", "")) for mid, _ in head]
            try:
                rerank_scores = self.reranker.score(query, docs)
                combined: dict[str, float] = {}
                for (mid, rrf_score), rerank_score in zip(head, rerank_scores, strict=False):
                    rerank_norm = 1.0 / (1.0 + math.exp(-float(rerank_score)))
                    combined[mid] = 0.6 * (rrf_score / max(ordered[0][1], 1e-9)) + 0.4 * rerank_norm
                for mid, score in ordered:
                    combined.setdefault(mid, 0.5 * score / max(ordered[0][1], 1e-9))
                ordered = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("reranker failed for one candidate: %s", exc)

        now = utc_now()
        out: list[dict[str, Any]] = []
        for mid, _score in ordered[: self.top_k]:
            row = rows[mid]
            try:
                row["age"] = human_age(parse_timestamp(row["recorded_at"]), now)
            except (ValueError, TypeError, KeyError):
                row["age"] = "unknown"
            out.append(row)
        return out
