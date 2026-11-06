"""Build the configured retrieval stack with graceful fallback."""

from __future__ import annotations

from ..config import Config
from ..consolidate.retrieve import Retriever
from ..store.db import Store
from ..util.logging import get_logger
from .openbmb import OpenBMBUnavailable, build_embedder, build_reranker
from .vector_index import VectorIndex

log = get_logger("retrieval.factory")


def build_retriever(cfg: Config, store: Store, *, backend_name: str = "", sync: bool = True) -> Retriever:
    rc = cfg.retrieval
    plain = Retriever(store, top_k=cfg.consolidate.top_k)
    if rc.embeddings == "off" and rc.reranker == "off":
        return plain
    if backend_name in {"mock", "none"}:
        # Tests and the offline demo must not download or load models.
        return plain
    try:
        embedder = None
        index = None
        if rc.embeddings != "off":
            embedder = build_embedder(rc)
            index = VectorIndex(
                store,
                embedder,
                model=rc.embedding_model,
                batch_size=rc.embedding_batch_size,
            )
            if sync:
                index.sync()
        reranker = build_reranker(rc) if rc.reranker != "off" else None
        return Retriever(
            store,
            top_k=cfg.consolidate.top_k,
            embedder=embedder,
            index=index,
            reranker=reranker,
            rrf_k=rc.rrf_k,
            rerank_top_n=rc.rerank_top_n,
            fts=rc.fts,
        )
    except (OpenBMBUnavailable, ImportError, OSError, RuntimeError, ValueError) as exc:
        if rc.strict:
            raise
        log.warning(
            "OpenBMB retrieval unavailable (%s); falling back to FTS5. "
            "Install with `uv pip install -e '.[openbmb]'` or set retrieval.embeddings='off'.",
            exc,
        )
        return plain
