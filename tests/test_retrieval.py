from __future__ import annotations

from typing import Any

from minicpm_dream.consolidate.retrieve import Retriever
from minicpm_dream.retrieval.vector_index import VectorIndex
from minicpm_dream.store.db import Store
from tests.conftest import make_candidate


class FakeEmbedder:
    dimension = 3

    @staticmethod
    def _vector(text: str) -> list[float]:
        low = text.lower()
        values = [
            1.0 if "sqlite" in low or "database" in low else 0.0,
            1.0 if "dark" in low or "editor" in low else 0.0,
            1.0 if "dentist" in low or "appointment" in low else 0.0,
        ]
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def encode_corpus(self, texts: list[str]) -> Any:
        return [self._vector(text) for text in texts]

    def encode_queries(self, texts: list[str]) -> Any:
        return [self._vector(text) for text in texts]

    def close(self) -> None:
        return None


class FakeReranker:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [self.scores.get(doc, 0.0) for doc in documents]

    def close(self) -> None:
        return None


def _add(store: Store, text: str, section: str = "projects") -> str:
    return store.insert_memory(
        kind="fact",
        section=section,
        text=text,
        evidence=[{"episode_id": "e", "quote": text[:20], "verified": True}],
        confidence=0.9,
        salience=0.5,
    )


def test_vector_index_sync_and_search(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        sqlite_id = _add(store, "The project uses SQLite for storage.")
        _add(store, "The user prefers dark mode in editors.", section="preferences")
        index = VectorIndex(store, FakeEmbedder(), model="fake", batch_size=2)
        assert index.sync() == 2
        results = index.search(FakeEmbedder().encode_queries(["sqlite database"])[0], limit=2)
        assert results and results[0][0] == sqlite_id
        assert index.count() == 2
    finally:
        store.close()


def test_vector_index_prunes_superseded(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        mid = _add(store, "The project uses SQLite for storage.")
        index = VectorIndex(store, FakeEmbedder(), model="fake")
        index.sync()
        store.supersede(mid, valid_to="2026-11-06")
        index.sync()
        assert index.count() == 0
    finally:
        store.close()


def test_hybrid_retriever_fuses_dense_and_fts(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        sqlite_id = _add(store, "The project uses SQLite for storage.")
        _add(store, "The user prefers dark mode in editors.", section="preferences")
        embedder = FakeEmbedder()
        index = VectorIndex(store, embedder, model="fake")
        index.sync()
        retriever = Retriever(store, top_k=2, embedder=embedder, index=index)
        rows = retriever.retrieve(make_candidate("database"))
        assert rows[0]["id"] == sqlite_id
    finally:
        store.close()


def test_reranker_can_reorder_candidates(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        _add(store, "The project uses SQLite for storage.")
        dentist_id = _add(store, "The dentist appointment is on Friday.", section="open_loops")
        embedder = FakeEmbedder()
        index = VectorIndex(store, embedder, model="fake")
        index.sync()
        reranker = FakeReranker({"The dentist appointment is on Friday.": 8.0})
        retriever = Retriever(
            store, top_k=2, embedder=embedder, index=index, reranker=reranker, rerank_top_n=2
        )
        rows = retriever.retrieve(make_candidate("dentist appointment"))
        assert rows[0]["id"] == dentist_id
    finally:
        store.close()


def test_plain_retriever_still_works(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        mid = _add(store, "The project uses SQLite for storage.")
        rows = Retriever(store, top_k=3).retrieve(make_candidate("SQLite storage"))
        assert rows[0]["id"] == mid
    finally:
        store.close()
