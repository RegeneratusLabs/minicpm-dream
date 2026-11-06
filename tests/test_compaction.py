from __future__ import annotations

from minicpm_dream.apply.compaction import consolidate_active
from minicpm_dream.config import ConsolidateCfg
from minicpm_dream.consolidate.retrieve import Retriever
from minicpm_dream.consolidate.sequential import reconcile_and_apply
from minicpm_dream.store.db import Store
from tests.conftest import make_candidate


def _add(store: Store, text: str, section: str = "preferences") -> str:
    return store.insert_memory(
        kind="fact", section=section, text=text,
        evidence=[{"episode_id": "e", "quote": text[:30], "verified": True}],
        confidence=0.9, salience=0.5,
    )


def test_compaction_archives_contained_duplicate(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        long_id = _add(store, "Josh prefers dark mode in all editors and dislikes long explanations.")
        short_id = _add(store, "The user dislikes long explanations.")
        result = consolidate_active(store)
        assert result["archived"] == 1
        assert store.get_memory(long_id)["status"] == "active"
        assert store.get_memory(short_id)["status"] == "archived"
        evidence = store.conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE memory_id=?", (long_id,)
        ).fetchone()[0]
        assert evidence >= 1
    finally:
        store.close()


def test_compaction_keeps_distinct_facts(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        _add(store, "Josh prefers dark mode.")
        _add(store, "Josh lives in Brisbane.")
        result = consolidate_active(store)
        assert result["archived"] == 0
    finally:
        store.close()


def test_sequential_reconcile_sees_the_previous_candidate(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        candidate = make_candidate("The project uses SQLite for storage.", section="projects")
        decisions, applied, _usage = reconcile_and_apply(
            store,
            __import__("minicpm_dream.backends.mock", fromlist=["MockBackend"]).MockBackend(),
            [candidate, candidate],
            Retriever(store, top_k=3),
            ConsolidateCfg(),
            run_id="run_seq",
            model="mock",
        )
        assert len(applied.memory_ids) == 1
        assert decisions[1].op == "NOOP"
        assert len(store.active_memories()) == 1
    finally:
        store.close()


def test_derived_candidate_bypasses_llm_reconcile(tmp_path):
    from minicpm_dream.backends.mock import MockBackend

    store = Store(tmp_path / "l.db")
    try:
        candidate = make_candidate("Cameron is the user's brother.", section="people")
        candidate.origin = "derived"
        backend = MockBackend()
        _decisions, applied, _usage = reconcile_and_apply(
            store, backend, [candidate], Retriever(store, top_k=3),
            ConsolidateCfg(), run_id="run_derived", model="mock",
        )
        assert len(applied.memory_ids) == 1
        assert backend.calls == []
        assert store.active_memories()[0]["origin"] == "inferred"
    finally:
        store.close()


def test_compaction_prefers_postgres_fact_over_sqlite(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        sqlite_id = _add(store, "The miniCPM-memory project uses Python 3.12 and SQLite for its ledger.", "projects")
        postgres_id = _add(store, "The miniCPM-memory project now uses PostgreSQL for its ledger.", "projects")
        result = consolidate_active(store)
        assert result["archived"] == 1
        assert store.get_memory(postgres_id)["status"] == "active"
        assert store.get_memory(sqlite_id)["status"] == "archived"
    finally:
        store.close()


def test_compaction_archives_migration_event_when_project_fact_exists(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        event_id = _add(
            store,
            "The miniCPM-memory ledger was migrated from SQLite to PostgreSQL on 2026-11-04.",
            "open_loops",
        )
        fact_id = _add(
            store,
            "The miniCPM-memory project now uses PostgreSQL for its ledger.",
            "projects",
        )
        result = consolidate_active(store)
        assert result["archived"] == 1
        assert store.get_memory(fact_id)["status"] == "active"
        assert store.get_memory(event_id)["status"] == "archived"
    finally:
        store.close()
