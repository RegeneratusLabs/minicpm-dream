from __future__ import annotations

from minicpm_dream.backends.mock import MockBackend
from minicpm_dream.config import ConsolidateCfg
from minicpm_dream.consolidate.policy import apply_budget
from minicpm_dream.consolidate.reconcile import reconcile_candidate
from minicpm_dream.store.db import Store
from tests.conftest import make_candidate


def test_add_without_existing_makes_no_model_call():
    backend = MockBackend()
    decision = reconcile_candidate(backend, make_candidate("A new fact."), [], ConsolidateCfg())
    assert decision.op == "ADD"
    assert backend.calls == []


def test_exact_duplicate_is_noop(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        mid = store.insert_memory(
            kind="fact", section="projects", text="The project uses SQLite.",
            evidence=[{"episode_id": "e", "quote": "q", "verified": True}],
        )
        rows = [store.get_memory(mid)]
    finally:
        store.close()
    decision = reconcile_candidate(MockBackend(), make_candidate("The project uses SQLite."), rows, ConsolidateCfg())
    assert decision.op == "NOOP"
    assert decision.target_ids == [mid]


def test_invented_target_id_is_reviewed(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        mid = store.insert_memory(
            kind="fact", section="projects", text="The project uses SQLite.",
            evidence=[{"episode_id": "e", "quote": "q", "verified": True}],
        )
        rows = [store.get_memory(mid)]
    finally:
        store.close()
    backend = MockBackend(
        {"reconcile": [{"op": "UPDATE", "target_ids": ["mem_invented"], "merged_text": "The project uses SQLite and FTS.", "reason": "changed", "confidence": 0.9}]}
    )
    decision = reconcile_candidate(backend, make_candidate("The project uses SQLite and FTS."), rows, ConsolidateCfg())
    assert decision.requires_review
    assert "invented target id" in " ".join(decision.errors)


def test_merge_rejects_new_numbers(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        a = store.insert_memory(kind="fact", section="projects", text="The project uses Python 3.12.",
                                evidence=[{"episode_id": "e", "quote": "q", "verified": True}])
        b = store.insert_memory(kind="fact", section="projects", text="The project stores data in SQLite.",
                                evidence=[{"episode_id": "e", "quote": "q", "verified": True}])
        rows = [store.get_memory(a), store.get_memory(b)]
    finally:
        store.close()
    backend = MockBackend(
        {"reconcile": [{"op": "MERGE", "target_ids": [a, b], "merged_text": "The project uses Python 3.14 and SQLite.", "reason": "implied", "confidence": 0.9}]}
    )
    decision = reconcile_candidate(backend, make_candidate("The project uses Python 3.13 and SQLite."), rows, ConsolidateCfg())
    assert decision.requires_review
    assert any("unsupported numbers" in error for error in decision.errors)


def test_budget_limits_new_and_total():
    cfg = ConsolidateCfg(max_changes_per_run=2, max_new_per_run=1)
    decisions = [make_candidate(f"Fact {i}") for i in range(5)]
    for decision in decisions:
        decision.__class__  # noqa: B018
    from minicpm_dream.consolidate.reconcile import Decision

    wrapped = [Decision(candidate=c, op="ADD", reason="new", confidence=0.9) for c in decisions]
    applied, deferred = apply_budget(wrapped, cfg)
    adds = [d for d in applied if d.op == "ADD"]
    assert len(adds) == 1
    assert len(deferred) == 4
