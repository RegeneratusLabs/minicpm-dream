from __future__ import annotations

import pytest

from minicpm_dream.apply.guards import check_loss
from minicpm_dream.apply.ledger import LossGuardError, apply_decisions
from minicpm_dream.consolidate.reconcile import Decision
from minicpm_dream.store.db import Store
from tests.conftest import make_candidate


def _add(store: Store, text: str) -> str:
    return store.insert_memory(
        kind="fact",
        section="projects",
        text=text,
        evidence=[{"episode_id": "ep_1", "quote": text[:20], "verified": True}],
        confidence=0.9,
    )


def test_apply_add_and_evidence(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        decision = Decision(candidate=make_candidate("A durable fact."), op="ADD", reason="new", confidence=0.9)
        result = apply_decisions(store, [decision], run_id="run_1", model="mock")
        assert result.counts["active"] == 1
        assert result.memory_ids
        evidence = store.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        assert evidence == 1
    finally:
        store.close()


def test_apply_supersede_keeps_history(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        old = _add(store, "The project uses SQLite.")
        candidate = make_candidate("The project uses PostgreSQL.")
        candidate.valid_from = "2026-11-06"
        decision = Decision(candidate=candidate, op="SUPERSEDE", target_ids=[old], reason="changed", confidence=0.9)
        apply_decisions(store, [decision], run_id="run_2", model="mock")
        assert store.get_memory(old)["status"] == "superseded"
        active = store.active_memories()
        assert len(active) == 1 and active[0]["supersedes_id"] == old
        assert active[0]["valid_from"] == "2026-11-06"
    finally:
        store.close()


def test_defer_creates_proposal(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        decision = Decision(candidate=make_candidate("Uncertain fact."), op="DEFER", reason="uncertain", confidence=0.2)
        result = apply_decisions(store, [decision], run_id="run_3", model="mock")
        assert result.proposals
        assert store.proposal_counts()["pending"] == 1
        assert store.counts()["active"] == 0
    finally:
        store.close()


def test_loss_guard_rolls_back(tmp_path):
    store = Store(tmp_path / "l.db")
    try:
        ids = [_add(store, f"Fact number {i} about the project.") for i in range(8)]
        candidate = make_candidate("One merged fact about the project.")
        decision = Decision(candidate=candidate, op="MERGE", target_ids=ids[:4], reason="implied", confidence=0.9)
        with pytest.raises(LossGuardError):
            apply_decisions(store, [decision], run_id="run_4", model="mock")
        assert store.counts()["active"] == 8
        assert all(store.get_memory(mid)["status"] == "active" for mid in ids)
    finally:
        store.close()


def test_check_loss_thresholds():
    ok = check_loss({"entries": 10, "chars": 1000}, {"entries": 9, "chars": 950})
    assert ok.ok
    bad = check_loss({"entries": 10, "chars": 1000}, {"entries": 5, "chars": 1000})
    assert not bad.ok
