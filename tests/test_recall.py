from __future__ import annotations

import json
from datetime import timedelta

from minicpm_dream.recall import RecallService
from minicpm_dream.store.db import Store
from minicpm_dream.util.timeutil import iso_utc, utc_now


def _store(cfg) -> Store:
    return Store(cfg.paths.db)


def _add(store: Store, text: str, *, section: str = "preferences", kind: str = "preference",
         salience: float = 0.8, quote: str = "", evidence: bool = True) -> str:
    rows = (
        [{"episode_id": "ep_1", "quote": quote or text, "verified": True}] if evidence else []
    )
    return store.insert_memory(
        kind=kind, section=section, text=text, evidence=rows, salience=salience, confidence=0.9
    )


def test_recall_returns_hits_with_receipts(cfg):
    with _store(cfg) as store:
        _add(store, "Josh prefers dark mode in all editors.", quote="I prefer dark mode in all my editors")
    service = RecallService(cfg)
    try:
        result = service.recall("what editor theme does josh prefer", k=3)
    finally:
        service.close()
    assert not result.abstained
    assert result.hits
    top = result.hits[0].as_dict()
    assert "dark mode" in top["text"]
    assert top["receipts"][0]["quote"] == "I prefer dark mode in all my editors"
    assert result.elapsed_ms >= 0
    assert result.detail == "fast"


def test_recall_abstains_when_nothing_matches(cfg):
    with _store(cfg) as store:
        _add(store, "Josh prefers dark mode in all editors.")
    service = RecallService(cfg)
    try:
        result = service.recall("which kubernetes ingress controller is used", k=3)
    finally:
        service.close()
    assert result.abstained
    assert result.reason


def test_recall_reports_a_superseded_fact_and_its_replacement(cfg):
    with _store(cfg) as store:
        old = _add(store, "The ledger uses SQLite for storage.", section="projects", kind="fact")
        store.supersede(old, valid_to="2026-11-04")
        new = store.insert_memory(
            kind="fact", section="projects", text="The ledger uses PostgreSQL for storage.",
            evidence=[{"episode_id": "ep_2", "quote": "we moved to postgres", "verified": True}],
            supersedes_id=old, salience=0.8,
        )
    service = RecallService(cfg)
    try:
        current = service.recall("which database does the ledger use", k=5)
        assert all(hit.memory["status"] == "active" for hit in current.hits)
        assert any("PostgreSQL" in hit.memory["text"] for hit in current.hits)
        history = service.recall("which database does the ledger use", k=5, include_superseded=True)
        old_hits = [hit for hit in history.hits if hit.memory["id"] == old]
        assert old_hits and "superseded" in old_hits[0].flags[0]
        assert old_hits[0].superseded_by == new
        assert "Replaced on" in old_hits[0].caveat
    finally:
        service.close()


def test_recall_flags_stale_and_contested(cfg):
    with _store(cfg) as store:
        mid = _add(store, "The homelab runs Ubuntu 24.04.", section="environment", kind="fact")
        old = iso_utc(utc_now() - timedelta(days=400))
        store.conn.execute("UPDATE memories SET recorded_at=? WHERE id=?", (old, mid))
        store.add_proposal(
            run_id="run_x",
            kind="memory",
            target="memory",
            payload={"text": "homelab", "decision": {"target_ids": [mid]}},
        )
    service = RecallService(cfg, stale_days=180)
    try:
        result = service.recall("what does the homelab run", k=3)
    finally:
        service.close()
    flags = result.hits[0].flags
    assert any(flag.startswith("stale") for flag in flags)
    assert any(flag.startswith("contested") for flag in flags)


def test_remember_adds_dedupes_and_supersedes(cfg):
    service = RecallService(cfg)
    try:
        first = service.remember("Josh prefers dark mode in editors.", section="preferences",
                                 kind="preference")
        assert first["ok"] and first["op"] == "ADD"
        again = service.remember("Josh prefers dark mode in editors.", section="preferences",
                                 kind="preference")
        assert again["op"] == "NOOP" and again["memory_id"] == first["memory_id"]
        with_quote = service.remember(
            "Josh prefers dark mode in all editors.",
            section="preferences",
            kind="preference",
            quote="I prefer dark mode in all my editors",
        )
        assert with_quote["ok"] and with_quote["op"] == "UPDATE"
        assert with_quote["supersedes"] == first["memory_id"]
        with Store(cfg.paths.db) as store:
            old = store.get_memory(first["memory_id"])
            assert old["status"] == "superseded"
            assert old["valid_to"]
    finally:
        service.close()


def test_remember_rejects_secrets_and_meta(cfg):
    service = RecallService(cfg)
    try:
        secret = service.remember("The API key is sk-abcdefghijklmnopqrstuvwxyz123456")
        assert not secret["ok"]
        meta = service.remember("The user is asking about lunch.")
        assert not meta["ok"]
        retraction = service.remember("I no longer use Ollama.")
        assert not retraction["ok"]
    finally:
        service.close()


def test_explain_shows_receipts_and_chain(cfg):
    with _store(cfg) as store:
        old = _add(store, "The project uses SQLite.", section="projects", kind="fact")
        store.supersede(old, valid_to="2026-11-04")
        new = store.insert_memory(
            kind="fact", section="projects", text="The project uses PostgreSQL.",
            evidence=[{"episode_id": "ep_2", "quote": "moved to postgres", "verified": True}],
            supersedes_id=old,
        )
    service = RecallService(cfg)
    try:
        payload = service.explain(new)
        status = json.loads(json.dumps(service.status(), default=str))
    finally:
        service.close()
    assert payload is not None
    assert payload["receipts"][0]["quote"] == "moved to postgres"
    assert payload["supersedes_chain"][0]["memory_id"] == old
    assert "memories" in status


def test_abstention_uses_the_best_covering_hit_not_just_the_top_row(cfg):
    with _store(cfg) as store:
        store.insert_memory(
            kind="fact", section="environment", text="The homelab host is an OptiPlex 7060.",
            evidence=[], salience=1.0, confidence=1.0,
        )
        store.insert_memory(
            kind="fact", section="environment", text="The homelab runs Ubuntu 24.04.",
            evidence=[], salience=0.0, confidence=0.5,
        )
    service = RecallService(cfg)
    try:
        result = service.recall("what operating system version does the homelab run", k=5)
    finally:
        service.close()
    assert not result.abstained
    assert any("Ubuntu" in hit.memory["text"] for hit in result.hits)
