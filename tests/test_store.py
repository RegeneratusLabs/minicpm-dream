from __future__ import annotations

from datetime import UTC, datetime

from minicpm_dream.store.db import Store


def _memory(store: Store, text: str, **kwargs):
    return store.insert_memory(
        kind=kwargs.pop("kind", "fact"),
        section=kwargs.pop("section", "projects"),
        text=text,
        evidence=[{"episode_id": "ep_1", "quote": text[:20], "verified": True}],
        confidence=0.9,
        **kwargs,
    )


def test_insert_search_and_counts(tmp_path):
    store = Store(tmp_path / "ledger.db")
    try:
        assert store.fts_enabled
        mid = _memory(store, "The miniCPM-memory project uses SQLite for storage.")
        assert store.get_memory(mid)["status"] == "active"
        rows = store.search("SQLite project")
        assert rows and rows[0]["id"] == mid
        counts = store.counts()
        assert counts["active"] == 1
        assert counts["chars"] > 0
    finally:
        store.close()


def test_supersede_and_history(tmp_path):
    store = Store(tmp_path / "ledger.db")
    try:
        old = _memory(store, "The project uses SQLite.", kind="fact")
        new = _memory(store, "The project uses PostgreSQL.", kind="fact", supersedes_id=old)
        store.supersede(old, valid_to="2026-11-06")
        old_row = store.get_memory(old)
        assert old_row["status"] == "superseded"
        assert old_row["valid_to"] == "2026-11-06"
        assert store.get_memory(new)["status"] == "active"
        assert store.counts()["active"] == 1
        versions = store.conn.execute(
            "SELECT COUNT(*) FROM memory_versions WHERE memory_id=?", (old,)
        ).fetchone()[0]
        assert versions == 1
    finally:
        store.close()


def test_forget_leaves_tombstone(tmp_path):
    store = Store(tmp_path / "ledger.db")
    try:
        mid = _memory(store, "A secret-ish personal fact.")
        assert store.forget(mid, reason="user_request")
        row = store.get_memory(mid)
        assert row["status"] == "tombstone"
        assert row["text"] == "[forgotten]"
        assert store.search("secret-ish") == []
    finally:
        store.close()


def test_proposals_and_kv(tmp_path):
    store = Store(tmp_path / "ledger.db")
    try:
        pid = store.add_proposal(
            run_id="run_1",
            kind="review",
            target="ledger",
            payload={"candidate": {"text": "x"}},
            cooldown_days=3,
        )
        pending = store.list_proposals()
        assert pending[0]["id"] == pid
        assert store.set_proposal_status(pid, "approved", reason="ok")
        assert store.proposal_counts()["approved"] == 1
        store.kv_set("cursor", "42")
        assert store.kv_get("cursor") == "42"
        until = datetime(2027, 1, 1, tzinfo=UTC)
        store.set_cooldown("k", "test", until)
        assert store.cooldown("k") is not None
    finally:
        store.close()
