from __future__ import annotations

from pathlib import Path

from minicpm_dream.apply.exports import (
    build_bundle,
    read_entries,
    render_entries,
    split_entries,
    unique_anchor,
    write_bundle,
)
from minicpm_dream.apply.hermes_memory import apply_actions
from minicpm_dream.store.db import Store


def test_split_and_render_round_trip():
    text = "First entry.\n§\nSecond entry.\n§\nThird entry."
    entries = split_entries(text)
    assert entries == ["First entry.", "Second entry.", "Third entry."]
    assert render_entries(entries).count("§") == 2


def test_unique_anchor():
    entries = ["Josh prefers dark mode in editors.", "Josh lives in Brisbane."]
    anchor = unique_anchor(entries[0], entries)
    assert anchor in entries[0]
    assert sum(1 for entry in entries if anchor in entry) == 1


def test_build_bundle_and_write(tmp_path):
    store = Store(tmp_path / "l.db")
    memory_path = tmp_path / "MEMORY.md"
    user_path = tmp_path / "USER.md"
    memory_path.write_text("", encoding="utf-8")
    user_path.write_text("", encoding="utf-8")
    try:
        store.insert_memory(
            kind="fact", section="projects", text="The project uses SQLite.",
            evidence=[{"episode_id": "e", "quote": "uses SQLite", "verified": True}], salience=0.9,
        )
        store.insert_memory(
            kind="preference", section="preferences", text="Josh prefers dark mode.",
            evidence=[{"episode_id": "e", "quote": "prefers dark mode", "verified": True}], salience=0.9,
        )
        bundle = build_bundle(
            store, memory_path=memory_path, user_path=user_path,
            memory_limit=2200, user_limit=1375,
        )
        assert "SQLite" in bundle.memory_text
        assert "dark mode" in bundle.user_text
        assert len(bundle.actions) == 2
        paths = write_bundle(bundle, tmp_path / "exports")
        assert Path(paths["review"]).is_file()
        assert Path(paths["memory"]).read_text().strip() == "The project uses SQLite."
    finally:
        store.close()


def test_apply_actions_replace_and_limit(tmp_path):
    path = tmp_path / "MEMORY.md"
    path.write_text("old entry here.\n§\nother entry.", encoding="utf-8")
    report = apply_actions(
        path,
        [{"action": "replace", "old_text": "old entry here", "content": "new entry here."}],
        target="memory",
        limit=1000,
        backup_dir=tmp_path / "backups",
    )
    assert report.ok
    assert read_entries(path) == ["new entry here.", "other entry."]
    assert Path(report.backup).is_file()

    report = apply_actions(
        path,
        [{"action": "add", "content": "x" * 500}],
        target="memory",
        limit=100,
        backup_dir=tmp_path / "backups",
    )
    assert not report.ok
    assert "over the" in report.error
    assert read_entries(path) == ["new entry here.", "other entry."]


def test_apply_actions_rejects_ambiguous_anchor(tmp_path):
    path = tmp_path / "USER.md"
    path.write_text("same text\n§\nsame text", encoding="utf-8")
    report = apply_actions(
        path,
        [{"action": "remove", "old_text": "same text"}],
        target="user",
        limit=1000,
        backup_dir=tmp_path / "backups",
    )
    assert not report.ok
    assert "matched 2" in report.error
