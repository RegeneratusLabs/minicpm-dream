from __future__ import annotations

from minicpm_dream.apply.exports import build_bundle, render_entries, split_entries
from minicpm_dream.store.db import Store


def _memories(store: Store, specs: list[tuple[str, str, float, str]]) -> list[str]:
    ids = []
    for section, text, salience, kind in specs:
        ids.append(
            store.insert_memory(
                kind=kind,
                section=section,
                text=text,
                evidence=[{"episode_id": "e", "quote": text, "verified": True}],
                salience=salience,
                confidence=0.9,
            )
        )
    return ids


def test_pinned_identity_survives_a_full_profile(tmp_path):
    store = Store(tmp_path / "l.db")
    memory_path = tmp_path / "MEMORY.md"
    user_path = tmp_path / "USER.md"
    memory_path.write_text("", encoding="utf-8")
    user_path.write_text("Josh is a developer in Brisbane.", encoding="utf-8")
    try:
        _memories(
            store,
            [
                ("identity", "Josh is a developer based in Brisbane.", 0.9, "fact"),
                ("projects", "The project uses Python 3.12 with SQLite.", 0.6, "fact"),
                ("projects", "The project keeps a local ledger at ~/.local/share.", 0.4, "fact"),
            ],
        )
        bundle = build_bundle(
            store,
            memory_path=memory_path,
            user_path=user_path,
            memory_limit=120,
            user_limit=1375,
            pinned_sections=["identity", "preferences"],
            compact=True,
        )
        assert bundle.user_entries == ["Josh is a developer in Brisbane."]  # human text wins
        assert not any(item["target"] == "user" for item in bundle.evicted)
        assert [a for a in bundle.actions if a["target"] == "user"] == []
        assert bundle.pressure["user"]["fill"] < 1.0
    finally:
        store.close()


def test_low_salience_entry_is_evicted_before_a_high_salience_one_is_refused(tmp_path):
    store = Store(tmp_path / "l.db")
    memory_path = tmp_path / "MEMORY.md"
    user_path = tmp_path / "USER.md"
    user_path.write_text("", encoding="utf-8")
    try:
        _memories(
            store,
            [
                ("projects", "The old build script lives in tools/legacy.sh.", 0.2, "fact"),
                ("projects", "The ledger is SQLite in ~/.local/share/minicpm-dream.", 0.9, "fact"),
            ],
        )
        memory_path.write_text("The old build script lives in tools/legacy.sh.\n", encoding="utf-8")
        bundle = build_bundle(
            store,
            memory_path=memory_path,
            user_path=user_path,
            memory_limit=70,
            user_limit=1375,
            pinned_sections=["identity", "preferences"],
            compact=True,
        )
        assert any(item["text"].startswith("The old build script") for item in bundle.evicted)
        assert any("SQLite" in entry for entry in bundle.memory_entries)
        assert len(render_entries(bundle.memory_entries)) <= 70
        assert any(action["action"] == "remove" for action in bundle.actions)
        assert any(action["action"] == "add" for action in bundle.actions)
    finally:
        store.close()


def test_no_compact_skips_instead_of_evicting(tmp_path):
    store = Store(tmp_path / "l.db")
    memory_path = tmp_path / "MEMORY.md"
    user_path = tmp_path / "USER.md"
    user_path.write_text("", encoding="utf-8")
    memory_path.write_text("The old build script lives in tools/legacy.sh.\n", encoding="utf-8")
    try:
        _memories(
            store,
            [
                ("projects", "The old build script lives in tools/legacy.sh.", 0.2, "fact"),
                ("projects", "The ledger is SQLite in ~/.local/share/minicpm-dream.", 0.9, "fact"),
            ],
        )
        bundle = build_bundle(
            store,
            memory_path=memory_path,
            user_path=user_path,
            memory_limit=70,
            user_limit=1375,
            compact=False,
        )
        assert bundle.evicted == []
        assert bundle.skipped
        assert bundle.memory_entries == ["The old build script lives in tools/legacy.sh."]
    finally:
        store.close()


def test_human_written_entries_are_never_evicted(tmp_path):
    store = Store(tmp_path / "l.db")
    memory_path = tmp_path / "MEMORY.md"
    user_path = tmp_path / "USER.md"
    user_path.write_text("", encoding="utf-8")
    memory_path.write_text("A note I typed by hand and want to keep.\n", encoding="utf-8")
    try:
        _memories(store, [("projects", "The ledger uses SQLite.", 0.9, "fact")])
        bundle = build_bundle(
            store,
            memory_path=memory_path,
            user_path=user_path,
            memory_limit=60,
            user_limit=1375,
            compact=True,
        )
        assert bundle.memory_entries == ["A note I typed by hand and want to keep."]
        assert bundle.evicted == []
        assert bundle.actions == []
    finally:
        store.close()


def test_pinned_overflow_is_reported_not_silently_truncated(tmp_path):
    store = Store(tmp_path / "l.db")
    memory_path = tmp_path / "MEMORY.md"
    user_path = tmp_path / "USER.md"
    memory_path.write_text("", encoding="utf-8")
    user_path.write_text("x" * 200, encoding="utf-8")
    try:
        _memories(store, [("projects", "The ledger uses SQLite.", 0.9, "fact")])
        bundle = build_bundle(
            store,
            memory_path=memory_path,
            user_path=user_path,
            memory_limit=200,
            user_limit=100,
            compact=True,
        )
        assert bundle.pressure["user"]["over_pinned"] is True
        assert bundle.user_entries == ["x" * 200]
        assert split_entries(user_path.read_text(encoding="utf-8")) == ["x" * 200]
    finally:
        store.close()


def test_large_ledger_planning_is_bounded_and_summarises_the_overflow(tmp_path):
    store = Store(tmp_path / "l.db")
    memory_path = tmp_path / "MEMORY.md"
    user_path = tmp_path / "USER.md"
    user_path.write_text("", encoding="utf-8")
    memory_path.write_text(
        "\n§\n".join(
            f"Hand-written note number {index} that stays in the profile forever."
            for index in range(40)
        ),
        encoding="utf-8",
    )
    try:
        for index in range(500):
            store.insert_memory(
                kind="fact",
                section="projects",
                text=f"Generated memory {index} about topic{index} and its bespoke setup.",
                evidence=[{"episode_id": "e", "quote": f"generated {index}", "verified": True}],
                salience=0.5,
            )
        bundle = build_bundle(
            store,
            memory_path=memory_path,
            user_path=user_path,
            memory_limit=800,
            user_limit=1375,
            compact=True,
        )
        skipped = bundle.skipped
        assert len(skipped) <= 51  # capped so review.json stays readable
        assert any("more candidates did not fit" in item["reason"] for item in skipped)
        assert bundle.pressure["memory"]["over_pinned"] is True
    finally:
        store.close()
