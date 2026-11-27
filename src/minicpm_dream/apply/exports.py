"""Build Hermes-format candidate memory files and a review manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..store.db import Store
from ..util.atomic import write_json_atomic, write_text_atomic
from ..util.text import normalize_text
from ..util.timeutil import iso_utc, utc_now
from .profile import plan_profile

USER_SECTIONS = {"identity", "preferences"}
MEMORY_SECTIONS = {"environment", "projects", "procedures", "lessons", "people"}
DIARY_ONLY = {"open_loops"}
SEP = "\n§\n"
DEFAULT_PINNED_SECTIONS = {"identity", "preferences"}


def split_entries(text: str) -> list[str]:
    if not text.strip():
        return []
    if "§" in text:
        parts = re.split(r"\n?\s*§\s*\n?", text)
    else:
        parts = re.split(r"\n\s*\n", text)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def render_entries(entries: list[str]) -> str:
    return SEP.join(entries) + ("\n" if entries else "")


def read_entries(path: str | Path) -> list[str]:
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    return split_entries(p.read_text(encoding="utf-8"))


def target_for_section(section: str) -> str | None:
    if section in USER_SECTIONS:
        return "user"
    if section in MEMORY_SECTIONS:
        return "memory"
    return None


def unique_anchor(entry: str, all_entries: list[str]) -> str:
    words = entry.split()
    for size in range(min(8, len(words)), 0, -1):
        for start in range(0, len(words) - size + 1):
            candidate = " ".join(words[start : start + size])
            if len(candidate) >= 12 and sum(1 for other in all_entries if candidate in other) == 1:
                return candidate
    return entry


def _entry_for_memory(memory: dict[str, Any]) -> str:
    text = normalize_text(str(memory.get("text", "")))
    if memory.get("kind") == "event" and memory.get("valid_from") and "(" not in text[-12:]:
        text = f"{text} ({memory['valid_from']})"
    return text[:300]


@dataclass
class ExportBundle:
    memory_entries: list[str] = field(default_factory=list)
    user_entries: list[str] = field(default_factory=list)
    memory_text: str = ""
    user_text: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    evicted: list[dict[str, Any]] = field(default_factory=list)
    limits: dict[str, int] = field(default_factory=dict)
    pressure: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def as_review(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "limits": self.limits,
            "memory": {"entries": self.memory_entries, "content": self.memory_text},
            "user": {"entries": self.user_entries, "content": self.user_text},
            "actions": self.actions,
            "skipped": self.skipped,
            "evicted": self.evicted,
            "pressure": self.pressure,
        }


def build_bundle(
    store: Store,
    *,
    memory_path: str | Path,
    user_path: str | Path,
    memory_limit: int,
    user_limit: int,
    pinned_sections: list[str] | None = None,
    compact: bool = True,
) -> ExportBundle:
    existing_memory = read_entries(memory_path)
    existing_user = read_entries(user_path)
    active = store.active_memories()
    superseded = [
        dict(row)
        for row in store.conn.execute(
            "SELECT * FROM memories WHERE status IN ('superseded','archived','tombstone')"
        ).fetchall()
    ]
    pinned = set(pinned_sections or DEFAULT_PINNED_SECTIONS)

    by_target: dict[str, list[dict[str, Any]]] = {"memory": [], "user": []}
    for memory in active:
        target = target_for_section(str(memory.get("section")))
        if target:
            by_target[target].append(memory)

    bundle = ExportBundle(
        limits={"memory": memory_limit, "user": user_limit},
        generated_at=iso_utc(utc_now()),
    )
    for target, limit, existing in (
        ("memory", memory_limit, existing_memory),
        ("user", user_limit, existing_user),
    ):
        plan = plan_profile(
            target=target,
            existing=existing,
            memories=by_target[target],
            superseded=superseded,
            limit=limit,
            entry_for=_entry_for_memory,
            render=render_entries,
            anchor_for=unique_anchor,
            pinned_sections=pinned,
            compact=compact,
        )
        bundle.actions.extend(plan.actions)
        bundle.skipped.extend(plan.skipped)
        bundle.evicted.extend(plan.evicted)
        if target == "memory":
            bundle.memory_entries = plan.entries
            bundle.memory_text = render_entries(plan.entries)
        else:
            bundle.user_entries = plan.entries
            bundle.user_text = render_entries(plan.entries)
        bundle.pressure[target] = plan.pressure
    return bundle


def write_bundle(bundle: ExportBundle, exports_dir: str | Path) -> dict[str, str]:
    out = Path(exports_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    try:
        out.chmod(0o700)
    except OSError:
        pass
    paths = {
        "memory": write_text_atomic(out / "MEMORY.candidate.md", bundle.memory_text),
        "user": write_text_atomic(out / "USER.candidate.md", bundle.user_text),
        "review": write_json_atomic(out / "review.json", bundle.as_review()),
    }
    return {k: str(v) for k, v in paths.items()}


def build_and_write(
    store: Store,
    cfg: Config,
    *,
    memory_limit: int | None = None,
    user_limit: int | None = None,
    compact: bool | None = None,
) -> tuple[ExportBundle, dict[str, str]]:
    memory_path = Path(cfg.hermes.home) / cfg.hermes.memory_file
    user_path = Path(cfg.hermes.home) / cfg.hermes.user_file
    bundle = build_bundle(
        store,
        memory_path=memory_path,
        user_path=user_path,
        memory_limit=memory_limit or cfg.hermes.memory_char_limit,
        user_limit=user_limit or cfg.hermes.user_char_limit,
        pinned_sections=cfg.hermes.pinned_sections,
        compact=cfg.hermes.compact_profile if compact is None else compact,
    )
    return bundle, write_bundle(bundle, cfg.exports_dir)
