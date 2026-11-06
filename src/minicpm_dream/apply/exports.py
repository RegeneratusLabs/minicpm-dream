"""Build Hermes-format candidate memory files and a review manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..store.db import Store
from ..util.atomic import write_json_atomic, write_text_atomic
from ..util.text import containment, jaccard, normalize_text
from ..util.timeutil import iso_utc, utc_now

USER_SECTIONS = {"identity", "preferences"}
MEMORY_SECTIONS = {"environment", "projects", "procedures", "lessons", "people"}
DIARY_ONLY = {"open_loops"}
SEP = "\n§\n"


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
    limits: dict[str, int] = field(default_factory=dict)
    generated_at: str = ""

    def as_review(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "limits": self.limits,
            "memory": {"entries": self.memory_entries, "content": self.memory_text},
            "user": {"entries": self.user_entries, "content": self.user_text},
            "actions": self.actions,
            "skipped": self.skipped,
        }


def _fit(entries: list[str], limit: int) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    skipped: list[str] = []
    for entry in entries:
        candidate = render_entries(kept + [entry])
        if len(candidate) <= limit:
            kept.append(entry)
        else:
            skipped.append(entry)
    return kept, skipped


def _matches_existing(text: str, existing: list[str]) -> str | None:
    best: tuple[float, str] | None = None
    for entry in existing:
        score = max(jaccard(text, entry), containment(text, entry))
        if score >= 0.82 and (best is None or score > best[0]):
            best = (score, entry)
    return best[1] if best else None


def build_bundle(
    store: Store,
    *,
    memory_path: str | Path,
    user_path: str | Path,
    memory_limit: int,
    user_limit: int,
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

    by_target: dict[str, list[dict[str, Any]]] = {"memory": [], "user": []}
    for memory in active:
        target = target_for_section(str(memory.get("section")))
        if target:
            by_target[target].append(memory)

    for target in by_target:
        by_target[target].sort(
            key=lambda m: (float(m.get("salience") or 0), str(m.get("recorded_at") or "")), reverse=True
        )

    bundle = ExportBundle(
        limits={"memory": memory_limit, "user": user_limit},
        generated_at=iso_utc(utc_now()),
    )
    for target, limit in (
        ("memory", memory_limit),
        ("user", user_limit),
    ):
        entries = [_entry_for_memory(m) for m in by_target[target]]
        entries = list(dict.fromkeys(entries))
        kept, skipped = _fit(entries, limit)
        if target == "memory":
            bundle.memory_entries = kept
            bundle.memory_text = render_entries(kept)
        else:
            bundle.user_entries = kept
            bundle.user_text = render_entries(kept)
        for text in skipped:
            bundle.skipped.append({"target": target, "text": text, "reason": "char budget"})

        existing = existing_memory if target == "memory" else existing_user
        handled: set[str] = set()
        for memory in by_target[target]:
            text = _entry_for_memory(memory)
            if _matches_existing(text, existing):
                continue
            old_id = memory.get("supersedes_id")
            if old_id:
                old = store.get_memory(str(old_id))
                old_text = normalize_text(str(old.get("text", ""))) if old else ""
                match = _matches_existing(old_text, existing) if old_text else None
                if match and match not in handled:
                    handled.add(match)
                    bundle.actions.append(
                        {
                            "target": target,
                            "action": "replace",
                            "old_text": unique_anchor(match, existing),
                            "content": text,
                            "memory_id": memory["id"],
                            "reason": "supersede",
                        }
                    )
                    continue
            bundle.actions.append(
                {
                    "target": target,
                    "action": "add",
                    "content": text,
                    "memory_id": memory["id"],
                    "reason": "new",
                }
            )

        for memory in superseded:
            if target_for_section(str(memory.get("section"))) != target:
                continue
            text = normalize_text(str(memory.get("text", "")))
            if not text or text == "[forgotten]":
                continue
            match = _matches_existing(text, existing)
            if match and match not in handled:
                handled.add(match)
                bundle.actions.append(
                    {
                        "target": target,
                        "action": "remove",
                        "old_text": unique_anchor(match, existing),
                        "memory_id": memory["id"],
                        "reason": str(memory.get("status")),
                    }
                )
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
) -> tuple[ExportBundle, dict[str, str]]:
    memory_path = Path(cfg.hermes.home) / cfg.hermes.memory_file
    user_path = Path(cfg.hermes.home) / cfg.hermes.user_file
    bundle = build_bundle(
        store,
        memory_path=memory_path,
        user_path=user_path,
        memory_limit=memory_limit or cfg.hermes.memory_char_limit,
        user_limit=user_limit or cfg.hermes.user_char_limit,
    )
    return bundle, write_bundle(bundle, cfg.exports_dir)
