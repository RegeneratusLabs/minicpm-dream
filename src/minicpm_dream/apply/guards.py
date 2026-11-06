"""Pre-write safety checks: secrets, budgets and the memory-loss guard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ingest.redact import has_secret
from ..util.text import is_probably_meta


@dataclass
class LossCheck:
    ok: bool
    before_entries: int
    after_entries: int
    before_chars: int
    after_chars: int
    violations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "before_entries": self.before_entries,
            "after_entries": self.after_entries,
            "before_chars": self.before_chars,
            "after_chars": self.after_chars,
            "violations": self.violations,
        }


def snapshot(store: Any) -> dict[str, int]:
    counts = store.counts()
    return {
        "entries": int(counts.get("active", 0)),
        "chars": int(counts.get("active_chars", counts.get("chars", 0))),
    }


def check_loss(before: dict[str, int], after: dict[str, int], *, threshold: float = 0.25) -> LossCheck:
    violations: list[str] = []
    before_entries = max(before["entries"], 0)
    before_chars = max(before["chars"], 0)
    if before_entries >= 4:
        lost = (before_entries - after["entries"]) / before_entries
        if lost > threshold:
            violations.append(f"active entries dropped {lost:.0%} (limit {threshold:.0%})")
    if before_chars >= 200:
        lost_chars = (before_chars - after["chars"]) / before_chars
        if lost_chars > threshold:
            violations.append(f"active characters dropped {lost_chars:.0%} (limit {threshold:.0%})")
    return LossCheck(
        ok=not violations,
        before_entries=before_entries,
        after_entries=after["entries"],
        before_chars=before_chars,
        after_chars=after["chars"],
        violations=violations,
    )


def check_memory_text(text: str) -> list[str]:
    problems: list[str] = []
    if not text.strip():
        problems.append("empty text")
    if is_probably_meta(text):
        problems.append("conversational meta-state")
    if has_secret(text):
        problems.append("secret-shaped text")
    if len(text) > 300:
        problems.append("text exceeds 300 characters")
    return problems


def check_char_budget(text: str, limit: int) -> bool:
    return len(text) <= limit
