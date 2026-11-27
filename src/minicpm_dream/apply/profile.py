"""Profile planning: pinned slots, eviction and compaction for the live profile.

The live profile (`MEMORY.md` / `USER.md`) is a small fixed-size view of the
ledger, not the ledger itself. When it fills up, dropping the incoming facts is
the worst option: the profile slowly freezes around whatever was learned first.
This module decides what deserves a slot, keeps identity/preferences pinned, and
evicts the lowest-salience entry instead of refusing the new one. Eviction only
removes text this tool wrote; the ledger row and its history stay put, and every
eviction is reported in `DREAMS.md`.

Planning is bounded: only the highest-salience candidates are considered, and
token sets are computed once, so a 3,000-memory ledger does not turn profile
render into a quadratic string comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..util.text import normalize_text, token_set

MATCH_THRESHOLD = 0.82
PINNED_PRIORITY = 1_000_000.0
# No profile can hold more than a few hundred entries; scanning the whole ledger
# per candidate is quadratic for no benefit.
MAX_CANDIDATES = 400


@dataclass
class ProfilePlan:
    target: str
    entries: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    evicted: list[dict[str, Any]] = field(default_factory=list)
    pinned: list[str] = field(default_factory=list)
    limit: int = 0
    chars: int = 0
    over_pinned: bool = False

    @property
    def pressure(self) -> dict[str, Any]:
        return {
            "chars": self.chars,
            "limit": self.limit,
            "fill": round(self.chars / max(1, self.limit), 3),
            "skipped": len(self.skipped),
            "evicted": len(self.evicted),
            "over_pinned": self.over_pinned,
        }


def _norm(entry: str) -> str:
    return normalize_text(entry).casefold()


def _overlap(a: set[str], b: set[str]) -> float:
    """Max of Jaccard and containment, the same rule the old matching used."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return max(inter / len(a | b), inter / min(len(a), len(b)))


def _pairs(memories: list[dict[str, Any]]) -> list[tuple[dict[str, Any], set[str]]]:
    return [(memory, token_set(str(memory.get("text", "")))) for memory in memories]


def _best_match(
    entry_tokens: set[str], pairs: list[tuple[dict[str, Any], set[str]]]
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = MATCH_THRESHOLD
    for memory, tokens in pairs:
        score = _overlap(entry_tokens, tokens)
        if score >= best_score:
            best_score = score
            best = memory
    return best


def match_memory(entry: str, memories: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best memory for one live entry, using the same threshold as planning."""
    return _best_match(token_set(entry), _pairs(memories))


def plan_profile(
    *,
    target: str,
    existing: list[str],
    memories: list[dict[str, Any]],
    superseded: list[dict[str, Any]],
    limit: int,
    entry_for: Callable[[dict[str, Any]], str],
    render: Callable[[list[str]], str],
    anchor_for: Callable[[str, list[str]], str],
    pinned_sections: set[str],
    compact: bool = True,
) -> ProfilePlan:
    """Plan the final profile text and the minimal diff to get there."""
    plan = ProfilePlan(target=target, limit=limit)
    usable = [m for m in memories if str(m.get("text", "")).strip()]
    variety = [
        m for m in superseded if str(m.get("text", "")).strip() not in {"", "[forgotten]"}
    ]
    usable_pairs = _pairs(usable)
    variety_pairs = _pairs(variety)

    live_matches = [_best_match(token_set(entry), usable_pairs) for entry in existing]
    stale_removals: list[int] = []
    for index, entry in enumerate(existing):
        if live_matches[index] is None and _best_match(token_set(entry), variety_pairs) is not None:
            stale_removals.append(index)

    def _pinned(memory: dict[str, Any] | None) -> bool:
        return memory is not None and str(memory.get("section")) in pinned_sections

    items: list[tuple[str, float, bool, dict[str, Any] | None]] = []
    seen: set[str] = set()
    for index, entry in enumerate(existing):
        if index in stale_removals:
            continue
        memory = live_matches[index]
        # Human-written entries that never came from the ledger are never evicted,
        # and with compaction off nothing already in the file is touched either.
        pinned = memory is None or _pinned(memory) or not compact
        key = _norm(entry)
        if key in seen:
            continue
        seen.add(key)
        priority = PINNED_PRIORITY if pinned else float((memory or {}).get("salience") or 0.0)
        items.append((entry, priority, pinned, memory))

    ordered = sorted(
        usable,
        key=lambda m: (float(m.get("salience") or 0.0), str(m.get("recorded_at") or "")),
        reverse=True,
    )[:MAX_CANDIDATES]
    seen_texts = [item[0] for item in items]
    seen_tokens = [token_set(text) for text in seen_texts]

    def _covered(entry_tokens: set[str]) -> bool:
        return any(_overlap(entry_tokens, other) >= MATCH_THRESHOLD for other in seen_tokens)

    for memory in ordered:
        entry = entry_for(memory)
        key = _norm(entry)
        if not entry or key in seen:
            continue
        tokens = token_set(entry)
        if _covered(tokens):
            continue
        seen.add(key)
        seen_texts.append(entry)
        seen_tokens.append(tokens)
        pinned = _pinned(memory)
        items.append(
            (
                entry,
                PINNED_PRIORITY if pinned else float(memory.get("salience") or 0.0),
                pinned,
                memory,
            )
        )

    pinned_items = [item for item in items if item[2]]
    others = sorted([item for item in items if not item[2]], key=lambda item: item[1], reverse=True)
    plan.pinned = [item[0] for item in pinned_items]

    final: list[tuple[str, float, bool, dict[str, Any] | None]] = list(pinned_items)

    def _fits(entries: list[tuple[str, float, bool, dict[str, Any] | None]]) -> bool:
        return len(render([entry for entry, _p, _pin, _m in entries])) <= limit

    if not _fits(final):
        plan.over_pinned = True
        plan.skipped.extend(
            {"target": target, "text": item[0], "reason": "profile is full of pinned entries"}
            for item in others
        )
    else:
        for item in others:
            if _fits(final + [item]):
                final.append(item)
                continue
            if not compact:
                plan.skipped.append({"target": target, "text": item[0], "reason": "char budget"})
                continue
            # Evict the lowest-salience entry this tool wrote to make room. Never
            # evict a pinned or human-written entry, and never evict something
            # more salient than the incoming memory.
            working = list(final)
            while not _fits(working + [item]):
                evictable = [entry for entry in working if not entry[2] and entry[1] < item[1]]
                if not evictable:
                    break
                working.remove(min(evictable, key=lambda entry: entry[1]))
            if _fits(working + [item]):
                final = working + [item]
            else:
                plan.skipped.append(
                    {"target": target, "text": item[0], "reason": "char budget; nothing evictable"}
                )

    plan.entries = [entry for entry, _p, _pin, _m in final]
    plan.chars = len(render(plan.entries))
    final_norms_for_report = {_norm(entry) for entry in plan.entries}
    for index, entry in enumerate(existing):
        if _norm(entry) in final_norms_for_report or index in stale_removals:
            continue
        memory = live_matches[index]
        plan.evicted.append(
            {
                "target": target,
                "text": entry,
                "reason": (
                    "profile is full of pinned entries"
                    if plan.over_pinned
                    else "profile budget: dropped for a higher-salience memory; kept in the ledger"
                ),
                "salience": round(float((memory or {}).get("salience") or 0.0), 3),
            }
        )

    # Diff existing -> final, keeping removals ahead of additions.
    current = [_norm(entry) for entry in existing]
    final_norms = {_norm(entry) for entry in plan.entries}

    replacements: dict[str, tuple[str, dict[str, Any]]] = {}
    for entry, _priority, _pinned_flag, memory in final:
        if _norm(entry) in current or memory is None:
            continue
        old_id = memory.get("supersedes_id")
        if not old_id:
            continue
        old = next((row for row in variety if str(row.get("id")) == str(old_id)), None)
        if old is None:
            continue
        old_tokens = token_set(str(old.get("text", "")))
        for live in existing:
            if _overlap(token_set(live), old_tokens) >= MATCH_THRESHOLD:
                replacements[entry] = (live, memory)
                break

    for index, entry in enumerate(existing):
        if _norm(entry) not in final_norms and entry not in {
            old for old, _m in replacements.values()
        }:
            plan.actions.append(
                {
                    "target": target,
                    "action": "remove",
                    "old_text": anchor_for(entry, existing),
                    "reason": "superseded" if index in stale_removals else "compacted",
                }
            )

    for entry, _priority, _pinned_flag, memory in final:
        if _norm(entry) in current:
            continue
        if entry in replacements:
            live, memory = replacements[entry]
            plan.actions.append(
                {
                    "target": target,
                    "action": "replace",
                    "old_text": anchor_for(live, existing),
                    "content": entry,
                    "memory_id": (memory or {}).get("id"),
                    "reason": "supersede",
                }
            )
            continue
        plan.actions.append(
            {
                "target": target,
                "action": "add",
                "content": entry,
                "memory_id": (memory or {}).get("id"),
                "reason": "new",
            }
        )
    _cap(plan.skipped, target, "candidates did not fit")
    _cap(plan.evicted, target, "entries were evicted")
    return plan


def _cap(items: list[dict[str, Any]], target: str, label: str, keep: int = 50) -> None:
    """Keep the review file readable when a large ledger overflows the profile."""
    if len(items) <= keep:
        return
    dropped = len(items) - keep
    del items[keep:]
    items.append(
        {
            "target": target,
            "text": "",
            "reason": f"{dropped} more {label}",
            "count": dropped,
        }
    )
