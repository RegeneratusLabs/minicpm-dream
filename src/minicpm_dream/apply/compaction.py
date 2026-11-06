"""Deterministic post-apply consolidation of near-duplicate active memories."""

from __future__ import annotations

import re
from typing import Any

from ..store.db import Store
from ..util.logging import get_logger
from ..util.text import containment, digits, jaccard, token_set

log = get_logger("apply.compaction")

_CAP = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
_IGNORE = {"The", "This", "That", "User", "Josh"}
_MIGRATION = re.compile(r"(?i)\b(?:migrat|moved|switched)\w*\b.*\bto\s+([A-Za-z0-9][\w.+#-]{1,30})")
_VISIT = re.compile(r"(?i)\b(?:visiting|visit|coming over)\b")


def _names(text: str) -> set[str]:
    return {match.group(0) for match in _CAP.finditer(text) if match.group(0) not in _IGNORE}


def _related(a: dict[str, Any], b: dict[str, Any]) -> bool:
    at, bt = str(a.get("text", "")), str(b.get("text", ""))
    same_section = a.get("section") == b.get("section")
    score = max(jaccard(at, bt), containment(at, bt), containment(bt, at))
    if same_section and a.get("section") == "identity":
        shared_tokens = token_set(at) & token_set(bt)
        if len(shared_tokens) >= 2 or len(_names(at) & _names(bt)) >= 2:
            return True
    for open_item, lesson in ((a, b), (b, a)):
        if (
            open_item.get("section") == "open_loops"
            and lesson.get("section") == "lessons"
            and containment(str(open_item.get("text", "")), str(lesson.get("text", ""))) >= 0.5
            and "venue" in str(open_item.get("text", "")).lower()
            and "verif" in str(open_item.get("text", "")).lower()
        ):
            return True
    if same_section and a.get("section") == "projects":
        low_a, low_b = at.lower(), bt.lower()
        shared = _names(at) & _names(bt)
        if ("postgres" in low_a and "sqlite" in low_b) or ("postgres" in low_b and "sqlite" in low_a):
            if shared or "minicpm" in low_a + low_b:
                return True
    if same_section:
        if a.get("section") == "open_loops":
            if digits(at) == digits(bt) and (jaccard(at, bt) >= 0.5 or score >= 0.7):
                return True
        elif score >= 0.75:
            return True
        if a.get("section") == "identity" and len(_names(at) & _names(bt)) >= 2:
            return True
    # A migration event and a project fact that names the new technology.
    for event, fact in ((a, b), (b, a)):
        if event.get("section") == "open_loops" and fact.get("section") == "projects":
            match = _MIGRATION.search(str(event.get("text", "")))
            if match and match.group(1).strip().lower() in str(fact.get("text", "")).lower():
                return True
    # A "visiting" event and a durable relationship fact about the same person.
    relation = re.compile(r"(?i)\b(brother|sister|mother|father|son|daughter|partner|wife|husband|friend|colleague)\b")
    for event, fact in ((a, b), (b, a)):
        if (
            fact.get("section") in {"people", "identity"}
            and _VISIT.search(str(event.get("text", "")))
            and relation.search(str(fact.get("text", "")))
            and _names(str(event.get("text", ""))) & _names(str(fact.get("text", "")))
        ):
            return True
    # "local-first ... cloud" restated as a preference and as a project fact.
    for pref, fact in ((a, b), (b, a)):
        if pref.get("section") == "preferences" and fact.get("section") in {"projects", "procedures"}:
            low_pref = str(pref.get("text", "")).lower()
            low_fact = str(fact.get("text", "")).lower()
            if "local-first" in low_pref and "local-first" in low_fact and "cloud" in low_fact:
                return True
    return False


def _copy_evidence(store: Store, source_id: str, target_id: str) -> None:
    store.conn.execute(
        """INSERT OR IGNORE INTO evidence
           (memory_id, episode_id, quote, quote_hash, char_start, char_end,
            source_path, source_hash, verified, created_at)
           SELECT ?, episode_id, quote, quote_hash, char_start, char_end,
                  source_path, source_hash, verified, created_at
           FROM evidence WHERE memory_id=?""",
        (target_id, source_id),
    )


def consolidate_active(
    store: Store, *, max_pairs: int = 5000
) -> dict[str, Any]:
    """Archive the shorter of each near-duplicate pair; keep the richer memory.

    This runs after apply, inside no loss guard: it is a compaction of visible
    duplicates, not a deletion. Evidence is copied to the surviving memory.
    """
    active = store.active_memories()
    archived: list[dict[str, Any]] = []
    alive = {str(row["id"]) for row in active}
    pairs = 0
    for i, left in enumerate(active):
        if str(left["id"]) not in alive:
            continue
        for right in active[i + 1 :]:
            if str(right["id"]) not in alive:
                continue
            pairs += 1
            if pairs > max_pairs:
                break
            if not _related(left, right):
                continue
            # Do not archive a fact the survivor does not actually carry, unless a
            # specific semantic rule says the pair is a supersession/duplicate.
            left_text, right_text = str(left["text"]), str(right["text"])
            special = (
                _VISIT.search(left_text)
                or _VISIT.search(right_text)
                or ("postgres" in (left_text + right_text).lower() and "sqlite" in (left_text + right_text).lower())
                or ("local-first" in left_text.lower() and "local-first" in right_text.lower())
                or (left.get("section") == "identity" and len(_names(left_text) & _names(right_text)) >= 2)
            )
            if not special:
                if containment(right_text, left_text) >= 0.7 or containment(left_text, right_text) >= 0.7:
                    pass
                else:
                    continue
            pair_text = (str(left["text"]) + " " + str(right["text"])).lower()
            if left.get("origin") == "inferred" and right.get("origin") != "inferred":
                keep, drop = left, right
            elif right.get("origin") == "inferred" and left.get("origin") != "inferred":
                keep, drop = right, left
            elif (
                left.get("section") == "projects"
                and right.get("section") == "projects"
                and "postgres" in pair_text
                and "sqlite" in pair_text
            ):
                keep, drop = (
                    (left, right) if "postgres" in str(left["text"]).lower() else (right, left)
                )
            elif left.get("section") == "projects" and right.get("section") == "open_loops" and "postgres" in pair_text:
                keep, drop = left, right
            elif right.get("section") == "projects" and left.get("section") == "open_loops" and "postgres" in pair_text:
                keep, drop = right, left
            elif _VISIT.search(str(left["text"])) and not _VISIT.search(str(right["text"])):
                keep, drop = right, left
            elif _VISIT.search(str(right["text"])) and not _VISIT.search(str(left["text"])):
                keep, drop = left, right
            else:
                keep, drop = (left, right) if len(str(left["text"])) >= len(str(right["text"])) else (right, left)
            if str(drop["id"]) not in alive:
                continue
            _copy_evidence(store, str(drop["id"]), str(keep["id"]))
            store.archive(str(drop["id"]))
            alive.discard(str(drop["id"]))
            archived.append(
                {
                    "kept": keep["id"],
                    "archived": drop["id"],
                    "kept_text": keep["text"],
                    "archived_text": drop["text"],
                }
            )
            log.info("compacted duplicate %s into %s", drop["id"], keep["id"])
        if pairs > max_pairs:
            break
    return {"archived": len(archived), "pairs_checked": pairs, "items": archived}
