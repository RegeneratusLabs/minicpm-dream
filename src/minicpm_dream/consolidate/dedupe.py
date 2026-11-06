"""Deterministic duplicate and conflict pre-checks."""

from __future__ import annotations

from typing import Any

from ..extract.validate import Candidate
from ..util.text import containment, digits, jaccard, norm_key


def exact_duplicate(candidate: Candidate, existing: dict[str, Any]) -> bool:
    return norm_key(candidate.text) == norm_key(str(existing.get("text", "")))


def near_duplicate(candidate: Candidate, existing: dict[str, Any], threshold: float = 0.9) -> bool:
    return jaccard(candidate.text, str(existing.get("text", ""))) >= threshold


def same_fact_different_numbers(candidate: Candidate, existing: dict[str, Any]) -> bool:
    old = str(existing.get("text", ""))
    if jaccard(candidate.text, old) < 0.45:
        return False
    new_digits, old_digits = digits(candidate.text), digits(old)
    return bool(new_digits) and bool(old_digits) and new_digits != old_digits


def duplicate_score(candidate: Candidate, existing: dict[str, Any]) -> float:
    old = str(existing.get("text", ""))
    return max(jaccard(candidate.text, old), containment(candidate.text, old))
