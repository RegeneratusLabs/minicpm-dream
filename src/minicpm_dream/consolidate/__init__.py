"""Candidate retrieval and reconciliation."""

from .dedupe import duplicate_score, exact_duplicate, near_duplicate, same_fact_different_numbers
from .policy import apply_budget, needs_review
from .reconcile import (
    Decision,
    build_messages,
    reconcile_all,
    reconcile_candidate,
    validate_decision,
)
from .retrieve import Retriever

__all__ = [
    "Decision",
    "Retriever",
    "apply_budget",
    "build_messages",
    "duplicate_score",
    "exact_duplicate",
    "near_duplicate",
    "needs_review",
    "reconcile_all",
    "reconcile_candidate",
    "same_fact_different_numbers",
    "validate_decision",
]
