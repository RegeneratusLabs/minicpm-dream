"""Budget enforcement and review routing for consolidation decisions."""

from __future__ import annotations

from ..config import ConsolidateCfg
from .reconcile import Decision


def apply_budget(decisions: list[Decision], cfg: ConsolidateCfg) -> tuple[list[Decision], list[Decision]]:
    """Return (applied, deferred). Highest-confidence changes win the nightly budget."""
    ordered = sorted(decisions, key=lambda d: d.confidence, reverse=True)
    changes = 0
    new = 0
    applied: list[Decision] = []
    deferred: list[Decision] = []
    for decision in ordered:
        if decision.op in {"NOOP", "DEFER"}:
            applied.append(decision)
            continue
        is_new = decision.op == "ADD"
        if changes >= cfg.max_changes_per_run or (is_new and new >= cfg.max_new_per_run):
            decision.op = "DEFER"
            decision.reason = "uncertain"
            decision.rationale = "nightly change budget reached; deferred to next run"
            decision.requires_review = False
            deferred.append(decision)
            continue
        changes += 1
        if is_new:
            new += 1
        applied.append(decision)
    return applied, deferred


def needs_review(decisions: list[Decision]) -> list[Decision]:
    return [d for d in decisions if d.requires_review or d.op == "DEFER"]
