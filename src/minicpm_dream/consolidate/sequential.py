"""Reconcile and apply candidates one at a time so later candidates see earlier ones."""

from __future__ import annotations

from typing import Any

from ..apply.guards import check_loss, snapshot
from ..apply.ledger import ApplyResult, LossGuardError, apply_one
from ..backends.base import Backend
from ..config import ConsolidateCfg
from ..extract.validate import Candidate
from ..store.db import Store
from ..util.logging import get_logger
from .reconcile import Decision, reconcile_candidate

log = get_logger("consolidate.sequential")


def reconcile_and_apply(
    store: Store,
    backend: Backend,
    candidates: list[Candidate],
    retriever: Any,
    cfg: ConsolidateCfg,
    *,
    run_id: str,
    model: str = "",
) -> tuple[list[Decision], ApplyResult, dict[str, Any]]:
    """One transaction, one decision at a time.

    The store's FTS index is updated by triggers inside the transaction, so the
    next candidate retrieves memories added earlier in the same run. This is what
    makes intra-run updates and duplicate suppression possible.
    """
    decisions: list[Decision] = []
    applied = ApplyResult()
    usage = {"calls": 0, "llm_calls": 0, "rejected": 0, "budget_deferred": 0}
    changes = 0
    new = 0
    before = snapshot(store)
    with store.transaction():
        ordered = [item for item in candidates if item.origin != "derived"]
        ordered.sort(key=lambda item: item.confidence, reverse=True)
        ordered.extend(item for item in candidates if item.origin == "derived")
        for candidate in ordered:
            rows = retriever.retrieve(candidate)
            if usage["llm_calls"] >= cfg.max_calls:
                decision = Decision(
                    candidate=candidate,
                    op="DEFER",
                    reason="uncertain",
                    confidence=0.0,
                    rationale="reconcile call budget reached",
                    requires_review=False,
                )
            elif candidate.origin == "derived":
                existing = store.find_active_by_norm(candidate.text)
                if existing is not None:
                    decision = Decision(
                        candidate=candidate,
                        op="NOOP",
                        target_ids=[str(existing["id"])],
                        reason="duplicate",
                        confidence=0.95,
                        rationale="deterministic derived fact already stored",
                    )
                else:
                    decision = Decision(
                        candidate=candidate,
                        op="ADD",
                        reason="new",
                        confidence=candidate.confidence,
                        rationale="deterministic derived fact",
                    )
            else:
                decision = reconcile_candidate(backend, candidate, rows, cfg)
                usage["calls"] += 1
                if decision.rationale.startswith("slm"):
                    usage["llm_calls"] += 1
            if decision.errors:
                usage["rejected"] += 1
            if decision.op in {"ADD", "UPDATE", "SUPERSEDE", "MERGE"}:
                is_new = decision.op == "ADD"
                if changes >= cfg.max_changes_per_run or (is_new and new >= cfg.max_new_per_run):
                    decision.op = "DEFER"
                    decision.reason = "uncertain"
                    decision.rationale = "nightly change budget reached; deferred to next run"
                    decision.requires_review = False
                    usage["budget_deferred"] += 1
                else:
                    changes += 1
                    if is_new:
                        new += 1
            decisions.append(decision)
            apply_one(store, decision, run_id=run_id, model=model, result=applied)
        after_counts = store.counts()
        after = {
            "entries": int(after_counts.get("active", 0)),
            "chars": int(after_counts.get("active_chars", after_counts.get("chars", 0))),
        }
        loss = check_loss(before, after, threshold=0.25)
        applied.loss = loss.as_dict()
        if not loss.ok:
            raise LossGuardError(loss)
    applied.counts = store.counts()
    log.info(
        "sequential reconcile: %s candidates, %s applied, %s deferred, %s LLM calls",
        len(candidates),
        len(applied.memory_ids),
        len(applied.deferred),
        usage["llm_calls"],
    )
    return decisions, applied, usage
