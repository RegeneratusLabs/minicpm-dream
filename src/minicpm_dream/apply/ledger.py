"""Apply reconciliation decisions to the bitemporal ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..config import ConsolidateCfg
from ..consolidate.reconcile import Decision
from ..ingest.redact import has_secret
from ..store.db import Store
from ..util.ids import new_id
from ..util.logging import get_logger
from ..util.timeutil import iso_utc, utc_now
from .guards import check_loss, check_memory_text, snapshot

log = get_logger("apply.ledger")


class LossGuardError(RuntimeError):
    """Raised to roll back a run that would destroy too much active memory."""

    def __init__(self, loss: Any):
        self.loss = loss
        super().__init__("; ".join(getattr(loss, "violations", ["memory loss guard tripped"])))

_KIND_WEIGHT = {
    "preference": 0.95,
    "identity": 0.95,
    "decision": 0.85,
    "task": 0.75,
    "self": 0.75,
    "fact": 0.65,
    "event": 0.55,
    "question": 0.3,
}
_SECTION_BONUS = {"identity": 0.15, "preferences": 0.12, "projects": 0.08, "people": 0.05}


def salience(decision: Decision) -> float:
    cand = decision.candidate
    evidence = min(1.0, len(cand.evidence) / 2.0)
    weight = _KIND_WEIGHT.get(cand.kind, 0.5) + _SECTION_BONUS.get(cand.section, 0.0)
    return round(min(1.0, 0.5 * cand.confidence + 0.2 * evidence + 0.3 * min(1.0, weight)), 3)


def _candidate_payload(decision: Decision) -> dict[str, Any]:
    cand = decision.candidate
    return {
        "candidate_id": cand.id,
        "kind": cand.kind,
        "section": cand.section,
        "text": cand.text,
        "subject": cand.subject,
        "entities": cand.entities,
        "valid_from": cand.valid_from,
        "valid_to": cand.valid_to,
        "confidence": cand.confidence,
        "evidence": [ev.__dict__ for ev in cand.evidence],
        "decision": {
            "op": decision.op,
            "target_ids": decision.target_ids,
            "merged_text": decision.merged_text,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "errors": decision.errors,
            "rationale": decision.rationale,
        },
    }


@dataclass
class ApplyResult:
    applied: list[dict[str, Any]] = field(default_factory=list)
    deferred: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    loss: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "deferred": self.deferred,
            "proposals": self.proposals,
            "memory_ids": self.memory_ids,
            "counts": self.counts,
            "errors": self.errors,
        }


def _record_op(
    store: Store,
    *,
    run_id: str,
    op: str,
    targets: list[str],
    payload: dict[str, Any],
    decision: Decision,
    applied: bool,
) -> str:
    op_id = new_id("op")
    store.conn.execute(
        """INSERT INTO ops (id, run_id, ts, op, target_ids_json, payload_json, decision,
                            rationale, validator_result, applied)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            op_id,
            run_id,
            iso_utc(utc_now()),
            op,
            json.dumps(targets),
            json.dumps(payload),
            decision.reason,
            decision.rationale,
            json.dumps(decision.errors),
            int(applied),
        ),
    )
    return op_id


def apply_decisions(
    store: Store,
    decisions: list[Decision],
    *,
    run_id: str,
    model: str = "",
    cfg: ConsolidateCfg | None = None,
) -> ApplyResult:
    cfg = cfg or ConsolidateCfg()
    result = ApplyResult()
    before = snapshot(store)
    with store.transaction():
        for decision in decisions:
            cand = decision.candidate
            problems = check_memory_text(decision.final_text)
            if problems:
                decision.errors.extend(problems)
                decision.requires_review = True
            if decision.errors:
                pid = store.add_proposal(
                    run_id=run_id,
                    kind="review",
                    target="ledger",
                    payload=_candidate_payload(decision),
                    cooldown_days=3,
                )
                result.proposals.append(pid)
                result.deferred.append(_candidate_payload(decision))
                _record_op(
                    store, run_id=run_id, op=decision.op, targets=decision.target_ids,
                    payload=_candidate_payload(decision), decision=decision, applied=False,
                )
                continue

            if decision.op == "NOOP":
                _record_op(
                    store, run_id=run_id, op="NOOP", targets=decision.target_ids,
                    payload=_candidate_payload(decision), decision=decision, applied=False,
                )
                result.applied.append({"op": "NOOP", "candidate": cand.text})
                continue
            if decision.op == "DEFER":
                pid = store.add_proposal(
                    run_id=run_id,
                    kind="review",
                    target="ledger",
                    payload=_candidate_payload(decision),
                    cooldown_days=3,
                )
                result.proposals.append(pid)
                result.deferred.append(_candidate_payload(decision))
                _record_op(
                    store, run_id=run_id, op="DEFER", targets=decision.target_ids,
                    payload=_candidate_payload(decision), decision=decision, applied=False,
                )
                continue

            if decision.op in {"UPDATE", "SUPERSEDE", "MERGE"}:
                for target in decision.target_ids:
                    store.supersede(target, valid_to=cand.valid_from or iso_utc(utc_now())[:10])

            mid = store.insert_memory(
                kind=cand.kind,
                section=cand.section,
                text=decision.final_text,
                evidence=[ev.__dict__ for ev in cand.evidence],
                subject=cand.subject,
                entities=cand.entities,
                valid_from=cand.valid_from,
                valid_to=cand.valid_to,
                confidence=decision.confidence,
                salience=salience(decision),
                origin="said",
                created_by="slm",
                model=model,
                prompt_version=decision.prompt_version,
                supersedes_id=decision.target_ids[0] if decision.target_ids else None,
                run_id=run_id,
            )
            result.memory_ids.append(mid)
            result.applied.append(
                {"op": decision.op, "memory_id": mid, "text": decision.final_text,
                 "targets": decision.target_ids}
            )
            _record_op(
                store, run_id=run_id, op=decision.op, targets=decision.target_ids,
                payload=_candidate_payload(decision), decision=decision, applied=True,
            )
        after_counts = store.counts()
        after = {
            "entries": int(after_counts.get("active", 0)),
            "chars": int(after_counts.get("active_chars", after_counts.get("chars", 0))),
        }
        loss = check_loss(before, after, threshold=0.25)
        result.loss = loss.as_dict()
        if not loss.ok:
            raise LossGuardError(loss)
    result.counts = store.counts()
    return result


def serialize_decisions(decisions: list[Decision]) -> list[dict[str, Any]]:
    out = []
    for d in decisions:
        payload = _candidate_payload(d)
        if has_secret(json.dumps(payload)):
            payload["candidate"]["evidence"] = []
        out.append(payload)
    return out
