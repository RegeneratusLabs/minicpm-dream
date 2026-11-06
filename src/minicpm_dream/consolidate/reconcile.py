"""Reconcile one candidate against retrieved memory with a narrow typed operation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import Backend, BackendError, extract_json
from ..config import ConsolidateCfg
from ..extract.schema import load_schema, prompt_version, render_prompt
from ..extract.validate import Candidate
from ..util.text import digits, norm_key
from .dedupe import exact_duplicate, near_duplicate, same_fact_different_numbers

PROMPT = "reconcile.j2"
OPS = {"ADD", "UPDATE", "SUPERSEDE", "MERGE", "NOOP", "DEFER"}
REASONS = {"new", "changed", "corrected", "no_longer_true", "duplicate", "implied", "uncertain"}
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@dataclass
class Decision:
    candidate: Candidate
    op: str
    target_ids: list[str] = field(default_factory=list)
    merged_text: str = ""
    reason: str = "new"
    confidence: float = 0.0
    rationale: str = ""
    errors: list[str] = field(default_factory=list)
    requires_review: bool = False
    prompt_version: str = prompt_version(PROMPT)

    @property
    def final_text(self) -> str:
        return (self.merged_text or self.candidate.text).strip()


def _format_existing(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(none)"
    lines = []
    for row in rows:
        lines.append(
            f"[id={row['id']}] section={row.get('section')} kind={row.get('kind')} "
            f"recorded={row.get('recorded_at')} ({row.get('age', 'unknown')})\n"
            f"{row.get('text', '')}"
        )
    return "\n\n".join(lines)


def build_messages(candidate: Candidate, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = render_prompt(PROMPT)
    user = (
        "NEW:\n"
        f"kind={candidate.kind} section={candidate.section} "
        f"subject={candidate.subject or '(none)'} date={candidate.valid_from or '(none)'}\n"
        f"{candidate.text}\n\n"
        "EXISTING:\n"
        f"{_format_existing(rows)}\n\n"
        "Return only the JSON operation."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _precheck(candidate: Candidate, rows: list[dict[str, Any]], cfg: ConsolidateCfg) -> Decision | None:
    for row in rows:
        if exact_duplicate(candidate, row):
            return Decision(
                candidate=candidate,
                op="NOOP",
                target_ids=[row["id"]],
                reason="duplicate",
                confidence=0.99,
                rationale="exact normalized text already stored",
            )
    for row in rows:
        if same_fact_different_numbers(candidate, row):
            return None  # let the model decide SUPERSEDE vs UPDATE
        if near_duplicate(candidate, row, cfg.similarity_threshold):
            if digits(candidate.text) == digits(str(row.get("text", ""))) and norm_key(
                candidate.text
            ) == norm_key(str(row.get("text", ""))):
                return Decision(candidate, "NOOP", [row["id"]], "", "duplicate", 0.95,
                                "near-identical duplicate")
    return None


def validate_decision(
    decision: Decision, rows: list[dict[str, Any]], candidate: Candidate, cfg: ConsolidateCfg
) -> Decision:
    ids = {row["id"] for row in rows}
    raw_targets = list(decision.target_ids)
    if any(tid not in ids for tid in raw_targets):
        decision.errors.append("invented target id")
    decision.target_ids = [tid for tid in raw_targets if tid in ids]
    if decision.op not in OPS:
        decision.errors.append(f"invalid op {decision.op!r}")
    if decision.op in {"UPDATE", "SUPERSEDE", "MERGE"} and not decision.target_ids:
        decision.errors.append("operation requires a target")
    if decision.op in {"ADD", "NOOP"} and decision.target_ids:
        # NOOP legitimately names the duplicate; ADD must not name anything.
        if decision.op == "ADD":
            decision.errors.append("ADD must not carry target ids")
    if decision.op == "MERGE" and len(decision.target_ids) < 2:
        decision.errors.append("MERGE requires at least two targets")
    if decision.reason not in REASONS:
        decision.errors.append(f"invalid reason {decision.reason!r}")
    if decision.op == "SUPERSEDE" and decision.reason == "new":
        decision.errors.append("SUPERSEDE cannot use reason 'new'")

    text = decision.final_text
    if len(text) < 3 or len(text) > 300:
        decision.errors.append("final text length out of range")
    sources = [candidate.text] + [str(row.get("text", "")) for row in rows if row["id"] in decision.target_ids]
    for ev in candidate.evidence:
        sources.append(ev.quote)
    allowed = digits(" ".join(sources)) | set(_DATE.findall(" ".join(sources)))
    novel = digits(text) - allowed
    if novel:
        decision.errors.append(f"merged text introduces unsupported numbers: {sorted(novel)}")
    for date_value in _DATE.findall(text):
        if date_value not in " ".join(sources):
            decision.errors.append(f"merged text introduces unsupported date {date_value}")

    if decision.confidence < cfg.min_confidence:
        decision.requires_review = True
    if decision.errors:
        decision.requires_review = True
    return decision


def reconcile_candidate(
    backend: Backend,
    candidate: Candidate,
    rows: list[dict[str, Any]],
    cfg: ConsolidateCfg,
) -> Decision:
    pre = _precheck(candidate, rows, cfg)
    if pre is not None:
        return pre
    if not rows:
        return Decision(
            candidate=candidate,
            op="ADD",
            reason="new",
            confidence=0.9,
            rationale="no related existing memory",
        )
    messages = build_messages(candidate, rows)
    try:
        completion = backend.complete(
            messages,
            schema=load_schema("reconcile.json"),
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        payload = extract_json(completion.text)
    except BackendError as exc:
        return Decision(
            candidate=candidate,
            op="DEFER",
            reason="uncertain",
            confidence=0.0,
            rationale=f"reconcile failed: {exc}",
            errors=[str(exc)],
            requires_review=True,
        )
    op = str(payload.get("op", "DEFER")).upper()
    raw_targets = payload.get("target_ids") or []
    target_ids = [str(t) for t in raw_targets] if isinstance(raw_targets, list) else []
    merged = payload.get("merged_text") or ""
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    decision = Decision(
        candidate=candidate,
        op=op,
        target_ids=target_ids[:3],
        merged_text=str(merged).strip()[:300],
        reason=str(payload.get("reason", "uncertain")).lower(),
        confidence=min(1.0, max(0.0, confidence)),
        rationale=f"slm op={op}",
    )
    if decision.op == "UPDATE" and not decision.merged_text:
        decision.merged_text = candidate.text
    return validate_decision(decision, rows, candidate, cfg)


def reconcile_all(
    backend: Backend,
    candidates: list[Candidate],
    retriever: Any,
    cfg: ConsolidateCfg,
    *,
    max_calls: int | None = None,
) -> tuple[list[Decision], dict[str, Any]]:
    decisions: list[Decision] = []
    usage = {"calls": 0, "llm_calls": 0, "rejected": 0}
    budget = cfg.max_calls if max_calls is None else min(max_calls, cfg.max_calls)
    for cand in candidates:
        rows = retriever.retrieve(cand)
        decision = reconcile_candidate(backend, cand, rows, cfg)
        # Precheck/no-retrieval decisions make no model call; count only LLM calls.
        if decision.prompt_version and decision.rationale.startswith("slm"):
            usage["llm_calls"] += 1
        usage["calls"] += 1
        if decision.errors:
            usage["rejected"] += 1
        decisions.append(decision)
        if usage["llm_calls"] >= budget:
            break
    return decisions, usage
