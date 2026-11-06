"""Evidence entailment check: does the cited quote actually support the claim?"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..backends.base import Backend, BackendError, extract_json
from ..config import ExtractCfg
from ..prepare.chunk import Chunk
from ..util.text import digits, token_set, truncate
from .schema import load_schema, render_prompt
from .validate import Candidate


@dataclass
class VerifyResult:
    candidate: Candidate
    supported: bool
    verdict: str = ""
    confidence: float = 0.0
    error: str = ""


def _context(episode_text: str, quote: str) -> str:
    text = episode_text
    idx = text.casefold().find(quote.casefold())
    if idx < 0:
        return truncate(text, 1200)
    start = max(0, idx - 400)
    end = min(len(text), idx + len(quote) + 400)
    return text[start:end]


def verify_candidate(
    backend: Backend,
    candidate: Candidate,
    chunk: Chunk,
    cfg: ExtractCfg,
) -> VerifyResult:
    if not candidate.evidence:
        return VerifyResult(candidate, False, error="no evidence")
    if not cfg.verify:
        return VerifyResult(candidate, True, verdict="skipped", confidence=1.0)
    episodes = {ep.id: ep for ep in chunk.episodes}
    ev = candidate.evidence[0]
    episode = episodes.get(ev.episode_id)
    if episode is None:
        return VerifyResult(candidate, False, error="evidence episode missing")
    claim_tokens = token_set(candidate.text)
    quote_tokens = token_set(ev.quote)
    source = f"{ev.quote} {episode.text} {candidate.valid_from or ''}"
    overlap = len(claim_tokens & quote_tokens) / len(claim_tokens) if claim_tokens else 0.0
    novel_digits = digits(candidate.text) - digits(source)
    claim_entities = {
        match.group(0)
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", candidate.text)
        if match.group(0).casefold() not in {"the", "this", "that", "user", "josh"}
    }
    novel_entities = {
        entity for entity in claim_entities if entity.casefold() not in source.casefold()
    }
    claim_negated = bool(re.search(r"(?i)\b(not|no longer|never|stopped)\b", candidate.text))
    quote_negated = bool(re.search(r"(?i)\b(not|no longer|never|stopped)\b", ev.quote))
    if (
        overlap >= 0.4
        and not novel_digits
        and not novel_entities
        and (claim_negated or not quote_negated)
    ):
        ev.verified = True
        return VerifyResult(candidate, True, verdict="quote-overlap", confidence=0.9)
    system = render_prompt("verify.j2")
    user = (
        f"CLAIM:\n{candidate.text}\n\n"
        f"QUOTE:\n{ev.quote}\n\n"
        f"CONTEXT:\n{_context(episode.text, ev.quote)}\n\n"
        "Return only the JSON verdict."
    )
    try:
        completion = backend.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema=load_schema("verify.json"),
            max_tokens=cfg.verify_max_tokens,
            temperature=0.0,
        )
        payload = extract_json(completion.text)
    except BackendError as exc:
        return VerifyResult(candidate, False, error=str(exc))
    verdict = str(payload.get("verdict", "")).lower()
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    supported = verdict == "supports" and confidence >= 0.5
    if not supported and cfg.verify_lenient and verdict == "insufficient" and not novel_digits:
        # A paraphrase is not a hallucination. Keep the quote-grounded claim, but
        # record that it did not get a clean entailment verdict.
        supported = True
        confidence = min(confidence, 0.5)
        verdict = "insufficient-lenient"
    if supported:
        ev.verified = verdict == "supports"
    return VerifyResult(candidate, supported, verdict=verdict, confidence=confidence)


def verify_all(
    backend: Backend,
    candidates: list[Candidate],
    chunks_by_index: dict[int, Chunk],
    cfg: ExtractCfg,
    *,
    max_calls: int = 60,
) -> tuple[list[Candidate], list[VerifyResult], dict[str, Any]]:
    accepted: list[Candidate] = []
    results: list[VerifyResult] = []
    usage = {"calls": 0, "rejected": 0}
    for cand in candidates:
        if usage["calls"] >= max_calls:
            results.append(VerifyResult(cand, False, error="verify budget exhausted"))
            usage["rejected"] += 1
            continue
        chunk = chunks_by_index.get(cand.chunk_index)
        if chunk is None and cand.evidence:
            episode_id = cand.evidence[0].episode_id
            chunk = next(
                (
                    candidate_chunk
                    for candidate_chunk in chunks_by_index.values()
                    if any(ep.id == episode_id for ep in candidate_chunk.episodes)
                ),
                None,
            )
        if chunk is None:
            results.append(VerifyResult(cand, False, error="chunk missing"))
            usage["rejected"] += 1
            continue
        result = verify_candidate(backend, cand, chunk, cfg)
        results.append(result)
        usage["calls"] += 1
        if result.supported:
            accepted.append(cand)
        else:
            usage["rejected"] += 1
    return accepted, results, usage
