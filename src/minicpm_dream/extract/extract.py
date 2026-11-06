"""Extraction stage: one constrained SLM call per chunk."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import Backend, BackendError, extract_json
from ..config import ExtractCfg
from ..prepare.chunk import Chunk, estimate_tokens, render_episode
from ..prepare.timeanchor import render_refs, resolve_dates
from ..util.text import truncate
from .derive import derive_candidates
from .schema import load_schema, prompt_version, render_prompt
from .validate import Candidate, dedupe_candidates, validate_item

PROMPT = "extract.j2"
COVERAGE_PROMPT = "extract_coverage.j2"
PROFILE_PROMPT = "extract_profile.j2"


@dataclass
class ExtractionResult:
    chunk_index: int
    candidates: list[Candidate] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    parse_error: bool = False
    prompt_version: str = prompt_version(PROMPT)


def build_messages(chunk: Chunk, cfg: ExtractCfg) -> list[dict[str, str]]:
    # Keep the system prompt static across chunks so llama.cpp can reuse its
    # prefix cache; the changing date refs go in the user message.
    system = render_prompt(PROMPT, max_items=cfg.max_items)
    user = (
        "DATE REFS:\n"
        f"{render_refs(chunk.date_refs)}\n\n"
        "EPISODES:\n"
        f"{chunk.text}\n\n"
        "Return only the JSON object now. Use at most "
        f"{cfg.max_items} items."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_coverage_messages(
    chunk: Chunk, cfg: ExtractCfg, existing: list[Candidate]
) -> list[dict[str, str]]:
    system = render_prompt(COVERAGE_PROMPT)
    existing_text = "\n".join(f"- {truncate(c.text, 160)}" for c in existing[:16]) or "(none)"
    user = (
        "DATE REFS:\n"
        f"{render_refs(chunk.date_refs)}\n\n"
        "ALREADY EXTRACTED:\n"
        f"{existing_text}\n\n"
        "EPISODES:\n"
        f"{chunk.text}\n\n"
        f"Return only NEW items, at most {cfg.coverage_max_items}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_profile_messages(chunk: Chunk, cfg: ExtractCfg) -> list[dict[str, str]]:
    system = render_prompt(PROFILE_PROMPT)
    user = (
        "DATE REFS:\n"
        f"{render_refs(chunk.date_refs)}\n\n"
        "EPISODES:\n"
        f"{chunk.text}\n\n"
        f"Return only NEW profile items, at most {cfg.profile_max_items}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def profile_extract(
    backend: Backend, chunk: Chunk, cfg: ExtractCfg
) -> ExtractionResult:
    messages = build_profile_messages(chunk, cfg)
    schema = copy.deepcopy(load_schema("extract.json"))
    schema["properties"]["items"]["maxItems"] = max(1, cfg.profile_max_items)
    completion = backend.complete(
        messages,
        schema=schema,
        max_tokens=cfg.max_tokens,
        temperature=min(0.5, cfg.temperature + 0.1),
    )
    result = ExtractionResult(
        chunk_index=chunk.index,
        raw=completion.text,
        usage=dict(completion.usage),
    )
    try:
        payload = extract_json(completion.text)
    except BackendError as exc:
        result.parse_error = True
        result.rejected.append({"reason": f"profile unparseable: {exc}", "raw": completion.text[:300]})
        return result
    items = payload.get("items")
    if not isinstance(items, list):
        result.parse_error = True
        result.rejected.append({"reason": "profile missing items array", "raw": completion.text[:300]})
        return result
    candidates: list[Candidate] = []
    for item in items[: cfg.profile_max_items]:
        candidate, errors = validate_item(item, chunk)
        if candidate is None:
            result.rejected.append({"reason": "; ".join(errors), "item": item})
        else:
            candidates.append(candidate)
    result.candidates = candidates
    return result


def coverage_extract(
    backend: Backend, chunk: Chunk, existing: list[Candidate], cfg: ExtractCfg
) -> ExtractionResult:
    messages = build_coverage_messages(chunk, cfg, existing)
    schema = copy.deepcopy(load_schema("extract.json"))
    schema["properties"]["items"]["maxItems"] = max(1, cfg.coverage_max_items)
    completion = backend.complete(
        messages,
        schema=schema,
        max_tokens=cfg.max_tokens,
        temperature=min(0.5, cfg.temperature + 0.15),
    )
    result = ExtractionResult(
        chunk_index=chunk.index,
        raw=completion.text,
        usage=dict(completion.usage),
    )
    try:
        payload = extract_json(completion.text)
    except BackendError as exc:
        result.parse_error = True
        result.rejected.append({"reason": f"coverage unparseable: {exc}", "raw": completion.text[:300]})
        return result
    items = payload.get("items")
    if not isinstance(items, list):
        result.parse_error = True
        result.rejected.append({"reason": "coverage missing items array", "raw": completion.text[:300]})
        return result
    candidates: list[Candidate] = []
    for item in items[: cfg.coverage_max_items]:
        candidate, errors = validate_item(item, chunk)
        if candidate is None:
            result.rejected.append({"reason": "; ".join(errors), "item": item})
        else:
            candidates.append(candidate)
    result.candidates = candidates
    return result


def extract_chunk(backend: Backend, chunk: Chunk, cfg: ExtractCfg) -> ExtractionResult:
    messages = build_messages(chunk, cfg)
    schema = copy.deepcopy(load_schema("extract.json"))
    schema["properties"]["items"]["maxItems"] = max(1, cfg.max_items)
    completion = backend.complete(
        messages,
        schema=schema,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    result = ExtractionResult(
        chunk_index=chunk.index,
        raw=completion.text,
        usage=dict(completion.usage),
    )
    try:
        payload = extract_json(completion.text)
    except BackendError as exc:
        result.parse_error = True
        result.rejected.append({"reason": f"unparseable: {exc}", "raw": completion.text[:300]})
        return result
    items = payload.get("items")
    if not isinstance(items, list):
        result.parse_error = True
        result.rejected.append({"reason": "missing items array", "raw": completion.text[:300]})
        return result
    candidates: list[Candidate] = []
    for item in items[: cfg.max_items]:
        candidate, errors = validate_item(item, chunk)
        if candidate is None:
            result.rejected.append({"reason": "; ".join(errors), "item": item})
        else:
            candidates.append(candidate)
    result.candidates = dedupe_candidates(candidates)
    return result


def single_episode_unit(episode: Any, *, max_chars: int = 4000) -> Chunk:
    """One-turn unit used by the message-level fallback."""
    text = render_episode(episode, max_chars=max_chars)
    refs = resolve_dates(episode.text, episode.spoken_at, episode.tz)
    return Chunk(
        index=-1,
        episodes=[episode],
        text=text,
        tokens=estimate_tokens(text),
        date_refs=refs,
    )


def extract_all(
    backend: Backend,
    chunks: list[Chunk],
    cfg: ExtractCfg,
    *,
    max_calls: int = 40,
) -> tuple[list[ExtractionResult], dict[str, Any]]:
    results: list[ExtractionResult] = []
    usage: dict[str, Any] = {"calls": 0, "parse_errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for chunk in chunks[:max_calls]:
        result = extract_chunk(backend, chunk, cfg)
        results.append(result)
        usage["calls"] += 1
        if result.parse_error:
            usage["parse_errors"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in result.usage and isinstance(result.usage[key], (int, float)):
                usage[key] = usage.get(key, 0) + int(result.usage[key])

        derived = derive_candidates(chunk)
        if derived:
            result.candidates = dedupe_candidates(result.candidates + derived)

        if cfg.message_fallback and usage["calls"] < max_calls:
            cited = {
                ev.episode_id
                for cand in result.candidates
                for ev in cand.evidence
            }
            candidates_for_messages = [
                ep
                for ep in chunk.episodes
                if (not cfg.message_fallback_uncited_only or ep.id not in cited)
                and (cfg.message_fallback_include_assistant or ep.role == "user")
                and len(ep.text) >= cfg.message_fallback_min_chars
            ][: cfg.message_fallback_max_messages]
            fallback_cfg = copy.copy(cfg)
            fallback_cfg.max_items = cfg.message_fallback_max_items
            fallback_cfg.max_tokens = cfg.message_fallback_max_tokens
            fallback_cfg.coverage_pass = False
            for episode in candidates_for_messages:
                if usage["calls"] >= max_calls:
                    break
                unit = single_episode_unit(episode)
                fallback = extract_chunk(backend, unit, fallback_cfg)
                usage["calls"] += 1
                if fallback.parse_error:
                    usage["parse_errors"] += 1
                if not fallback.candidates and usage["calls"] < max_calls:
                    retry_cfg = copy.copy(fallback_cfg)
                    retry_cfg.temperature = 0.55
                    retry = extract_chunk(backend, unit, retry_cfg)
                    usage["calls"] += 1
                    if retry.candidates:
                        fallback = retry
                    else:
                        fallback.rejected.extend(retry.rejected)
                if fallback.candidates:
                    for candidate in fallback.candidates:
                        candidate.chunk_index = chunk.index
                    result.candidates = dedupe_candidates(result.candidates + fallback.candidates)
                result.rejected.extend(fallback.rejected)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if key in fallback.usage and isinstance(fallback.usage[key], (int, float)):
                        usage[key] = usage.get(key, 0) + int(fallback.usage[key])

        if cfg.profile_pass and usage["calls"] < max_calls:
            profile = profile_extract(backend, chunk, cfg)
            usage["calls"] += 1
            if profile.parse_error:
                usage["parse_errors"] += 1
            if profile.candidates:
                result.candidates = dedupe_candidates(result.candidates + profile.candidates)
            result.rejected.extend(profile.rejected)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if key in profile.usage and isinstance(profile.usage[key], (int, float)):
                    usage[key] = usage.get(key, 0) + int(profile.usage[key])
            result.raw = result.raw + "\n--- profile ---\n" + profile.raw

        if cfg.coverage_pass and usage["calls"] < max_calls:
            coverage = coverage_extract(backend, chunk, result.candidates, cfg)
            usage["calls"] += 1
            if coverage.parse_error:
                usage["parse_errors"] += 1
            if coverage.candidates:
                result.candidates = dedupe_candidates(result.candidates + coverage.candidates)
            result.rejected.extend(coverage.rejected)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if key in coverage.usage and isinstance(coverage.usage[key], (int, float)):
                    usage[key] = usage.get(key, 0) + int(coverage.usage[key])
            result.raw = result.raw + "\n--- coverage ---\n" + coverage.raw
    return results, usage
