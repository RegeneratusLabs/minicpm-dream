"""Extraction stage: one constrained SLM call per chunk."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import Backend, BackendError, extract_json
from ..config import ExtractCfg
from ..prepare.chunk import Chunk
from ..prepare.timeanchor import render_refs
from .schema import load_schema, prompt_version, render_prompt
from .validate import Candidate, dedupe_candidates, validate_item

PROMPT = "extract.j2"


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
    system = render_prompt(PROMPT, date_refs=render_refs(chunk.date_refs), max_items=cfg.max_items)
    user = (
        "EPISODES:\n"
        f"{chunk.text}\n\n"
        "Return only the JSON object now. Use at most "
        f"{cfg.max_items} items."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
    return results, usage
