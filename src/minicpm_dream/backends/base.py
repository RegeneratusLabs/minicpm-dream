"""Backend protocol and shared helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


class BackendError(RuntimeError):
    """The backend answered, but not usefully."""


class BackendUnavailable(BackendError):
    """The backend cannot be reached or configured."""


@dataclass
class Completion:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


@runtime_checkable
class Backend(Protocol):
    name: str
    model: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> Completion: ...

    def close(self) -> None: ...


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = _CODE_FENCE.sub("", text).strip()
    return text


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model response.

    Grammar-constrained decoding should make this unnecessary, but unconstrained
    backends and prompt regressions must not crash the pipeline.
    """
    cleaned = strip_code_fence(text)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise BackendError("model response was not a JSON object")


def task_marker(messages: list[dict[str, str]]) -> str:
    """Extract the `TASK: NAME` marker used by prompts and the mock backend."""
    for msg in messages:
        if msg.get("role") == "system":
            match = re.search(r"TASK:\s*([A-Z_]+)", msg.get("content", ""))
            if match:
                return match.group(1)
    for msg in messages:
        match = re.search(r"TASK:\s*([A-Z_]+)", msg.get("content", ""))
        if match:
            return match.group(1)
    return "UNKNOWN"
