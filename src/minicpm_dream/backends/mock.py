"""Deterministic mock backend for tests, CI and the offline demo."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from .base import Completion, task_marker


class MockBackend:
    """Return scripted JSON per task; fall back to safe, useful defaults."""

    name = "mock"

    def __init__(
        self,
        script: Mapping[str, Iterable[str]] | None = None,
        *,
        model: str = "mock",
        default_temperature: float = 0.0,
    ):
        self.model = model
        self.default_temperature = default_temperature
        def as_text(value: Any) -> str:
            return value if isinstance(value, str) else json.dumps(value)

        self._queues: dict[str, deque[str]] = {
            str(k).upper(): deque(as_text(v) for v in values) for k, values in (script or {}).items()
        }
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        task = task_marker(messages)
        self.calls.append({"task": task, "messages": messages, "schema": schema})
        queue = self._queues.get(task)
        if queue:
            text = queue.popleft()
        else:
            text = self._default(task)
        return Completion(text=text, model=self.model, usage={"prompt_tokens": 0, "completion_tokens": 0})

    @staticmethod
    def _default(task: str) -> str:
        if task == "EXTRACT":
            return json.dumps({"items": []})
        if task == "VERIFY":
            return json.dumps({"verdict": "supports", "confidence": 0.9})
        if task == "RECONCILE":
            return json.dumps(
                {
                    "op": "ADD",
                    "target_ids": [],
                    "merged_text": None,
                    "reason": "new",
                    "confidence": 0.8,
                }
            )
        return "{}"

    def close(self) -> None:
        return None
