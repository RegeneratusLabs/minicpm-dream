"""OpenAI-compatible chat completions backend (llama.cpp, vLLM, SGLang, Ollama, LM Studio)."""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..util.logging import get_logger
from .base import BackendError, BackendUnavailable, Completion

log = get_logger("backends.openai")


class OpenAICompatBackend:
    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        default_temperature: float = 0.2,
        default_max_tokens: int = 1200,
        disable_thinking: bool = False,
        structured_mode: str = "auto",
        reasoning_effort: str | None = None,
    ):
        if not base_url:
            raise BackendUnavailable("model.base_url is empty (set it or use backend=llama_cpp)")
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        self.base_url = normalized
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.disable_thinking = disable_thinking
        self.structured_mode = structured_mode if structured_mode in {"auto", "json_schema", "json_object", "none"} else "auto"
        self.reasoning_effort = reasoning_effort
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout_s, headers=headers)

    def _set_response_format(self, payload: dict[str, Any], schema: dict[str, Any]) -> None:
        mode = self.structured_mode
        if mode in {"auto", "json_schema"}:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema},
            }
        elif mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        else:
            payload.pop("response_format", None)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if schema is not None:
            self._set_response_format(payload, schema)
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(f"{self.base_url}/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == 2:
                    raise BackendUnavailable(f"cannot reach model endpoint {self.base_url}: {exc}") from exc
                time.sleep(0.75 * (attempt + 1))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_error = BackendError(f"endpoint returned {response.status_code}: {response.text[:300]}")
                if attempt == 2:
                    raise last_error
                time.sleep(1.0 * (attempt + 1))
                continue
            if response.status_code == 400 and self.structured_mode == "auto" and schema is not None:
                body_text = response.text.lower()
                if "response_format" in body_text or "json_schema" in body_text:
                    log.warning("endpoint rejected json_schema; retrying with json_object")
                    self.structured_mode = "json_object"
                    payload["response_format"] = {"type": "json_object"}
                    continue
            if response.status_code >= 400:
                raise BackendError(
                    f"endpoint returned {response.status_code}: {response.text[:500]}"
                )
            try:
                body = response.json()
                choice = body["choices"][0]
                message = choice["message"]
                text = message.get("content") or ""
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise BackendError(f"malformed chat completion response: {exc}") from exc
            if not text and (message.get("reasoning_content") or choice.get("finish_reason") == "length"):
                raise BackendError(
                    "model returned no content (reasoning consumed the token budget). "
                    "Disable thinking in the server: llama.cpp `--reasoning off --reasoning-budget 0`; "
                    "vLLM/SGLang: disable the model's thinking template or lower reasoning budget."
                )
            return Completion(
                text=text,
                model=str(body.get("model") or self.model),
                usage=dict(body.get("usage") or {}),
                latency_s=time.monotonic() - started,
            )
        raise BackendError(f"request failed after retries: {last_error}")

    def close(self) -> None:
        self._client.close()
