"""Backend construction from configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import Config
from ..models import resolve_model_path
from .base import Backend, BackendUnavailable
from .llama_cpp import LlamaCppBackend
from .mock import MockBackend
from .openai_compat import OpenAICompatBackend


def make_backend(
    cfg: Config,
    *,
    mock_script: Mapping[str, Any] | str | Path | None = None,
    start: bool = True,
) -> Backend:
    backend = cfg.model.backend.strip().lower()
    if backend in {"mock", "none"}:
        script: Mapping[str, Any] | None = None
        if mock_script:
            if isinstance(mock_script, (str, Path)):
                script = json.loads(Path(mock_script).read_text(encoding="utf-8"))
            else:
                script = mock_script
        return MockBackend(script)
    if backend in {"openai", "openai_compat", "vllm", "sglang", "ollama"}:
        key = os.environ.get(cfg.model.api_key_env) if cfg.model.api_key_env else None
        return OpenAICompatBackend(
            base_url=cfg.model.base_url,
            model=cfg.model.name,
            api_key=key,
            timeout_s=cfg.model.timeout_s,
            default_temperature=cfg.model.temperature,
            default_max_tokens=cfg.model.max_tokens,
            disable_thinking=cfg.model.disable_thinking,
        )
    if backend == "llama_cpp":
        model_path = resolve_model_path(cfg)
        client = LlamaCppBackend(
            model_path=str(model_path),
            llama_server=cfg.model.llama_server,
            context=cfg.model.context,
            threads=cfg.model.threads,
            gpu_layers=cfg.model.gpu_layers,
            startup_timeout_s=cfg.model.startup_timeout_s,
            timeout_s=cfg.model.timeout_s,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
            log_path=Path(cfg.paths.logs) / "llama-server.log",
            disable_thinking=cfg.model.disable_thinking,
        )
        return client.start() if start else client
    raise BackendUnavailable(
        f"unknown backend '{cfg.model.backend}'; use llama_cpp, openai or mock"
    )
