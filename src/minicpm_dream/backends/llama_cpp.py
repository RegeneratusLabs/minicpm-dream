"""Managed llama.cpp server for the local MiniCPM model."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from ..util.atomic import ensure_dir
from ..util.logging import get_logger
from .base import BackendUnavailable, Completion
from .openai_compat import OpenAICompatBackend

log = get_logger("backends.llama")


def find_llama_server(explicit: str = "") -> str:
    candidates = [explicit, os.environ.get("LLAMA_SERVER", ""), shutil.which("llama-server") or ""]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())
    raise BackendUnavailable(
        "llama-server not found. Install llama.cpp and put `llama-server` on PATH, "
        "set model.llama_server / LLAMA_SERVER, or use an OpenAI-compatible endpoint "
        "(model.backend='openai', model.base_url='http://host:port/v1')."
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaCppBackend:
    """Start one llama-server, expose the OpenAI-compatible surface, stop it on close."""

    name = "llama_cpp"

    def __init__(
        self,
        *,
        model_path: str,
        llama_server: str = "",
        context: int = 8192,
        threads: int = 4,
        gpu_layers: int = 0,
        startup_timeout_s: float = 120.0,
        timeout_s: float = 180.0,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        log_path: str | Path | None = None,
        extra_args: list[str] | None = None,
        disable_thinking: bool = True,
    ):
        self.model_path = str(Path(model_path).expanduser())
        if not Path(self.model_path).is_file():
            raise BackendUnavailable(
                f"model file not found: {self.model_path}. Run `minicpm-dream setup --download` "
                "or pass --model-path."
            )
        self.binary = find_llama_server(llama_server)
        self.context = context
        self.threads = threads
        self.gpu_layers = gpu_layers
        self.startup_timeout_s = startup_timeout_s
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log_path = Path(log_path).expanduser() if log_path else None
        self.extra_args = extra_args or []
        self.disable_thinking = disable_thinking
        self.model = Path(self.model_path).stem
        self.port = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._client: OpenAICompatBackend | None = None
        self._log_handle: Any = None

    def _command(self, port: int) -> list[str]:
        thinking = ["--reasoning", "off", "--reasoning-budget", "0"] if self.disable_thinking else []
        return [
            self.binary,
            "-m", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(self.context),
            "-t", str(self.threads),
            "-ngl", str(self.gpu_layers),
            "--jinja",
            "--parallel", "1",
            *thinking,
            *self.extra_args,
        ]

    def start(self) -> LlamaCppBackend:
        if self._process is not None:
            return self
        self.port = _free_port()
        if self.log_path:
            ensure_dir(self.log_path.parent)
            self._log_handle = open(self.log_path, "ab")
            out = self._log_handle
        else:
            out = subprocess.DEVNULL
        log.info("starting %s (port %s, ctx %s, threads %s)", self.binary, self.port, self.context, self.threads)
        try:
            self._process = subprocess.Popen(
                self._command(self.port),
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise BackendUnavailable(f"failed to start llama-server: {exc}") from exc
        base = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + self.startup_timeout_s
        with httpx.Client(timeout=5.0) as probe:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise BackendUnavailable(
                        "llama-server exited during startup; see the log at "
                        f"{self.log_path or 'the process output'}"
                    )
                try:
                    if probe.get(f"{base}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            else:
                self.stop()
                raise BackendUnavailable(
                    f"llama-server did not become healthy within {self.startup_timeout_s:.0f}s"
                )
        self._client = OpenAICompatBackend(
            base_url=f"{base}/v1",
            model=self.model,
            timeout_s=self.timeout_s,
            default_temperature=self.temperature,
            default_max_tokens=self.max_tokens,
            disable_thinking=self.disable_thinking,
        )
        log.info("llama-server ready at %s", base)
        return self

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        if self._client is None:
            self.start()
        assert self._client is not None
        return self._client.complete(
            messages, schema=schema, max_tokens=max_tokens, temperature=temperature
        )

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> LlamaCppBackend:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
