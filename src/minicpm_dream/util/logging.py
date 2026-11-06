"""Logging setup: human-readable stderr plus an optional JSONL audit log."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .atomic import ensure_dir

_CONFIGURED = False


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(verbose: bool = False, log_file: str | Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("minicpm_dream")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.DEBUG if verbose else logging.INFO)
    stderr.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(stderr)

    if log_file:
        p = Path(log_file).expanduser()
        ensure_dir(p.parent)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JsonlFormatter())
        root.addHandler(fh)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"minicpm_dream.{name}")
