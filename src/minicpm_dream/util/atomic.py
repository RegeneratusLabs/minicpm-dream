"""Atomic file writes with restrictive permissions by default."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass
    return p


def write_text_atomic(path: str | Path, text: str, *, mode: int = 0o600) -> Path:
    p = Path(path).expanduser()
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, p)
        try:
            dir_fd = os.open(p.parent, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def write_json_atomic(path: str | Path, payload: object, *, mode: int = 0o600) -> Path:
    return write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", mode=mode
    )


def read_text(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")
