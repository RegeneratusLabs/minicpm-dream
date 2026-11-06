"""Cross-platform advisory file lock used to serialize consolidation runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]


class LockError(RuntimeError):
    pass


class FileLock:
    """Advisory exclusive lock. Blocks up to ``timeout`` seconds."""

    def __init__(self, path: str | Path, timeout: float = 15.0):
        self.path = Path(path).expanduser()
        self.timeout = timeout
        self._fh: Any = None

    def acquire(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover
                    raise LockError("no file locking primitive available")
                self._fh = fh
                return self
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    fh.close()
                    raise LockError(
                        f"another minicpm-dream run holds {self.path}; "
                        "wait for it to finish or remove a stale lock"
                    ) from None
                time.sleep(0.2)

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
