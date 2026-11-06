from __future__ import annotations

from pathlib import Path

import pytest

from minicpm_dream.util.atomic import write_text_atomic
from minicpm_dream.util.ids import new_id, stable_id
from minicpm_dream.util.locking import FileLock, LockError
from minicpm_dream.util.text import (
    containment,
    is_probably_meta,
    jaccard,
    norm_key,
    normalize_text,
    quote_in_source,
)


def test_ids_are_sortable_and_prefixed():
    a = new_id("mem")
    b = new_id("mem")
    assert a.startswith("mem_") and b.startswith("mem_")
    assert a != b
    assert stable_id("x", "a", 1) == stable_id("x", "a", 1)


def test_text_helpers():
    assert normalize_text("  Hello\n\nworld  ") == "Hello world"
    assert norm_key("Hello, WORLD!") == "hello world"
    assert jaccard("dark mode editors", "dark mode in editors") > 0.7
    assert containment("dark mode", "I prefer dark mode in editors") == 1.0
    assert quote_in_source("dark mode", "I prefer   Dark Mode here")
    assert not quote_in_source("light mode", "I prefer dark mode")
    assert is_probably_meta("The user is asking about dinner")
    assert not is_probably_meta("Josh prefers dark mode")


def test_atomic_write_and_lock(tmp_path: Path):
    target = tmp_path / "nested" / "file.txt"
    write_text_atomic(target, "hello")
    assert target.read_text() == "hello"
    lock = FileLock(tmp_path / "run.lock", timeout=0.2)
    with lock:
        with pytest.raises(LockError):
            FileLock(tmp_path / "run.lock", timeout=0.1).acquire()
