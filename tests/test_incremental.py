from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from minicpm_dream.backends.mock import MockBackend
from minicpm_dream.pipeline import run_pass
from minicpm_dream.util.timeutil import day_window

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evals" / "synthetic-days" / "basic"


def _backend() -> MockBackend:
    return MockBackend(json.loads((FIXTURE / "mock_script.json").read_text(encoding="utf-8")))


def test_incremental_second_pass_reads_nothing_new(cfg):
    since, until = day_window(date(2026, 11, 5), cfg.timezone)
    first = run_pass(
        cfg,
        _backend(),
        source="jsonl",
        since=since,
        until=until,
        window_label="first",
        input_path=str(FIXTURE / "episodes.jsonl"),
        incremental=True,
    )
    assert first.status == "ok"
    assert first.counters["episodes_kept"] > 0
    second = run_pass(
        cfg,
        _backend(),
        source="jsonl",
        since=since,
        until=until,
        window_label="second",
        input_path=str(FIXTURE / "episodes.jsonl"),
        incremental=True,
    )
    assert second.status == "empty"
    assert second.counters["episodes_kept"] == 0
    assert second.counters["candidates"] == 0


def test_cursor_is_only_written_on_a_successful_pass(cfg):
    since, until = day_window(date(2026, 11, 5), cfg.timezone)
    run_pass(
        cfg,
        _backend(),
        source="jsonl",
        since=since,
        until=until,
        window_label="first",
        input_path=str(FIXTURE / "episodes.jsonl"),
        incremental=False,
    )
    from minicpm_dream.store.db import Store

    with Store(cfg.paths.db) as store:
        assert store.kv_get("cursor:jsonl") is None
    run_pass(
        cfg,
        _backend(),
        source="jsonl",
        since=since,
        until=until,
        window_label="second",
        input_path=str(FIXTURE / "episodes.jsonl"),
        incremental=True,
    )
    with Store(cfg.paths.db) as store:
        assert store.kv_get("cursor:jsonl")
