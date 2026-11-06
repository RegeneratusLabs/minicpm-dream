from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from minicpm_dream.backends.mock import MockBackend
from minicpm_dream.pipeline import run_pass
from minicpm_dream.store.db import Store
from minicpm_dream.util.timeutil import day_window

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evals" / "synthetic-days" / "basic"


def _backend() -> MockBackend:
    return MockBackend(json.loads((FIXTURE / "mock_script.json").read_text(encoding="utf-8")))


def test_pipeline_end_to_end(cfg):
    since, until = day_window(date(2026, 11, 5), cfg.timezone)
    result = run_pass(
        cfg,
        _backend(),
        source="jsonl",
        since=since,
        until=until,
        window_label="test window",
        input_path=str(FIXTURE / "episodes.jsonl"),
    )
    assert result.status == "ok"
    assert result.counters["episodes_kept"] == 7  # secret-bearing turn dropped
    assert result.counters["candidates"] == 3
    assert result.counters["verified"] == 3
    assert result.counters["applied"] == 3
    assert result.loss and result.loss["ok"]
    with Store(cfg.paths.db) as store:
        active = store.active_memories()
        assert len(active) == 3
        assert any("dentist" in row["text"] and row["valid_from"] == "2026-11-06" for row in active)
        assert store.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 3
    assert Path(result.dream_path).is_file()
    review = json.loads((cfg.exports_dir / "review.json").read_text(encoding="utf-8"))
    assert len(review["actions"]) == 2  # open_loops are diary-only
    assert (cfg.exports_dir / "MEMORY.candidate.md").is_file()
    assert (cfg.exports_dir / "USER.candidate.md").is_file()


def test_second_run_does_not_duplicate(cfg):
    since, until = day_window(date(2026, 11, 5), cfg.timezone)
    for _ in range(2):
        backend = MockBackend(json.loads((FIXTURE / "mock_script.json").read_text(encoding="utf-8")))
        result = run_pass(
            cfg,
            backend,
            source="jsonl",
            since=since,
            until=until,
            window_label="test window",
            input_path=str(FIXTURE / "episodes.jsonl"),
        )
        assert result.status == "ok"
    with Store(cfg.paths.db) as store:
        assert len(store.active_memories()) == 3


def test_dry_run_does_not_touch_real_ledger(cfg):
    since, until = day_window(date(2026, 11, 5), cfg.timezone)
    result = run_pass(
        cfg,
        _backend(),
        source="jsonl",
        since=since,
        until=until,
        window_label="test window",
        input_path=str(FIXTURE / "episodes.jsonl"),
        dry_run=True,
    )
    assert result.status == "ok"
    assert not Path(cfg.paths.db).exists()
    assert not Path(cfg.synth.diary_path).exists()
