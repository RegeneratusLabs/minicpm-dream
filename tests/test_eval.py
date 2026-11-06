from __future__ import annotations

import json
from pathlib import Path

from minicpm_dream.backends.mock import MockBackend
from minicpm_dream.eval.harness import discover_cases, run_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "evals" / "synthetic-days" / "basic"
SCRIPT = SUITE / "mock_script.json"


def test_discover_cases():
    cases = discover_cases(SUITE)
    assert [case.name for case in cases] == ["basic"]


def test_eval_suite_with_mock(cfg):
    backend = MockBackend(json.loads(SCRIPT.read_text(encoding="utf-8")))
    report = run_suite(cfg, backend, SUITE)
    assert report["cases"] == 1
    metrics = report["metrics"]
    assert metrics["precision"] >= 0.99
    assert metrics["recall"] >= 0.99
    assert metrics["schema_valid"] == 1.0
    assert metrics["safety_violations"] == 0
    assert metrics["temporal_accuracy"] == 1.0
