from __future__ import annotations

import json
from pathlib import Path

from minicpm_dream.cli import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evals" / "synthetic-days" / "basic"


def test_doctor_is_usable_without_a_model(tmp_path, capsys):
    code = main(["doctor", "--data-dir", str(tmp_path / "data"), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert any(check["name"] == "sqlite fts5" and check["ok"] for check in out["checks"])


def test_cli_run_and_gate(tmp_path, capsys):
    data_dir = str(tmp_path / "data")
    hermes_home = tmp_path / "hermes"
    (hermes_home / "memories").mkdir(parents=True)
    for name in ("MEMORY.md", "USER.md"):
        (hermes_home / "memories" / name).write_text("", encoding="utf-8")
    code = main(
        [
            "run",
            "--source", "jsonl",
            "--input", str(FIXTURE / "episodes.jsonl"),
            "--backend", "mock",
            "--mock-script", str(FIXTURE / "mock_script.json"),
            "--data-dir", data_dir,
            "--hermes-home", str(hermes_home),
            "--since", "2026-11-04T14:00:00+00:00",
            "--until", "2026-11-05T14:00:00+00:00",
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "3 changes" in text

    code = main(["gate", "--data-dir", data_dir])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["wakeAgent"] is True
    assert payload["context"]["review_actions"] == 2

    code = main(["review", "list", "--data-dir", data_dir])
    assert code == 0
    assert "proposals" in capsys.readouterr().out
