from __future__ import annotations

from minicpm_dream.config import Config, load_config


def test_defaults_resolve(tmp_path):
    cfg = load_config(tmp_path / "none.toml", data_dir=str(tmp_path / "data"))
    assert cfg.paths.db.endswith("ledger.db")
    assert cfg.synth.diary_path.endswith("DREAMS.md")
    assert cfg.hermes.state_db.endswith("state.db")
    assert cfg.timezone


def test_toml_and_env(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        """
timezone = "UTC"
[model]
backend = "mock"
name = "minicpm5-1b-q4"
threads = 2
[extract]
max_items = 4
[hermes]
home = "/tmp/hermes-test"
""",
        encoding="utf-8",
    )
    cfg = load_config(path, data_dir=str(tmp_path / "data"))
    assert cfg.model.backend == "mock"
    assert cfg.model.threads == 2
    assert cfg.extract.max_items == 4
    assert cfg.hermes.home == "/tmp/hermes-test"
    monkeypatch.setenv("MINICPM_DREAM_BACKEND", "mock")
    assert load_config(path).model.backend == "mock"
    assert isinstance(cfg, Config)
