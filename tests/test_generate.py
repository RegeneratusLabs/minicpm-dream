from __future__ import annotations

from datetime import date

from minicpm_dream.eval.generate import generate, write_suite


def test_generator_is_deterministic_and_grounded():
    first = generate(seed=7, days=3, facts=24, start_date=date(2026, 11, 1))
    second = generate(seed=7, days=3, facts=24, start_date=date(2026, 11, 1))
    assert first.episodes == second.episodes
    assert first.gold() == second.gold()
    assert len(first.facts) == 24
    assert len(first.labels) >= 24
    contents = [ep["content"] for ep in first.episodes]
    for label in first.labels:
        assert any(label.quote in content for content in contents), label.quote
    days = {ep["timestamp"][:10] for ep in first.episodes}
    assert len(days) == 3


def test_write_suite(tmp_path):
    generated = generate(seed=3, days=2, facts=12)
    paths = write_suite(generated, tmp_path / "suite")
    assert (tmp_path / "suite" / "episodes.jsonl").is_file()
    assert (tmp_path / "suite" / "gold.json").is_file()
    assert (tmp_path / "suite" / "labels.json").is_file()
    assert len(paths) == 4
