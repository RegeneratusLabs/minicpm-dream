from __future__ import annotations

import json
from pathlib import Path

from minicpm_dream.backends.mock import MockBackend
from minicpm_dream.config import RecallCfg
from minicpm_dream.eval.harness import run_case_ledger
from minicpm_dream.eval.qa import QAItem, auto_items, evaluate_qa_case, score_answer

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evals" / "synthetic-days" / "basic"


def test_score_answer_requires_terms_and_respects_exclusions():
    item = QAItem(question="q", must_include=["postgres"], must_exclude=["sqlite"])
    assert score_answer(item, "It uses PostgreSQL.")
    assert not score_answer(item, "It uses SQLite.")
    assert not score_answer(item, "UNKNOWN")


def test_score_answer_matches_written_dates():
    item = QAItem(question="when", must_include=["2026-11-06"])
    assert score_answer(item, "The dentist is on 6 November 2026.")
    assert score_answer(item, "November 6, 2026 at 3pm")
    assert not score_answer(item, "The dentist is on 2026-11-07.")


def test_score_answer_supports_alternatives():
    item = QAItem(question="tea", must_include_any=["jasmine", "茉莉", "花茶"])
    assert score_answer(item, "茉莉花茶")
    assert score_answer(item, "Jasmine tea.")
    assert not score_answer(item, "Green tea.")


def test_auto_items_are_cloze_questions_that_do_not_give_the_answer():
    gold = json.loads((FIXTURE / "gold.json").read_text(encoding="utf-8"))
    items = auto_items(gold)
    assert items
    for item in items:
        assert item.origin == "auto"
        for term in item.must_include:
            assert term not in item.question.casefold()


def test_evaluate_qa_case_measures_the_gap_between_with_and_without_memory(cfg):
    cfg.recall = RecallCfg()
    backend = MockBackend(json.loads((FIXTURE / "mock_script.json").read_text(encoding="utf-8")))
    report = evaluate_qa_case(
        cfg, backend, FIXTURE, run_ledger=run_case_ledger, k=6, baseline=True
    )
    assert report["items"] >= 4
    assert report["accuracy_with_memory"] >= 0.5
    assert report["accuracy_without_memory"] == 0.0
    assert report["delta"] >= 0.5
    assert report["retrieval_hit_rate"] >= 0.5
    assert report["recall_p50_ms"] >= 0
    assert report["ledger"]["active"] >= 3  # counted before the temp ledger is removed
    assert any(row["retrieval_hit"] for row in report["per_item"])
    # The without-memory answers must be explicit unknowns, not invented facts.
    assert all(
        row["answer_without_memory"].strip().upper() == "UNKNOWN" for row in report["per_item"]
    )


def test_load_items_reads_the_shipped_qa_fixtures():
    from minicpm_dream.eval.qa import load_items

    items = load_items(FIXTURE)
    assert len(items) >= 4
    assert all(item.question for item in items)
