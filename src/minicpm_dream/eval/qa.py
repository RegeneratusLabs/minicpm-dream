"""End-to-end QA evaluation: does the agent answer correctly *because of* memory?

The extraction suite in `harness.py` scores how closely stored sentences match a
gold list. That is a diagnostic. It cannot tell you whether the memory is useful,
because the useful question is "when the user asks this, does the agent get the
right answer?" This module builds a ledger from a transcript, asks questions
through the same recall path an agent uses at turn time, and scores the answers
with and without memory.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..backends.base import Backend
from ..config import Config
from ..recall import RecallService
from ..store.db import Store
from ..util.text import containment, jaccard, normalize_text

ANSWER_TASK = "ANSWER"

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY = re.compile(r"\b(\d{1,2})\s+(?:of\s+)?([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")
_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b")
_BLANK = "____"

_SKIP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "was",
    "were", "are", "his", "her", "their", "user", "users", "into", "over",
    "when", "what", "which", "there", "will", "would", "should", "could",
}


def extract_dates(text: str) -> set[date]:
    out: set[date] = set()
    for match in _ISO.finditer(text or ""):
        try:
            out.add(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
        except ValueError:
            continue
    for match in _DMY.finditer(text or ""):
        month = _MONTHS.get(match.group(2).lower())
        if month:
            try:
                out.add(date(int(match.group(3)), month, int(match.group(1))))
            except ValueError:
                pass
    for match in _MDY.finditer(text or ""):
        month = _MONTHS.get(match.group(1).lower())
        if month:
            try:
                out.add(date(int(match.group(3)), month, int(match.group(2))))
            except ValueError:
                pass
    return out


@dataclass
class QAItem:
    question: str
    must_include: list[str] = field(default_factory=list)
    must_include_any: list[str] = field(default_factory=list)
    must_exclude: list[str] = field(default_factory=list)
    gold_texts: list[str] = field(default_factory=list)
    sections: list[str] | None = None
    note: str = ""
    origin: str = "hand"

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "must_include": self.must_include,
            "must_include_any": self.must_include_any,
            "must_exclude": self.must_exclude,
            "gold_texts": self.gold_texts,
            "sections": self.sections,
            "note": self.note,
            "origin": self.origin,
        }


def load_items(case: Path) -> list[QAItem]:
    path = case / "qa.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    items: list[QAItem] = []
    for raw in payload.get("items", []):
        if not isinstance(raw, dict) or not raw.get("question"):
            continue
        items.append(
            QAItem(
                question=str(raw["question"]),
                must_include=[str(t) for t in raw.get("must_include", [])],
                must_include_any=[str(t) for t in raw.get("must_include_any", [])],
                must_exclude=[str(t) for t in raw.get("must_exclude", [])],
                gold_texts=[str(t) for t in raw.get("gold_texts", [])],
                sections=[str(s) for s in raw["sections"]] if raw.get("sections") else None,
                note=str(raw.get("note", "")),
            )
        )
    return items


def _blank_candidate(text: str) -> tuple[str, str] | None:
    words = normalize_text(text).split()
    if len(words) < 4:
        return None
    best: tuple[int, int, str] | None = None
    for index, word in enumerate(words):
        core = word.strip(".,;:()'\"!?")
        if len(core) < 3 or core.lower() in _SKIP_WORDS:
            continue
        score = 0
        if any(ch.isdigit() for ch in core):
            score += 3
        if core[0].isupper() and index > 0:
            score += 2
        if len(core) >= 5:
            score += 1
        if best is None or score > best[0]:
            best = (score, index, core)
    if best is None:
        return None
    score, index, core = best
    blanked = " ".join(words[:index] + [_BLANK] + words[index + 1 :])
    return blanked, core


def auto_items(gold: dict[str, Any]) -> list[QAItem]:
    """Turn gold facts into cloze questions answerable only from memory."""
    items: list[QAItem] = []
    seen: set[str] = set()
    for fact in gold.get("facts", []):
        text = normalize_text(str(fact.get("text", "")))
        if not text or text in seen:
            continue
        blanked = _blank_candidate(text)
        if blanked is None:
            continue
        question, answer = blanked
        seen.add(text)
        items.append(
            QAItem(
                question=(
                    "Fill in the missing detail in this statement about the user, using memory "
                    f"only: {question}"
                ),
                must_include=[answer.lower()],
                gold_texts=[text],
                sections=[str(fact["section"])] if fact.get("section") else None,
                origin="auto",
            )
        )
    return items


def score_answer(item: QAItem, answer: str) -> bool:
    text = (answer or "").casefold()
    if not text.strip():
        return False
    for term in item.must_include:
        if _term_present(term, text, answer):
            continue
        return False
    if item.must_include_any and not any(
        _term_present(term, text, answer) for term in item.must_include_any
    ):
        return False
    return not any(term.casefold() in text for term in item.must_exclude)


def _term_present(term: str, folded_answer: str, raw_answer: str) -> bool:
    needle = term.casefold().strip()
    if not needle:
        return True
    if needle in folded_answer:
        return True
    try:
        wanted = date.fromisoformat(needle)
    except ValueError:
        return False
    return wanted in extract_dates(raw_answer)


def _hit_matches(hit: dict[str, Any], gold_texts: list[str]) -> bool:
    if not gold_texts:
        return False
    text = str(hit.get("text", ""))
    for gold in gold_texts:
        if max(jaccard(gold, text), containment(gold, text), containment(text, gold)) >= 0.5:
            return True
    return False


def _answer(backend: Backend, item: QAItem, memory_block: str) -> tuple[str, dict[str, Any]]:
    system = (
        "You answer questions about a user from a private memory of their past conversations. "
        "Use only the memory provided. If the memory does not contain the answer, reply with "
        "exactly UNKNOWN. Answer in one short sentence and never invent dates or names.\n"
        f"TASK: {ANSWER_TASK}"
    )
    memory = memory_block.strip() or "(no memory available)"
    user = f"MEMORY\n{memory}\n\nQUESTION\n{item.question}"
    completion = backend.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        schema=None,
        max_tokens=120,
        temperature=0.0,
    )
    return completion.text, dict(completion.usage or {})


def evaluate_qa_case(
    cfg: Config,
    backend: Backend,
    case: Path,
    *,
    run_ledger: Any,
    k: int = 6,
    stale_days: int = 180,
    baseline: bool = True,
    items: list[QAItem] | None = None,
) -> dict[str, Any]:
    """Answer every question through the real recall path, with and without memory."""
    if items is None:
        items = load_items(case)
    if not items:
        gold = json.loads((case / "gold.json").read_text(encoding="utf-8")) if (case / "gold.json").is_file() else {}
        items = auto_items(gold)
    if not items:
        return {"case": case.name, "items": 0, "error": "no QA items and no gold facts to derive them from"}

    run_cfg, result, temp_dir, _duration = run_ledger(cfg, backend, case)
    service = RecallService(run_cfg, stale_days=stale_days, min_coverage=run_cfg.recall.min_coverage)
    per_item: list[dict[str, Any]] = []
    prompt_tokens_with = 0
    prompt_tokens_without = 0
    recall_ms: list[float] = []
    try:
        for item in items:
            started = time.perf_counter()
            recall = service.recall(item.question, k=k, sections=item.sections)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            recall_ms.append(elapsed)
            hit = any(_hit_matches(hit.as_dict(), item.gold_texts) for hit in recall.hits)
            block = "\n".join(recall.render().splitlines()) if recall.hits and not recall.abstained else ""
            answer_with, usage_with = _answer(backend, item, block)
            prompt_tokens_with += int(usage_with.get("prompt_tokens", 0))
            answer_without = ""
            if baseline:
                answer_without, usage_without = _answer(backend, item, "")
                prompt_tokens_without += int(usage_without.get("prompt_tokens", 0))
            per_item.append(
                {
                    **item.as_dict(),
                    "answer_with_memory": answer_with,
                    "ok_with_memory": score_answer(item, answer_with),
                    "answer_without_memory": answer_without,
                    "ok_without_memory": score_answer(item, answer_without) if baseline else None,
                    "retrieval_hit": hit,
                    "abstained": recall.abstained,
                    "recall_ms": elapsed,
                    "hits": [h.as_dict() for h in recall.hits[:3]],
                }
            )
    finally:
        service.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(Path(temp_dir) / "ledger.db") + suffix).unlink(missing_ok=True)
            except OSError:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)

    n = len(per_item)
    with_ok = sum(1 for row in per_item if row["ok_with_memory"])
    without_ok = sum(1 for row in per_item if row["ok_without_memory"])
    recall_ms.sort()

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
        return round(values[index], 2)

    return {
        "case": case.name,
        "items": n,
        "ledger": {
            "status": result.status,
            "candidates": result.counters.get("candidates", 0),
            "applied": result.counters.get("applied", 0),
            "active": len(store_active(run_cfg)),
        },
        "retrieval_hit_rate": round(sum(1 for row in per_item if row["retrieval_hit"]) / n, 3),
        "abstain_rate": round(sum(1 for row in per_item if row["abstained"]) / n, 3),
        "accuracy_with_memory": round(with_ok / n, 3),
        "accuracy_without_memory": round(without_ok / n, 3) if baseline else None,
        "delta": round((with_ok - without_ok) / n, 3) if baseline else None,
        "recall_p50_ms": percentile(recall_ms, 0.5),
        "recall_p95_ms": percentile(recall_ms, 0.95),
        "prompt_tokens_with_memory": prompt_tokens_with,
        "prompt_tokens_without_memory": prompt_tokens_without,
        "per_item": per_item,
    }


def store_active(cfg: Config) -> list[dict[str, Any]]:
    try:
        with Store(cfg.paths.db) as store:
            return store.active_memories()
    except Exception:  # noqa: BLE001
        return []


def run_qa_suite(
    cfg: Config,
    backend: Backend,
    suite: str | Path,
    *,
    run_ledger: Any,
    k: int = 6,
    stale_days: int = 180,
    baseline: bool = True,
) -> dict[str, Any]:
    base = Path(suite).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"eval suite not found: {base}")
    cases = (
        [base]
        if (base / "episodes.jsonl").is_file()
        else sorted(child for child in base.iterdir() if child.is_dir() and (child / "episodes.jsonl").is_file())
    )
    if not cases:
        raise FileNotFoundError(f"no cases with episodes.jsonl under {base}")
    per_case = [
        evaluate_qa_case(
            cfg, backend, case, run_ledger=run_ledger, k=k, stale_days=stale_days, baseline=baseline
        )
        for case in cases
    ]
    scored = [case for case in per_case if case.get("items")]
    n = max(1, len(scored))
    metrics = {
        "items": sum(int(case["items"]) for case in scored),
        "retrieval_hit_rate": round(sum(float(case["retrieval_hit_rate"]) for case in scored) / n, 3),
        "accuracy_with_memory": round(sum(float(case["accuracy_with_memory"]) for case in scored) / n, 3),
        "accuracy_without_memory": (
            round(sum(float(case["accuracy_without_memory"] or 0) for case in scored) / n, 3)
            if baseline
            else None
        ),
        "delta": (
            round(sum(float(case["delta"] or 0) for case in scored) / n, 3) if baseline else None
        ),
        "recall_p50_ms": max((float(case["recall_p50_ms"]) for case in scored), default=0.0),
        "recall_p95_ms": max((float(case["recall_p95_ms"]) for case in scored), default=0.0),
        "prompt_tokens_with_memory": sum(int(case["prompt_tokens_with_memory"]) for case in scored),
        "prompt_tokens_without_memory": sum(
            int(case["prompt_tokens_without_memory"]) for case in scored
        ),
    }
    return {"suite": str(base), "cases": len(cases), "metrics": metrics, "per_case": per_case}
