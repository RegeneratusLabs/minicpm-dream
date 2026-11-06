"""Small, deterministic evaluation harness for extraction/consolidation quality."""

from __future__ import annotations

import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from ..backends.base import Backend
from ..config import Config, PathsCfg
from ..ingest.redact import has_secret
from ..pipeline import run_pass
from ..store.db import Store
from ..util.text import containment, jaccard
from ..util.timeutil import parse_timestamp

MATCH_THRESHOLD = 0.6


def discover_cases(suite: str | Path) -> list[Path]:
    base = Path(suite).expanduser()
    if (base / "episodes.jsonl").is_file():
        return [base]
    if not base.is_dir():
        raise FileNotFoundError(f"eval suite not found: {base}")
    return sorted(
        child for child in base.iterdir() if child.is_dir() and (child / "episodes.jsonl").is_file()
    )


def load_gold(case: Path) -> dict[str, Any]:
    path = case / "gold.json"
    if not path.is_file():
        return {"facts": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _score(gold: str, memories: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for memory in memories:
        text = str(memory.get("text", ""))
        score = max(jaccard(gold, text), containment(gold, text), containment(text, gold))
        if score > best[0]:
            best = (score, memory)
    return best


def evaluate_case(cfg: Config, backend: Backend, case: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"minicpm-dream-eval-{case.name}-") as td:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.paths = PathsCfg(data_dir=td)
        run_cfg.hermes.home = str(Path(td) / "hermes")
        run_cfg.hermes.write_mode = "proposals"
        run_cfg.synth.diary_path = str(Path(td) / "DREAMS.md")
        run_cfg.resolve()
        (Path(run_cfg.hermes.home) / "memories").mkdir(parents=True, exist_ok=True)
        for name in ("MEMORY.md", "USER.md"):
            (Path(run_cfg.hermes.home) / "memories" / name).write_text("", encoding="utf-8")

        episodes = [
            json.loads(line)
            for line in (case / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stamps = [parse_timestamp(item.get("timestamp"), run_cfg.timezone) for item in episodes]
        since = min(stamps).replace(microsecond=0)
        until = max(stamps).replace(microsecond=0)
        from datetime import timedelta

        since = since - timedelta(seconds=1)
        until = until + timedelta(seconds=1)

        started = time.monotonic()
        result = run_pass(
            run_cfg,
            backend,
            source="jsonl",
            since=since,
            until=until,
            window_label=f"eval:{case.name}",
            input_path=str(case / "episodes.jsonl"),
        )
        duration = time.monotonic() - started

        with Store(run_cfg.paths.db) as store:
            active = store.active_memories()
        counters = result.counters
        gold = load_gold(case).get("facts", [])
        matched: set[str] = set()
        true_positive = 0
        temporal_ok = 0
        temporal_expected = 0
        per_fact = []
        for fact in gold:
            score, memory = _score(str(fact.get("text", "")), active)
            hit = score >= MATCH_THRESHOLD and memory is not None
            if hit and memory is not None:
                true_positive += 1
                matched.add(str(memory.get("id")))
            per_fact.append(
                {
                    "text": fact.get("text", ""),
                    "matched": hit,
                    "score": round(score, 3),
                    "memory_id": memory.get("id") if memory else None,
                }
            )
            if fact.get("valid_from"):
                temporal_expected += 1
                if hit and memory and str(memory.get("valid_from") or "") == str(fact["valid_from"]):
                    temporal_ok += 1
        false_positive = max(0, len(active) - len(matched))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, len(gold))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        parse_errors = int(counters.get("parse_errors", 0))
        chunks = int(counters.get("chunks", 0))
        schema_valid = 1.0 if parse_errors == 0 else round(max(0, chunks - parse_errors) / max(1, chunks), 3)
        safety = sum(1 for memory in active if has_secret(str(memory.get("text", ""))))
        return {
            "case": case.name,
            "episodes": len(episodes),
            "active_memories": len(active),
            "gold_facts": len(gold),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "schema_valid": schema_valid,
            "temporal_accuracy": round(temporal_ok / max(1, temporal_expected), 3),
            "safety_violations": safety,
            "applied": int(counters.get("applied", 0)),
            "deferred": int(counters.get("deferred", 0)),
            "duration_s": round(duration, 2),
            "per_fact": per_fact,
            "run_status": result.status,
        }


def run_suite(cfg: Config, backend: Backend, suite: str | Path) -> dict[str, Any]:
    cases = discover_cases(suite)
    if not cases:
        raise FileNotFoundError(f"no cases with episodes.jsonl under {suite}")
    per_case = [evaluate_case(cfg, backend, case) for case in cases]
    n = len(per_case)

    def mean(key: str) -> float:
        return round(sum(float(case[key]) for case in per_case) / n, 3)

    metrics = {
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "schema_valid": mean("schema_valid"),
        "temporal_accuracy": mean("temporal_accuracy"),
        "safety_violations": sum(int(case["safety_violations"]) for case in per_case),
        "duration_s": round(sum(float(case["duration_s"]) for case in per_case), 2),
    }
    return {"suite": str(Path(suite)), "cases": n, "metrics": metrics, "per_case": per_case}
