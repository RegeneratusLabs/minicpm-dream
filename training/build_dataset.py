#!/usr/bin/env python
"""Build SFT JSONL from generated eval suites using the production prompts.

Every extraction example uses the exact system/user prompt and JSON schema the
runtime uses. Reconcile examples use the exact reconcile prompt. This is what
makes a fine-tune a drop-in replacement rather than a new interface.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from minicpm_dream.config import ExtractCfg, IngestCfg
from minicpm_dream.consolidate.reconcile import build_messages as reconcile_messages
from minicpm_dream.extract.extract import build_messages as extract_messages
from minicpm_dream.extract.validate import Candidate, Evidence
from minicpm_dream.ingest import read_jsonl
from minicpm_dream.prepare import build_chunks
from minicpm_dream.util.text import quote_in_source


def _load_suite(path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    labels = json.loads((path / "labels.json").read_text(encoding="utf-8"))
    tz = "Australia/Brisbane"
    first = json.loads((path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    tz = first.get("tz") or tz
    episodes = list(read_jsonl(path / "episodes.jsonl", tz=tz))
    return labels, episodes


def _episode_for_quote(episodes: list[Any], quote: str) -> Any | None:
    matches = [ep for ep in episodes if quote_in_source(quote, ep.text)]
    return matches[0] if matches else None


def _project(text: str) -> str:
    match = re.search(r"The ([^ ]+) project", text, re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _extract_record(
    chunk: Any,
    selected: list[dict[str, Any]],
    episodes: list[Any],
    cfg: ExtractCfg,
) -> dict[str, Any] | None:
    items: list[dict[str, Any]] = []
    for label in selected:
        episode = _episode_for_quote(episodes, label["quote"])
        if episode is None:
            continue
        items.append(
            {
                "kind": label["kind"],
                "section": label["section"],
                "text": label["text"],
                "subject": "",
                "entities": [],
                "date_ref": "none",
                "confidence": 0.9,
                "evidence": [{"episode_id": episode.id, "quote": label["quote"]}],
            }
        )
    if not items:
        return None
    target = {"items": items}
    return {
        "task": "extract",
        "messages": extract_messages(chunk, cfg) + [
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        ],
        "meta": {"chunk": chunk.index, "episodes": len(chunk.episodes), "items": len(items)},
    }


def _reconcile_records(
    labels: list[dict[str, Any]], episodes: list[Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label in labels:
        if label["category"] != "update" or not label.get("final"):
            continue
        old = next(
            (
                item
                for item in labels
                if item["category"] == "update"
                and not item.get("final")
                and _project(item["text"]) == _project(label["text"])
            ),
            None,
        )
        if old is None:
            continue
        episode = _episode_for_quote(episodes, label["quote"])
        if episode is None:
            continue
        candidate = Candidate(
            kind="fact",
            section="projects",
            text=label["text"],
            subject="",
            entities=[],
            date_ref="none",
            valid_to_ref="none",
            confidence=0.9,
            evidence=[Evidence(episode.id, label["quote"], verified=True)],
            chunk_index=0,
            valid_from=label.get("valid_from"),
        )
        row = {
            "id": "mem_old",
            "text": old["text"],
            "section": "projects",
            "kind": "fact",
            "recorded_at": datetime.fromisoformat(episode.spoken_at_local).isoformat(),
            "age": "10d",
        }
        target = {
            "op": "SUPERSEDE",
            "target_ids": ["mem_old"],
            "merged_text": label["text"],
            "reason": "changed",
            "confidence": 0.9,
        }
        records.append(
            {
                "task": "reconcile",
                "messages": reconcile_messages(candidate, [row])
                + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
                "meta": {"source": "update", "target": "mem_old"},
            }
        )
    return records


def build(
    suites: list[Path],
    *,
    chunk_tokens: int = 1500,
    valid_ratio: float = 0.1,
    tasks: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tasks = tasks or {"extract", "reconcile"}
    extract_cfg = ExtractCfg(max_chunk_tokens=chunk_tokens)
    ingest_cfg = IngestCfg()
    extract_records: list[dict[str, Any]] = []
    reconcile_records: list[dict[str, Any]] = []
    for suite in suites:
        labels, episodes = _load_suite(suite)
        if not episodes:
            continue
        chunks = build_chunks(episodes, extract=extract_cfg, ingest=ingest_cfg, tz=episodes[0].tz)
        for chunk in chunks:
            selected = [
                label
                for label in labels
                if any(quote_in_source(label["quote"], ep.text) for ep in chunk.episodes)
            ]
            record = _extract_record(chunk, selected, episodes, extract_cfg)
            if record:
                extract_records.append(record)
        reconcile_records.extend(_reconcile_records(labels, episodes))

    records: list[dict[str, Any]] = []
    if "extract" in tasks:
        records.extend(extract_records)
    if "reconcile" in tasks:
        records.extend(reconcile_records)
    # Deterministic split by record index.
    valid: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if valid_ratio > 0 and index % max(1, round(1 / max(valid_ratio, 0.01))) == 0:
            valid.append(record)
        else:
            train.append(record)
    manifest = {
        "suites": [str(path) for path in suites],
        "train": len(train),
        "valid": len(valid),
        "extract": len(extract_records),
        "reconcile": len(reconcile_records),
        "tasks": sorted(tasks),
    }
    return train, valid, manifest


def write_records(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", required=True, help="suite directory; repeatable")
    parser.add_argument("--out", default="training/data")
    parser.add_argument("--chunk-tokens", type=int, default=1500)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--tasks", default="extract,reconcile")
    args = parser.parse_args(argv)
    suites = [Path(item) for item in args.suite]
    tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    train, valid, manifest = build(
        suites, chunk_tokens=args.chunk_tokens, valid_ratio=args.valid_ratio, tasks=tasks
    )
    out = Path(args.out)
    write_records(train, out / "train.jsonl")
    write_records(valid, out / "valid.jsonl")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
