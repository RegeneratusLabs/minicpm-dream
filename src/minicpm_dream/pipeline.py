"""The nightly pass: ingest → extract → verify → reconcile → apply → synthesize."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .apply import ApplyReport, build_and_write, consolidate_active, serialize_decisions, snapshot
from .apply.exports import build_bundle
from .apply.ledger import LossGuardError
from .backends.base import Backend
from .config import Config
from .consolidate.reconcile import Decision
from .consolidate.sequential import reconcile_and_apply
from .extract import Candidate, extract_all, verify_all
from .ingest import Episode, EpisodeFilter, HermesAdapter, read_jsonl
from .prepare import build_chunks
from .retrieval.factory import build_retriever
from .store.db import Store
from .synth.dream_md import append_section, render_section
from .util.ids import new_id, stable_hash
from .util.locking import FileLock
from .util.logging import get_logger
from .util.timeutil import iso_utc, utc_now

log = get_logger("pipeline")


@dataclass
class RunResult:
    run_id: str
    status: str
    source: str
    window_start: str
    window_end: str
    counters: dict[str, Any] = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)
    bundle: dict[str, Any] | None = None
    export_paths: dict[str, str] = field(default_factory=dict)
    dream_path: str = ""
    loss: dict[str, Any] | None = None
    reports: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "source": self.source,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "counters": self.counters,
            "decisions": serialize_decisions(self.decisions),
            "bundle": self.bundle,
            "export_paths": self.export_paths,
            "dream_path": self.dream_path,
            "loss": self.loss,
            "reports": self.reports,
            "usage": self.usage,
            "error": self.error,
        }


def collect_episodes(
    cfg: Config,
    *,
    source: str,
    since: datetime,
    until: datetime,
    input_path: str | None,
    limit: int | None,
) -> tuple[list[Episode], Any]:
    if source == "hermes":
        adapter = HermesAdapter(cfg)
        generator = adapter.iter_episodes(since=since, until=until, tz=cfg.timezone, limit=limit)
    elif source == "jsonl":
        if not input_path:
            raise ValueError("--input is required for source=jsonl")
        generator = read_jsonl(input_path, tz=cfg.timezone)
    else:
        raise ValueError(f"unknown source {source!r}; use hermes or jsonl")

    filt = EpisodeFilter(cfg.ingest, cfg.privacy)
    kept: list[Episode] = []
    seen = 0
    for ep in generator:
        seen += 1
        if not (since <= ep.spoken_at < until):
            continue
        if filt.keep(ep):
            kept.append(ep)
        if len(kept) >= cfg.budgets.max_episodes:
            break
    return kept, filt.stats


def run_pass(
    cfg: Config,
    backend: Backend,
    *,
    source: str,
    since: datetime,
    until: datetime,
    window_label: str,
    input_path: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    apply_hermes: bool = False,
) -> RunResult:
    run_id = new_id("run")
    started = utc_now()
    t0 = time.monotonic()
    window_start = iso_utc(since)
    window_end = iso_utc(until)
    temp_db: Path | None = None
    if dry_run:
        temp_db = Path(cfg.paths.data_dir) / f".dryrun-{run_id}.db"
    db_path = temp_db or Path(cfg.paths.db)
    store = Store(db_path)
    result = RunResult(
        run_id=run_id,
        status="running",
        source=source,
        window_start=window_start,
        window_end=window_end,
    )
    counters: dict[str, Any] = {
        "episodes_seen": 0,
        "episodes_kept": 0,
        "chunks": 0,
        "candidates": 0,
        "verified": 0,
        "rejected_extract": 0,
        "parse_errors": 0,
        "rejected_verify": 0,
        "decisions": 0,
        "applied": 0,
        "deferred": 0,
        "proposals": 0,
        "consolidated": 0,
        "duration_s": 0.0,
    }
    usage: dict[str, Any] = {"extract": {}, "verify": {}, "reconcile": {}}
    notes: list[str] = []
    filt_stats: Any = None
    retriever: Any = None

    try:
        with FileLock(cfg.lock_path):
            store.record_run(
                run_id=run_id,
                source=source,
                window_start=window_start,
                window_end=window_end,
                config_hash=stable_hash(json.dumps(cfg.as_dict(), sort_keys=True, default=str)),
                model=backend.model,
                backend=backend.name,
            )
            episodes, filt_stats = collect_episodes(
                cfg, source=source, since=since, until=until, input_path=input_path, limit=limit
            )
            counters["episodes_seen"] = filt_stats.seen
            counters["episodes_kept"] = len(episodes)
            if temp_db is not None:
                notes.append("dry run: ledger writes were isolated in a temporary database")

            if not episodes:
                counters["duration_s"] = round(time.monotonic() - t0, 1)
                result.status = "empty"
                result.counters = counters
                result.usage = usage
                store.finish_run(run_id, status="empty", counters=counters, usage=usage)
                return result

            with store.transaction():
                for ep in episodes:
                    store.insert_episode(ep.to_record(store_text=cfg.privacy.store_raw != "none"))

            chunks = build_chunks(
                episodes, extract=cfg.extract, ingest=cfg.ingest, tz=cfg.timezone
            )
            counters["chunks"] = len(chunks)

            extraction, extract_usage = extract_all(
                backend, chunks, cfg.extract, max_calls=cfg.consolidate.max_calls
            )
            usage["extract"] = extract_usage
            candidates: list[Candidate] = [c for res in extraction for c in res.candidates]
            counters["candidates"] = len(candidates)
            counters["rejected_extract"] = sum(len(res.rejected) for res in extraction)
            counters["parse_errors"] = int(extract_usage.get("parse_errors", 0))

            accepted: list[Candidate] = candidates
            if cfg.extract.verify and candidates:
                accepted, _, verify_usage = verify_all(
                    backend,
                    candidates,
                    {chunk.index: chunk for chunk in chunks},
                    cfg.extract,
                    max_calls=max(60, len(candidates)),
                )
                usage["verify"] = verify_usage
                counters["verified"] = len(accepted)
                counters["rejected_verify"] = int(verify_usage.get("rejected", 0))
            else:
                counters["verified"] = len(candidates)

            if time.monotonic() - t0 > cfg.budgets.max_minutes * 60:
                notes.append("run budget exhausted before reconciliation; remaining candidates dropped")
                accepted = []

            retriever = build_retriever(cfg, store, backend_name=backend.name)
            before = snapshot(store)
            try:
                decisions, applied, reconcile_usage = reconcile_and_apply(
                    store,
                    backend,
                    accepted,
                    retriever,
                    cfg.consolidate,
                    run_id=run_id,
                    model=backend.model,
                )
            except LossGuardError as exc:
                result.status = "blocked"
                result.error = f"loss guard: {exc}"
                result.loss = exc.loss.as_dict() if hasattr(exc.loss, "as_dict") else None
                counters["duration_s"] = round(time.monotonic() - t0, 1)
                result.counters = counters
                result.usage = usage
                store.finish_run(run_id, status="blocked", counters=counters, usage=usage, error=result.error)
                notes.append("ledger transaction rolled back by the loss guard")
                return result

            usage["reconcile"] = reconcile_usage
            counters["decisions"] = len(decisions)
            result.decisions = decisions
            result.loss = applied.loss
            compaction = consolidate_active(store)
            counters["consolidated"] = int(compaction.get("archived", 0))
            if retriever is not None:
                retriever.sync()
            counters["applied"] = len(applied.memory_ids)
            counters["deferred"] = len(applied.deferred)
            counters["proposals"] = len(applied.proposals)
            after = snapshot(store)
            log.info(
                "applied %s memories, deferred %s, loss %s→%s",
                len(applied.memory_ids),
                len(applied.deferred),
                before["entries"],
                after["entries"],
            )

            promoted = []
            changed = []
            for entry in applied.applied:
                if entry.get("op") in {"ADD", "UPDATE", "SUPERSEDE", "MERGE"}:
                    memory = store.get_memory(str(entry.get("memory_id", ""))) or {}
                    item = {
                        "memory_id": entry.get("memory_id"),
                        "text": entry.get("text"),
                        "section": memory.get("section", ""),
                        "kind": memory.get("kind", ""),
                        "evidence": [
                            dict(row)
                            for row in store.conn.execute(
                                "SELECT episode_id, quote FROM evidence WHERE memory_id=?",
                                (entry.get("memory_id"),),
                            ).fetchall()
                        ],
                    }
                    promoted.append(item)
                    if entry.get("targets"):
                        changed.append(
                            {"op": entry.get("op"), "targets": entry.get("targets"), "text": entry.get("text")}
                        )

            need_review = []
            for pid in applied.proposals:
                proposal = store.get_proposal(pid) or {}
                payload = proposal.get("payload", {})
                need_review.append(
                    {
                        "proposal_id": pid,
                        "text": payload.get("text", ""),
                        "reason": payload.get("decision", {}).get("rationale", ""),
                    }
                )

            memory_limit, user_limit = cfg.hermes.memory_char_limit, cfg.hermes.user_char_limit
            try:
                hermes = HermesAdapter(cfg)
                if hermes.inspect().available:
                    memory_limit, user_limit = hermes.read_memory_limits()
            except Exception:  # noqa: BLE001
                pass

            if dry_run:
                bundle = build_bundle(
                    store,
                    memory_path=Path(cfg.hermes.home) / cfg.hermes.memory_file,
                    user_path=Path(cfg.hermes.home) / cfg.hermes.user_file,
                    memory_limit=memory_limit,
                    user_limit=user_limit,
                )
                export_paths: dict[str, str] = {}
            else:
                bundle, export_paths = build_and_write(
                    store, cfg, memory_limit=memory_limit, user_limit=user_limit
                )
            result.bundle = bundle.as_review()
            result.export_paths = export_paths
            pressure = bundle.pressure or {}
            if any(float((value or {}).get("fill", 0)) >= 0.85 for value in pressure.values()) or bundle.skipped:
                notes.append(
                    "memory pressure: "
                    + ", ".join(
                        f"{target} {value.get('chars', 0)}/{value.get('limit', 0)}"
                        for target, value in pressure.items()
                    )
                )

            reports: list[ApplyReport] = []
            if apply_hermes and not dry_run:
                if cfg.hermes.write_mode != "apply":
                    notes.append(
                        "apply-hermes requested but hermes.write_mode=proposals; nothing written to Hermes"
                    )
                else:
                    from .apply import apply_review_actions

                    reports = apply_review_actions(
                        cfg, bundle.actions, memory_limit=memory_limit, user_limit=user_limit
                    )
            result.reports = [r.as_dict() for r in reports]

            section = render_section(
                started_at=started.astimezone().strftime("%Y-%m-%d %H:%M"),
                window=window_label,
                run_id=run_id,
                model=backend.model,
                counts={
                    **counters,
                    "duration_s": round(time.monotonic() - t0, 1),
                },
                promoted=promoted,
                changed=changed,
                needs_review=need_review,
                skipped=bundle.skipped,
                safety={
                    "loss": applied.loss,
                    "filters": filt_stats.as_dict() if filt_stats else {},
                    "notes": notes + (
                        [f"compacted {counters['consolidated']} duplicate active memories"]
                        if counters.get("consolidated")
                        else []
                    ),
                },
            )
            if not dry_run:
                dream_path = append_section(
                    cfg.synth.diary_path, section, keep_sections=cfg.synth.keep_sections
                )
                result.dream_path = str(dream_path)

            counters["duration_s"] = round(time.monotonic() - t0, 1)
            result.counters = counters
            result.usage = usage
            result.status = "ok"
            store.finish_run(run_id, status="ok", counters=counters, usage=usage)
            return result
    except Exception as exc:  # noqa: BLE001
        counters["duration_s"] = round(time.monotonic() - t0, 1)
        result.status = "failed"
        result.error = str(exc)
        result.counters = counters
        result.usage = usage
        try:
            store.finish_run(run_id, status="failed", counters=counters, usage=usage, error=str(exc))
        except Exception:  # noqa: BLE001
            pass
        log.exception("run failed")
        raise
    finally:
        if retriever is not None:
            retriever.close()
        store.close()
        if temp_db is not None:
            for suffix in ("", "-wal", "-shm"):
                try:
                    Path(str(temp_db) + suffix).unlink(missing_ok=True)
                except OSError:
                    pass
