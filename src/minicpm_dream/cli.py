"""Command-line interface for minicpm-dream."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .apply import apply_review_actions, build_and_write
from .backends import BackendUnavailable
from .backends.factory import make_backend
from .config import Config, load_config
from .ingest import HermesAdapter
from .models import MODELS, download_model, get_spec, human_size, model_status, resolve_model_path
from .pipeline import collect_episodes, run_pass
from .retrieval.factory import build_retriever
from .retrieval.openbmb import cache_status, download_models
from .store.db import Store
from .util.atomic import ensure_dir, write_text_atomic
from .util.logging import setup_logging
from .util.timeutil import (
    day_window,
    iso_utc,
    parse_timestamp,
    previous_day,
    utc_now,
    week_window,
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("--data-dir", help="override the data directory")
    parser.add_argument("--hermes-home", help="override the Hermes home (memories, state.db)")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def _load(args: argparse.Namespace, *, backend: str | None = None, model: str | None = None,
          model_path: str | None = None) -> Config:
    cfg = load_config(getattr(args, "config", None), data_dir=getattr(args, "data_dir", None))
    if backend:
        cfg.model.backend = backend
    if model:
        cfg.model.name = model
    if model_path:
        cfg.model.model_path = str(Path(model_path).expanduser())
    if getattr(args, "hermes_home", None):
        cfg.hermes.home = str(Path(args.hermes_home).expanduser())
        cfg.resolve()
    setup_logging(getattr(args, "verbose", False), Path(cfg.paths.logs) / "minicpm-dream.log")
    return cfg


def _apply_mode(cfg: Config, args: argparse.Namespace) -> None:
    if getattr(args, "quality", False):
        cfg.extract.max_chunk_tokens = 350
        cfg.extract.max_items = 8
        cfg.extract.message_fallback = True
        cfg.extract.coverage_pass = False
        cfg.extract.profile_pass = False
        cfg.extract.verify_lenient = True
    elif getattr(args, "fast", False):
        cfg.extract.max_chunk_tokens = 3000
        cfg.extract.message_fallback = False
        cfg.extract.coverage_pass = False
        cfg.extract.profile_pass = False


def _window(args: argparse.Namespace, cfg: Config) -> tuple[datetime, datetime, str]:
    if getattr(args, "since", None) or getattr(args, "until", None):
        if not (args.since and args.until):
            raise ValueError("--since and --until must be provided together")
        since = parse_timestamp(args.since, cfg.timezone)
        until = parse_timestamp(args.until, cfg.timezone)
        label = f"{iso_utc(since)} → {iso_utc(until)}"
        return since, until, label
    day = previous_day(cfg.timezone)
    if getattr(args, "window", "day") == "week":
        since, until = week_window(day, cfg.timezone)
        return since, until, f"weekly window ending {day.isoformat()}"
    since, until = day_window(day, cfg.timezone)
    return since, until, f"daily window {day.isoformat()}"


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
        return
    for line in _human(obj):
        print(line)


def _human(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        lines = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{key}: {json.dumps(value, ensure_ascii=False, default=str)}")
            else:
                lines.append(f"{key}: {value}")
        return lines
    if isinstance(obj, list):
        return [json.dumps(item, ensure_ascii=False, default=str) for item in obj]
    return [str(obj)]


# -- commands ---------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _load(args)
    checks: list[tuple[str, bool, str]] = []

    checks.append(("python", sys.version_info >= (3, 11), sys.version.split()[0]))
    fts_ok = True
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        con.close()
    except sqlite3.OperationalError as exc:
        fts_ok = False
        checks.append(("sqlite fts5", False, str(exc)))
    if fts_ok:
        checks.append(("sqlite fts5", True, sqlite3.sqlite_version))

    data_dir = Path(cfg.paths.data_dir)
    try:
        ensure_dir(data_dir)
        probe = data_dir / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(("data dir", True, str(data_dir)))
    except OSError as exc:
        checks.append(("data dir", False, f"{data_dir}: {exc}"))

    status = model_status(cfg)
    checks.append(
        ("model", bool(status.get("ok")), f"{status.get('key')} @ {status.get('path')}")
    )
    if cfg.model.backend == "llama_cpp":
        binary = cfg.model.llama_server or shutil.which("llama-server") or ""
        checks.append(
            ("llama-server", bool(binary), binary or "not found (setup docs: docs/operations.md)")
        )
    else:
        checks.append(("backend", True, cfg.model.backend))

    if cfg.retrieval.embeddings == "openbmb" or cfg.retrieval.reranker == "openbmb":
        import importlib.util as _ilu

        deps = all(_ilu.find_spec(name) is not None for name in ("torch", "transformers", "numpy"))
        checks.append(("openbmb deps", deps, "torch + transformers + numpy" if deps else "install .[openbmb]"))
        if deps:
            embedding_cache = cache_status(cfg.retrieval.embedding_model)
            reranker_cache = cache_status(cfg.retrieval.reranker_model)
            checks.append(
                ("embedding model", embedding_cache not in {"not cached", "unknown"}, embedding_cache)
            )
            checks.append(
                ("reranker model", reranker_cache not in {"not cached", "unknown"}, reranker_cache)
            )

    hermes = HermesAdapter(cfg)
    info = hermes.inspect()
    checks.append(("hermes state.db", info.available, info.note or str(info.path)))
    memory_path = Path(cfg.hermes.home) / cfg.hermes.memory_file
    user_path = Path(cfg.hermes.home) / cfg.hermes.user_file
    checks.append(("Hermes MEMORY.md", memory_path.is_file(), str(memory_path)))
    checks.append(("Hermes USER.md", user_path.is_file(), str(user_path)))

    if args.json:
        print(
            json.dumps(
                {
                    "ok": all(ok for _, ok, _ in checks if _ != "model"),
                    "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
                    "config": cfg.config_path or "(defaults)",
                },
                indent=2,
            )
        )
    else:
        print(f"minicpm-dream {__version__}  (config: {cfg.config_path or 'defaults'})")
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '!'} {name:18s} {detail}")
        print()
        print("Next: minicpm-dream setup --download   then   minicpm-dream run --dry-run")
    core = {"python", "sqlite fts5", "data dir"}
    return 0 if all(ok for name, ok, _ in checks if name in core) else 1


def cmd_setup(args: argparse.Namespace) -> int:
    cfg = _load(args, model=args.model, model_path=args.model_path)
    if args.list_models:
        for key, spec in MODELS.items():
            print(f"{key:20s} {spec.filename:32s} {human_size(spec.size):>9s}  {spec.notes}")
        return 0
    try:
        spec = get_spec(cfg.model.name)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    path = resolve_model_path(cfg)
    if args.check:
        status = model_status(cfg)
        print(json.dumps(status, indent=2) if args.json else _human(status)[0])
        if cfg.retrieval.embeddings == "openbmb":
            print(f"embedding: {cache_status(cfg.retrieval.embedding_model)}")
            print(f"reranker:  {cache_status(cfg.retrieval.reranker_model)}")
        return 0 if status.get("ok") else 1
    if args.download_embeddings:
        try:
            paths = download_models(
                cfg.retrieval.embedding_model, cfg.retrieval.reranker_model
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for repo, repo_path in paths.items():
            print(f"{repo}: {repo_path}")
        return 0
    if not args.download:
        print(f"model: {spec.key}")
        print(f"path:  {path}")
        print(f"size:  {human_size(spec.size)}  ({spec.license})")
        print("Run with --download to fetch it, or pass --model-path for an offline copy.")
        return 0
    token = os.environ.get("HF_TOKEN")
    try:
        saved = download_model(
            spec, path, verify=not args.skip_sha, token=token, force=args.force
        )
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"model ready: {saved} ({human_size(saved.stat().st_size)})")
    if not (cfg.model.llama_server or shutil.which("llama-server")):
        print("! llama-server not found. Install llama.cpp and put it on PATH, or set model.llama_server.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _load(args)
    since, until, label = _window(args, cfg)
    episodes, stats = collect_episodes(
        cfg, source=args.source, since=since, until=until, input_path=args.input, limit=args.limit
    )
    payload = {
        "window": label,
        "source": args.source,
        "episodes": len(episodes),
        "filters": stats.as_dict(),
    }
    if args.out:
        text = "\n".join(ep.to_json() for ep in episodes) + ("\n" if episodes else "")
        if args.out == "-":
            sys.stdout.write(text)
        else:
            write_text_atomic(args.out, text)
            payload["out"] = str(Path(args.out).expanduser())
    _print(payload, args.json)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args, backend=args.backend, model=args.model, model_path=args.model_path)
    _apply_mode(cfg, args)
    if getattr(args, "max_minutes", None):
        cfg.budgets.max_minutes = args.max_minutes
    since, until, label = _window(args, cfg)
    backend = make_backend(cfg, mock_script=args.mock_script)
    try:
        result = run_pass(
            cfg,
            backend,
            source=args.source,
            since=since,
            until=until,
            window_label=label,
            input_path=args.input,
            limit=args.limit,
            dry_run=args.dry_run,
            apply_hermes=args.apply_hermes,
        )
    finally:
        backend.close()
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        c = result.counters
        print(
            f"run {result.run_id} {result.status}: {c.get('episodes_kept', 0)} episodes → "
            f"{c.get('candidates', 0)} candidates → {c.get('verified', 0)} verified → "
            f"{c.get('applied', 0)} changes ({c.get('deferred', 0)} deferred)"
        )
        if result.dream_path:
            print(f"diary:   {result.dream_path}")
        if result.export_paths:
            print(f"exports: {result.export_paths.get('review', '')}")
        if result.error:
            print(f"error:   {result.error}")
        if args.dry_run:
            print("dry run: no ledger, diary or Hermes files were written")
    return 0 if result.status in {"ok", "empty"} else 1


def cmd_review(args: argparse.Namespace) -> int:
    cfg = _load(args)
    with Store(cfg.paths.db) as store:
        if args.action == "list":
            pending = store.list_proposals("pending")
            counts = store.proposal_counts()
            out: dict[str, Any] = {
                "counts": counts,
                "pending": [
                    {
                        "id": p["id"],
                        "kind": p["kind"],
                        "target": p["target"],
                        "text": p["payload"].get("text", ""),
                        "decision": p["payload"].get("decision", {}),
                    }
                    for p in pending
                ],
            }
            if args.json:
                print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
            else:
                print(f"proposals: {counts}")
                for item in out["pending"]:
                    decision = item["decision"]
                    print(
                        f"  {item['id']}  {item['kind']}/{item['target']}  "
                        f"[{decision.get('op', '?')}] {item['text']}"
                    )
                if not out["pending"]:
                    print("  (none)")
            return 0
        if args.action == "show":
            proposal = store.get_proposal(args.id)
            if not proposal:
                print(f"error: no proposal {args.id}", file=sys.stderr)
                return 1
            print(json.dumps(proposal, indent=2, ensure_ascii=False, default=str))
            return 0
        if args.action == "approve":
            proposal = store.get_proposal(args.id)
            if not proposal:
                print(f"error: no proposal {args.id}", file=sys.stderr)
                return 1
            store.set_proposal_status(args.id, "approved", reviewer="user", reason=args.reason)
            action = proposal["payload"].get("hermes_action")
            if action:
                reports = apply_review_actions(cfg, [action])
                print(json.dumps([r.as_dict() for r in reports], indent=2))
            else:
                print(f"approved {args.id}; no direct Hermes action attached")
            return 0
        if args.action == "reject":
            proposal = store.get_proposal(args.id)
            if not proposal:
                print(f"error: no proposal {args.id}", file=sys.stderr)
                return 1
            store.set_proposal_status(args.id, "rejected", reviewer="user", reason=args.reason)
            until = utc_now() + timedelta(days=180)
            store.set_cooldown(f"proposal:{args.id}", "reject", until)
            store.conn.execute(
                "INSERT OR REPLACE INTO rejects(key, memory_id, text, reason, created_at, reask_after) "
                "VALUES (?,?,?,?,?,?)",
                (
                    f"proposal:{args.id}",
                    proposal["payload"].get("candidate_id"),
                    proposal["payload"].get("text", ""),
                    args.reason or "",
                    iso_utc(utc_now()),
                    iso_utc(until),
                ),
            )
            print(f"rejected {args.id}")
            return 0
    return 2


def cmd_export(args: argparse.Namespace) -> int:
    cfg = _load(args)
    with Store(cfg.paths.db) as store:
        memory_limit, user_limit = cfg.hermes.memory_char_limit, cfg.hermes.user_char_limit
        hermes = HermesAdapter(cfg)
        if hermes.inspect().available:
            memory_limit, user_limit = hermes.read_memory_limits()
        bundle, paths = build_and_write(
            store, cfg, memory_limit=memory_limit, user_limit=user_limit
        )
        payload: dict[str, Any] = {
            "paths": paths,
            "actions": len(bundle.actions),
            "memory_chars": len(bundle.memory_text),
            "user_chars": len(bundle.user_text),
            "limits": bundle.limits,
            "skipped": bundle.skipped,
            "pressure": bundle.pressure,
        }
        if args.apply_hermes:
            if cfg.hermes.write_mode != "apply":
                payload["apply"] = "refused: hermes.write_mode=proposals"
            else:
                reports = apply_review_actions(
                    cfg, bundle.actions, memory_limit=memory_limit, user_limit=user_limit
                )
                payload["apply"] = [r.as_dict() for r in reports]
        _print(payload, args.json)
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    cfg = _load(args)
    pending = 0
    blocked = 0
    actions = 0
    pressure = False
    if not Path(cfg.paths.db).is_file():
        print(
            json.dumps(
                {
                    "wakeAgent": False,
                    "context": {"reason": "no ledger yet", "review_actions": 0},
                }
            )
        )
        return 0
    try:
        with Store(cfg.paths.db) as store:
            pending = int(store.proposal_counts().get("pending", 0))
            blocked = int(
                store.conn.execute(
                    "SELECT COUNT(*) FROM runs WHERE status='blocked' AND started_at > ?",
                    (iso_utc(utc_now() - timedelta(days=1)),),
                ).fetchone()[0]
            )
    except Exception:  # noqa: BLE001
        pending = 0
    review = cfg.exports_dir / "review.json"
    if review.is_file():
        try:
            payload = json.loads(review.read_text(encoding="utf-8"))
            actions = len(payload.get("actions", []))
            pressure = bool(payload.get("skipped")) or any(
                float((value or {}).get("fill", 0)) >= 0.85
                for value in (payload.get("pressure") or {}).values()
            )
        except (OSError, json.JSONDecodeError):
            actions = 0
    wake = pending > 0 or actions > 0 or blocked > 0 or pressure
    payload = {
        "wakeAgent": wake,
        "context": {
            "pending_proposals": pending,
            "review_actions": actions,
            "memory_pressure": pressure,
            "blocked_runs_24h": blocked,
            "review_path": str(review),
        },
    }
    print(json.dumps(payload))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg = _load(args)
    with Store(cfg.paths.db) as store:
        if args.include_superseded:
            like = f"%{args.query}%"
            rows = [
                dict(row)
                for row in store.conn.execute(
                    "SELECT * FROM memories WHERE text LIKE ? OR subject LIKE ? "
                    "ORDER BY recorded_at DESC LIMIT ?",
                    (like, like, args.k),
                ).fetchall()
            ]
        else:
            rows = store.search(args.query, limit=args.k)
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
        else:
            for row in rows:
                print(f"{row['id']}  [{row['status']}] {row['section']} — {row['text']}")
            if not rows:
                print("(no matches)")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    cfg = _load(args)
    with Store(cfg.paths.db) as store:
        memory = store.get_memory(args.id)
        if not memory:
            print(f"error: no memory {args.id}", file=sys.stderr)
            return 1
        evidence = [
            dict(row)
            for row in store.conn.execute(
                "SELECT episode_id, quote, verified FROM evidence WHERE memory_id=?", (args.id,)
            ).fetchall()
        ]
        ops = [
            dict(row)
            for row in store.conn.execute(
                "SELECT id, run_id, ts, op, decision, rationale FROM ops WHERE payload_json LIKE ? "
                "ORDER BY ts DESC",
                (f"%{args.id}%",),
            ).fetchall()
        ]
        versions = [
            dict(row)
            for row in store.conn.execute(
                "SELECT version, text, valid_from, valid_to, created_at FROM memory_versions "
                "WHERE memory_id=? ORDER BY version",
                (args.id,),
            ).fetchall()
        ]
        print(
            json.dumps(
                {"memory": memory, "evidence": evidence, "ops": ops, "versions": versions},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if not args.yes:
        print("refusing to forget without --yes", file=sys.stderr)
        return 2
    with Store(cfg.paths.db) as store:
        ok = store.forget(args.id, reason=args.reason)
        print(f"forgotten: {args.id}" if ok else f"error: no memory {args.id}")
    return 0 if ok else 1


def cmd_maintain(args: argparse.Namespace) -> int:
    cfg = _load(args)
    with Store(cfg.paths.db) as store:
        n = store.archive_stale(older_than_days=args.archive_days, min_confidence=args.min_confidence)
        print(f"archived {n} stale memories")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    cfg = _load(args)
    with Store(cfg.paths.db) as store:
        if args.rebuild:
            store.clear_vectors(model=cfg.retrieval.embedding_model)
        retriever = build_retriever(cfg, store, backend_name="cli", sync=False)
        try:
            added = retriever.sync()
            count = retriever.index.count() if retriever.index is not None else 0
        finally:
            retriever.close()
    payload = {
        "embedding_model": cfg.retrieval.embedding_model,
        "vectors": count,
        "added": added,
        "rebuild": bool(args.rebuild),
    }
    _print(payload, args.json)
    return 0


def cmd_generate_eval(args: argparse.Namespace) -> int:
    from datetime import date as _date

    from .eval.generate import generate, write_suite

    start = _date.fromisoformat(args.start_date) if args.start_date else None
    generated = generate(
        seed=args.seed,
        days=args.days,
        facts=args.facts,
        start_date=start,
        tz=args.tz or load_config(args.config).timezone,
    )
    paths = write_suite(generated, args.out, force=args.force)
    payload = {
        "suite": str(Path(args.out).expanduser()),
        "seed": generated.seed,
        "days": generated.days,
        "episodes": len(generated.episodes),
        "labels": len(generated.labels),
        "gold_facts": len(generated.facts),
        "files": paths,
    }
    _print(payload, args.json)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.harness import run_suite

    cfg = _load(args, backend=args.backend, model=args.model, model_path=args.model_path)
    _apply_mode(cfg, args)
    if getattr(args, "chunk_tokens", None):
        cfg.extract.max_chunk_tokens = args.chunk_tokens
    if getattr(args, "max_items", None):
        cfg.extract.max_items = args.max_items
    backend = make_backend(cfg, mock_script=args.mock_script)
    try:
        report = run_suite(cfg, backend, suite=args.suite or cfg.eval.suite)
    finally:
        backend.close()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"suite: {report['suite']}")
        for key, value in report["metrics"].items():
            print(f"  {key:22s} {value}")
        print(f"  cases                 {report['cases']}")
    return 0 if report["metrics"].get("schema_valid", 0) >= 1.0 else 1


def cmd_install_gate(args: argparse.Namespace) -> int:
    cfg = _load(args)
    scripts = Path(cfg.hermes.home) / "scripts"
    target = scripts / "minicpm-dream-gate.sh"
    content = (
        "#!/usr/bin/env bash\n"
        "# Installed by minicpm-dream. Hermes runs cron scripts only from ~/.hermes/scripts/.\n"
        f'exec "{sys.executable}" -m minicpm_dream.cli gate --source hermes "$@"\n'
    )
    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        ok = current == content
        print(f"{'ok' if ok else 'outdated'}: {target}")
        return 0 if ok else 1
    ensure_dir(scripts)
    write_text_atomic(target, content, mode=0o755)
    print(f"installed: {target}")
    print(
        "Create a Hermes cron job with this script, e.g.\n"
        "  hermes cron create '0 3 * * *' 'Review the minicpm-dream proposals in "
        f"{cfg.exports_dir}/review.json' --script minicpm-dream-gate.sh"
    )
    return 0


# -- parser -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minicpm-dream",
        description="Local MiniCPM memory consolidation for agent harnesses.",
    )
    parser.add_argument("--version", action="version", version=f"minicpm-dream {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check the environment")
    _add_common(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("setup", help="inspect or download the MiniCPM model")
    _add_common(p)
    p.add_argument("--download", action="store_true")
    p.add_argument("--download-embeddings", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--model")
    p.add_argument("--model-path")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-sha", action="store_true")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("ingest", help="read a window of transcripts")
    _add_common(p)
    p.add_argument("--source", choices=["hermes", "jsonl"], default="hermes")
    p.add_argument("--input", help="JSONL path for --source jsonl")
    p.add_argument("--window", choices=["day", "week"], default="day")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--out", help="write normalized episodes here ('-' for stdout)")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("run", help="run one consolidation pass")
    _add_common(p)
    p.add_argument("--source", choices=["hermes", "jsonl"], default="hermes")
    p.add_argument("--input")
    p.add_argument("--window", choices=["day", "week"], default="day")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--limit", type=int)
    p.add_argument("--backend", choices=["llama_cpp", "openai", "mock"])
    p.add_argument("--model")
    p.add_argument("--model-path")
    p.add_argument("--mock-script")
    p.add_argument("--max-minutes", type=float)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quality", action="store_true", help="slower, all-message fallback")
    mode.add_argument("--fast", action="store_true", help="fewer calls, lower recall")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply-hermes", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("review", help="review deferred decisions")
    _add_common(p)
    p.add_argument("action", choices=["list", "show", "approve", "reject"])
    p.add_argument("id", nargs="?")
    p.add_argument("--reason")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("export", help="write Hermes candidate files and review.json")
    _add_common(p)
    p.add_argument("--apply-hermes", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("gate", help="emit a Hermes cron wakeAgent decision")
    _add_common(p)
    p.add_argument("--source", default="hermes")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("search", help="search the ledger")
    _add_common(p)
    p.add_argument("query")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--include-superseded", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("explain", help="show provenance for one memory")
    _add_common(p)
    p.add_argument("id")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("forget", help="tombstone one memory")
    _add_common(p)
    p.add_argument("id")
    p.add_argument("--reason", default="user_request")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("maintain", help="archive stale memories")
    _add_common(p)
    p.add_argument("--archive-days", type=int, default=180)
    p.add_argument("--min-confidence", type=float, default=0.7)
    p.set_defaults(func=cmd_maintain)

    p = sub.add_parser("generate-eval", help="generate a synthetic evaluation suite")
    _add_common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--facts", type=int, default=30)
    p.add_argument("--start-date")
    p.add_argument("--tz")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_generate_eval)

    p = sub.add_parser("index", help="build or refresh OpenBMB memory vectors")
    _add_common(p)
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("eval", help="run an evaluation suite")
    _add_common(p)
    p.add_argument("--suite")
    p.add_argument("--backend", choices=["llama_cpp", "openai", "mock"])
    p.add_argument("--model")
    p.add_argument("--model-path")
    p.add_argument("--mock-script")
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--max-items", type=int)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quality", action="store_true", help="slower, all-message fallback")
    mode.add_argument("--fast", action="store_true", help="fewer calls, lower recall")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("install-hermes-gate", help="install the Hermes cron gate script")
    _add_common(p)
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=cmd_install_gate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
