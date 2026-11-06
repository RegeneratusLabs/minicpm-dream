#!/usr/bin/env python
"""Compare the local MiniCPM stack with a cloud model on the same eval suite.

Example:
    HF_HOME=./data/hf-cache .venv/bin/python scripts/compare_models.py \
      --suite evals/synthetic-days/full-day

The cloud key is read from an environment variable (default DEEPSEEK_API_KEY),
optionally from ~/.hermes/.env. Keys are never printed. The comparison itself
sends the suite's (synthetic or redacted) text to the cloud endpoint; do not
point this at private data without deciding that is acceptable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from minicpm_dream.backends.llama_cpp import LlamaCppBackend
from minicpm_dream.backends.openai_compat import OpenAICompatBackend
from minicpm_dream.config import load_config
from minicpm_dream.eval.harness import run_suite
from minicpm_dream.util.text import containment, jaccard


def load_env_file(path: str | None) -> None:
    if not path:
        return
    p = Path(path).expanduser()
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def _overlap(a: list[dict], b: list[dict], threshold: float = 0.6) -> dict[str, int]:
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            if j in matched_b:
                continue
            score = max(
                jaccard(str(left.get("text", "")), str(right.get("text", ""))),
                containment(str(left.get("text", "")), str(right.get("text", ""))),
            )
            if score >= threshold:
                matched_a.add(i)
                matched_b.add(j)
                break
    return {
        "matched": len(matched_a),
        "local_only": len(a) - len(matched_a),
        "cloud_only": len(b) - len(matched_b),
    }


def summarize(report: dict, other: dict | None = None) -> dict:
    metrics = report["metrics"]
    out = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "schema_valid": metrics["schema_valid"],
        "temporal": metrics["temporal_accuracy"],
        "safety": metrics["safety_violations"],
        "seconds": metrics["duration_s"],
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "completion_tokens": metrics.get("completion_tokens", 0),
    }
    if other is not None:
        cases = []
        for left, right in zip(report["per_case"], other["per_case"], strict=False):
            cases.append(
                {
                    "case": left["case"],
                    "local_memories": len(left.get("memories", [])),
                    "cloud_memories": len(right.get("memories", [])),
                    **_overlap(left.get("memories", []), right.get("memories", [])),
                }
            )
        out["agreement"] = cases
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="evals/synthetic-days/full-day")
    parser.add_argument("--env-file", default="~/.hermes/.env")
    parser.add_argument("--local-model-path", default="data/models/MiniCPM5-2B-Q4_K_M.gguf")
    parser.add_argument("--local-server", default=os.environ.get("LLAMA_SERVER", "llama-server"))
    parser.add_argument("--local-threads", type=int, default=4)
    parser.add_argument("--chunk-tokens", type=int, default=350)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--cloud-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--cloud-model", default="deepseek-flash")
    parser.add_argument("--cloud-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--cloud-reasoning-effort", default="none")
    parser.add_argument("--out", default="compare-report.json")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--skip-cloud", action="store_true")
    args = parser.parse_args(argv)

    load_env_file(args.env_file)
    cfg = load_config(Path("~/.config/minicpm-dream/compare.toml"))
    cfg.retrieval.embeddings = "off"
    cfg.retrieval.reranker = "off"
    cfg.extract.max_chunk_tokens = args.chunk_tokens
    cfg.extract.max_items = args.max_items
    cfg.extract.coverage_pass = args.coverage
    cfg.extract.message_fallback = not args.no_fallback
    if args.no_verify:
        cfg.extract.verify = False

    reports: dict[str, dict] = {}
    errors: dict[str, str] = {}

    local = None
    if not args.skip_local:
        try:
            local = LlamaCppBackend(
                model_path=args.local_model_path,
                llama_server=args.local_server,
                context=8192,
                threads=args.local_threads,
                temperature=0.2,
                max_tokens=1200,
                log_path="data/logs/compare-local.log",
            )
            local.start()
            reports["local"] = run_suite(cfg, local, args.suite)
        except Exception as exc:  # noqa: BLE001
            errors["local"] = str(exc)
        finally:
            if local is not None:
                local.close()

    cloud = None
    if not args.skip_cloud:
        key = os.environ.get(args.cloud_key_env)
        if not key:
            errors["cloud"] = f"{args.cloud_key_env} is not set"
        else:
            try:
                cloud = OpenAICompatBackend(
                    base_url=args.cloud_base_url,
                    model=args.cloud_model,
                    api_key=key,
                    timeout_s=180,
                    default_temperature=0.2,
                    default_max_tokens=1200,
                    disable_thinking=False,
                    structured_mode="auto",
                    reasoning_effort=args.cloud_reasoning_effort,
                )
                reports["cloud"] = run_suite(cfg, cloud, args.suite)
            except Exception as exc:  # noqa: BLE001
                errors["cloud"] = str(exc)
            finally:
                if cloud is not None:
                    cloud.close()

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": args.suite,
        "cloud_model": args.cloud_model,
        "local_model": Path(args.local_model_path).name,
        "summary": {
            name: summarize(report, reports.get("cloud") if name == "local" else reports.get("local"))
            for name, report in reports.items()
        },
        "errors": errors,
        "reports": reports,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        out.chmod(0o600)
    except OSError:
        pass

    print(f"suite: {args.suite}")
    print(f"{'backend':8s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'schema':>7s} {'temporal':>8s} {'safety':>7s} {'sec':>8s}")
    for name, report in reports.items():
        m = report["metrics"]
        print(
            f"{name:8s} {m['precision']:9.2f} {m['recall']:7.2f} {m['f1']:6.2f} "
            f"{m['schema_valid']:7.2f} {m['temporal_accuracy']:8.2f} {m['safety_violations']:7d} "
            f"{m['duration_s']:8.1f}"
        )
    for name, error in errors.items():
        print(f"{name}: ERROR {error}", file=sys.stderr)
    print(f"report: {out}")
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
