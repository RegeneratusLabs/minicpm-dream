"""MCP tool definitions and dispatch for the turn-time memory surface."""

from __future__ import annotations

from typing import Any

from .. import __version__
from ..recall import RecallService
from .protocol import INVALID_PARAMS, RpcError

TOOLS: list[dict[str, Any]] = [
    {
        "name": "recall",
        "description": (
            "Look up durable memories about the user, their projects and their open loops. "
            "Call this before answering questions about preferences, history, names, dates, "
            "decisions or unfinished work. Every hit carries the verbatim quote it came from "
            "and a caveat when it is stale, contested or superseded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you need to remember."},
                "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 6},
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to identity, preferences, projects, people, "
                    "environment, procedures, lessons or open_loops.",
                },
                "include_superseded": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also return facts that were true earlier and what replaced them.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Store one durable fact, preference or decision immediately, without waiting for the "
            "nightly pass. Pass the user's verbatim words as `quote` when you have them so the "
            "memory keeps a receipt. Corrections supersede the old entry instead of deleting it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "One short atomic statement."},
                "kind": {
                    "type": "string",
                    "enum": ["fact", "event", "preference", "task", "decision", "self", "question"],
                    "default": "fact",
                },
                "section": {
                    "type": "string",
                    "enum": [
                        "identity",
                        "preferences",
                        "projects",
                        "people",
                        "environment",
                        "procedures",
                        "lessons",
                        "open_loops",
                    ],
                    "default": "projects",
                },
                "quote": {"type": "string", "description": "The verbatim source quote, if any."},
                "episode_id": {
                    "type": "string",
                    "description": "Episode id the quote came from, when known.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "explain",
        "description": "Show one memory with its receipts, version history and supersession chain.",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string"}},
            "required": ["memory_id"],
        },
    },
    {
        "name": "forget",
        "description": (
            "Tombstone one memory because the user asked for it to be forgotten. This keeps a "
            "receipt-free tombstone and deletes the quotes; it cannot be undone."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true; only call this when the user explicitly asked.",
                },
            },
            "required": ["memory_id", "confirm"],
        },
    },
    {
        "name": "memory_status",
        "description": "How much memory exists, whether retrieval is hybrid, and when the last pass ran.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _text(content: str, structured: dict[str, Any] | None = None, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": content}]}
    if structured is not None:
        result["structuredContent"] = structured
    if is_error:
        result["isError"] = True
    return result


def _require(args: dict[str, Any], key: str, kind: type) -> Any:
    value = args.get(key)
    if not isinstance(value, kind):
        raise RpcError(INVALID_PARAMS, f"`{key}` must be a {kind.__name__}")
    return value


def call_tool(service: RecallService, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}
    if not isinstance(args, dict):
        raise RpcError(INVALID_PARAMS, "`arguments` must be an object")

    if name == "recall":
        query = _require(args, "query", str)
        k = int(args.get("k") or 6)
        sections = args.get("sections")
        if sections is not None and not isinstance(sections, list):
            raise RpcError(INVALID_PARAMS, "`sections` must be an array of strings")
        recalled = service.recall(
            query,
            k=max(1, min(20, k)),
            sections=[str(s) for s in sections] if sections else None,
            include_superseded=bool(args.get("include_superseded", False)),
        )
        return _text(recalled.render(), recalled.as_dict())

    if name == "remember":
        text = _require(args, "text", str)
        stored = service.remember(
            text,
            kind=str(args.get("kind") or "fact"),
            section=str(args.get("section") or "projects"),
            quote=str(args.get("quote") or ""),
            episode_id=str(args.get("episode_id") or ""),
        )
        if not stored.get("ok"):
            return _text(
                "Not stored: " + "; ".join(stored.get("errors", ["invalid memory"])),
                stored,
                is_error=True,
            )
        verb = {
            "ADD": "Stored",
            "UPDATE": "Updated (the previous entry is kept in history)",
            "NOOP": "Already stored",
        }.get(str(stored.get("op")), "Stored")
        receipt = " (with a checked quote)" if stored.get("verified") else ""
        return _text(f"{verb}: {stored.get('text')}{receipt}", stored)

    if name == "explain":
        memory_id = _require(args, "memory_id", str)
        payload = service.explain(memory_id)
        if payload is None:
            return _text(f"No memory {memory_id}.", None, is_error=True)
        lines = [f"[{memory_id}] {payload['memory'].get('text')}"]
        for receipt in payload["receipts"]:
            lines.append(f"  ↳ {receipt.get('spoken_at') or ''} \"{receipt.get('quote')}\"")
        for entry in payload["supersedes_chain"]:
            lines.append(f"  ← supersedes {entry['memory_id']}: {entry['text']} ({entry['status']})")
        if payload["flags"]:
            lines.append(f"  flags: {', '.join(payload['flags'])}")
        return _text("\n".join(lines), payload)

    if name == "forget":
        memory_id = _require(args, "memory_id", str)
        if args.get("confirm") is not True:
            return _text(
                "Refusing to forget without confirm=true; ask the user first.",
                None,
                is_error=True,
            )
        ok = service.store.forget(memory_id)
        return _text(
            f"Forgotten: {memory_id} (tombstone kept, quotes deleted)." if ok else f"No memory {memory_id}.",
            {"forgotten": ok, "memory_id": memory_id},
            is_error=not ok,
        )

    if name == "memory_status":
        payload = service.status()
        counts = payload["memories"]
        text = (
            f"{counts.get('active', 0)} active memories ({counts.get('active_chars', 0)} chars), "
            f"retrieval={'hybrid' if payload['hybrid'] else 'fts5'}, "
            f"last run={((payload['last_run'] or {}).get('finished_at') or 'never')}."
        )
        return _text(text, payload)

    raise RpcError(INVALID_PARAMS, f"unknown tool {name!r}")


def server_info() -> dict[str, Any]:
    return {"name": "minicpm-dream", "version": __version__}
