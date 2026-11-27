"""MCP stdio server: expose the ledger to any agent that speaks MCP."""

from __future__ import annotations

import sys
import traceback
from typing import Any, TextIO

from ..config import Config
from ..recall import DEFAULT_STALE_DAYS, RecallService
from ..util.logging import get_logger
from .protocol import (
    DEFAULT_PROTOCOL_VERSION,
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    SUPPORTED_PROTOCOL_VERSIONS,
    RpcError,
    is_notification,
    read_message,
    result_response,
    validate_request,
    write_message,
)
from .tools import TOOLS, call_tool, server_info

log = get_logger("serve")


def build_recall_service(cfg: Config, *, hybrid: bool = False, stale_days: int = DEFAULT_STALE_DAYS) -> RecallService:
    """Fast by default: FTS5 only. Hybrid loads the local embedding model on demand."""
    service = RecallService(cfg, stale_days=stale_days)
    if not hybrid:
        return service
    try:
        from ..retrieval.openbmb import build_embedder
        from ..retrieval.vector_index import VectorIndex

        embedder = build_embedder(cfg.retrieval)
        index = VectorIndex(
            service.store,
            embedder,
            model=cfg.retrieval.embedding_model,
            batch_size=cfg.retrieval.embedding_batch_size,
        )
        index.sync()
        service.embedder = embedder
        service.index = index
    except Exception as exc:  # noqa: BLE001 - the server must start without the optional stack
        log.warning("hybrid recall unavailable (%s); using FTS5", exc)
    return service


def _initialize(params: dict[str, Any]) -> dict[str, Any]:
    requested = str(params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION)
    version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
    return {
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": server_info(),
        "instructions": (
            "Private local memory for this user. Call `recall` before answering questions about "
            "their preferences, projects, people, dates or unfinished work; cite the returned "
            "receipts. Call `remember` when the user states something durable. Hits flagged "
            "stale/contested/superseded must not be stated as current fact."
        ),
    }


def handle_message(
    service: RecallService,
    message: dict[str, Any],
    *,
    max_tools: int | None = None,
) -> dict[str, Any] | None:
    request_id, method = validate_request(message)
    params = message.get("params") or {}
    if not isinstance(params, dict):
        raise RpcError(-32602, "`params` must be an object")

    if method == "initialize":
        return result_response(request_id, _initialize(params))
    if method == "ping":
        return result_response(request_id, {})
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return None
    if method.startswith("notifications/"):
        return None
    if method == "tools/list":
        tools = TOOLS if max_tools is None else TOOLS[:max_tools]
        return result_response(request_id, {"tools": tools})
    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str):
            raise RpcError(-32602, "`name` must be a string")
        arguments = params.get("arguments") or {}
        return result_response(request_id, call_tool(service, name, arguments))
    if method in {"shutdown", "exit"}:
        return result_response(request_id, {})
    raise RpcError(METHOD_NOT_FOUND, f"unknown method {method!r}")


def serve_stdio(
    cfg: Config,
    *,
    hybrid: bool = False,
    stale_days: int = DEFAULT_STALE_DAYS,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run the MCP loop until stdin closes. Returns a process exit code."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    service = build_recall_service(cfg, hybrid=hybrid, stale_days=stale_days)
    log.info("mcp server ready (%s)", "hybrid" if service.hybrid else "fts5")
    try:
        while True:
            try:
                message = read_message(stdin)
            except RpcError as exc:
                write_message(stdout, exc.as_dict(None))
                continue
            if message is None:
                break
            if is_notification(message):
                # JSON-RPC notifications never get a response.
                continue
            request_id = message.get("id")
            try:
                response = handle_message(service, message)
            except RpcError as exc:
                response = exc.as_dict(request_id)
            except Exception as exc:  # noqa: BLE001 - never kill the server on one tool call
                log.error("unhandled error: %s", traceback.format_exc())
                response = RpcError(INTERNAL_ERROR, f"{type(exc).__name__}: {exc}").as_dict(request_id)
            if response is not None:
                write_message(stdout, response)
            if message.get("method") in {"shutdown", "exit"}:
                break
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
    return 0
