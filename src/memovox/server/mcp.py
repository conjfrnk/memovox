"""Agent-native MCP server over stdio (spec §8) — implemented with the standard
library (no ``mcp`` package required).

Speaks newline-delimited JSON-RPC 2.0, the MCP stdio transport. Wire it into
Claude Code / Claude Desktop to ingest a talk and immediately interrogate it
without leaving the editor. Exposed tools:
    ingest_video, search_knowledge, get_claim_provenance,
    synthesize_topic, find_contradictions
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from ..sdk import Memovox

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "ingest_video",
        "description": "Ingest a video/audio/transcript (local path or URL) into the knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Local file path or URL."},
                "source_url": {"type": "string", "description": "Canonical URL for deep links."},
                "title": {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "Ask a grounded question; returns an answer with timestamped citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "modality": {"type": "string", "enum": ["any", "speech", "visual"]},
                "video_id": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_claim_provenance",
        "description": "Resolve a claim_id to its source provenance (video, span, deep link).",
        "inputSchema": {
            "type": "object",
            "properties": {"claim_id": {"type": "string"}},
            "required": ["claim_id"],
        },
    },
    {
        "name": "synthesize_topic",
        "description": "Synthesize what the corpus says about a topic, with citations.",
        "inputSchema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
    {
        "name": "find_contradictions",
        "description": "Find cross-corpus contradictions, optionally restricted to a topic.",
        "inputSchema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
        },
    },
    {
        "name": "consolidate",
        "description": "Run cross-corpus consolidation (topic induction, contradiction/agreement "
                       "detection, consensus, dedup). Run after ingesting new videos.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "claim_timeline",
        "description": "Trace how a position/number about an entity or topic changed over time "
                       "(ordered, deep-linked evolution steps).",
        "inputSchema": {
            "type": "object",
            "properties": {"entity": {"type": "string"}, "topic": {"type": "string"}},
        },
    },
    {
        "name": "job_status",
        "description": "Resolve a background job id (e.g. from consolidate) to its "
                       "state/result/error.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
]


class McpServer:
    def __init__(self, mv: Memovox) -> None:
        self.mv = mv

    # -- JSON-RPC dispatch (unit-testable) --------------------------------

    def handle(self, request: dict) -> Optional[dict]:
        """Handle one JSON-RPC request; return a response, or None for notifications."""
        method = request.get("method")
        req_id = request.get("id")
        is_notification = "id" not in request

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "memovox", "version": _version()},
                }
            elif method in ("notifications/initialized", "initialized"):
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call_tool(request.get("params") or {})
            else:
                if is_notification:
                    return None
                return _error(req_id, -32601, f"Method not found: {method}")
        except Exception as exc:  # surface tool errors as JSON-RPC errors
            if is_notification:
                return None
            return _error(req_id, -32603, str(exc))

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _call_tool(self, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return _tool_text(f"Unknown tool: {name}", is_error=True)
        return handler(args)

    # -- tools ------------------------------------------------------------

    def _tool_ingest_video(self, args: dict) -> dict:
        report = self.mv.ingest(args["url"], source_url=args.get("source_url"), title=args.get("title"))
        return _tool_json(report.to_dict())

    def _tool_search_knowledge(self, args: dict) -> dict:
        answer = self.mv.ask(args["query"], video_id=args.get("video_id"),
                             modality=args.get("modality", "any"))
        return _tool_json(answer.to_dict())

    def _tool_get_claim_provenance(self, args: dict) -> dict:
        prov = self.mv.get_provenance(args["claim_id"])
        if prov is None:
            return _tool_text(f"No claim {args['claim_id']!r}", is_error=True)
        return _tool_json(prov)

    def _tool_synthesize_topic(self, args: dict) -> dict:
        syn = self.mv.synthesize(args["topic"])
        return _tool_json(syn.to_dict())

    def _tool_find_contradictions(self, args: dict) -> dict:
        pairs = self.mv.contradictions(topic=args.get("topic"))
        return _tool_json([p.to_dict() for p in pairs])

    def _tool_consolidate(self, args: dict) -> dict:
        # M3.3: non-blocking — enqueue + return a handle so the JSON-RPC loop never
        # freezes on a long consolidation. Resolve completion via job_status.
        return _tool_json(self.mv.enqueue_consolidate())

    def _tool_job_status(self, args: dict) -> dict:
        job = self.mv.job_status(args.get("job_id", ""))
        if job is None:
            return _tool_text(f"No job {args.get('job_id')!r}", is_error=True)
        return _tool_json(job)

    def _tool_claim_timeline(self, args: dict) -> dict:
        # M3.1: reuse loom/evolution via the SDK (no new ordering logic)
        return _tool_json(self.mv.evolution(entity=args.get("entity"), topic=args.get("topic")))


def _version() -> str:
    from .. import __version__

    return __version__


def _error(req_id, code, message) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _tool_text(text: str, *, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _tool_json(obj) -> dict:
    return _tool_text(json.dumps(obj, indent=2, ensure_ascii=False))


def serve_stdio(mv: Memovox, *, stdin=None, stdout=None) -> None:
    """Run the MCP server loop over newline-delimited JSON-RPC on stdio."""
    server = McpServer(mv)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue
        response = server.handle(request)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
