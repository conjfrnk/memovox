"""Agent-native MCP server over stdio (spec §8) — implemented with the standard
library (no ``mcp`` package required).

Speaks newline-delimited JSON-RPC 2.0, the MCP stdio transport. Wire it into
Claude Code / Claude Desktop to ingest a talk and immediately interrogate it
without leaving the editor. Exposed tools:
    ingest_video, search_knowledge, list_videos, get_claim_provenance,
    synthesize_topic, find_contradictions, claim_timeline, consolidate,
    job_status

The server ``instructions`` string and the tool descriptions are written for
the *model*, not for developers: they name the user intents that should
trigger each tool ("watch this video", "what does the talk say about X?") so
MCP clients reach for memovox unprompted instead of only when named.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

from ..sdk import Memovox

# Newest first. Our wire surface (initialize + tools/list + tools/call with
# text content) is the common subset of every revision, so we echo the
# client's version when we recognise it and offer our newest otherwise
# (spec: lifecycle/version negotiation) — some clients disconnect when the
# server names a version they never asked for.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

# Injected into the client's system prompt (InitializeResult.instructions) —
# this is what makes "watch this video" route here without naming memovox.
INSTRUCTIONS = """\
memovox turns videos into a queryable knowledge base where every answer carries \
timestamped citations and deep links.

Use this server without being asked for it by name whenever the user:
- shares a video/audio URL (YouTube, a talk, a podcast) or a local media/transcript \
file and wants it watched, summarized, analyzed, or queried -> call ingest_video, \
tell the user ingestion is underway (a long video takes minutes), poll job_status a \
few times, then answer with search_knowledge once it succeeds. If it is still \
running after a few polls, stop and tell the user it continues in the background — \
the job_id stays valid, so check job_status again when they next ask.
- asks any question about a video or talk that may already be ingested -> \
search_knowledge; call list_videos first if unsure what is available.
- asks what their videos collectively say about a topic -> synthesize_topic; where \
sources disagree -> find_contradictions; how a claim or number evolved over time -> \
claim_timeline.

Run consolidate (async; poll job_status) after ingesting several videos or before \
cross-video questions (synthesize_topic, find_contradictions, claim_timeline) — \
single-video questions via search_knowledge need no consolidation. Always surface \
the citations (timestamps and deep links) in answers; when asked where a claim came \
from, resolve it with get_claim_provenance."""

TOOLS = [
    {
        "name": "ingest_video",
        "description": "Watch/ingest a video so it becomes queryable knowledge. Use whenever "
                       "the user shares a video or audio URL (YouTube, talks, podcasts) or a "
                       "local media/transcript file and wants it watched, summarized, "
                       "analyzed, or asked about. Returns {job_id, state} immediately; "
                       "ingestion runs in the background and can take several minutes for a "
                       "long video. Poll job_status until state is 'succeeded', then query "
                       "with search_knowledge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string",
                        "description": "Video/audio URL (https://...) or local file path "
                                       "(media file or transcript)."},
                "source_url": {"type": "string",
                               "description": "Canonical public URL used for timestamped "
                                              "deep links — set it when ingesting a local "
                                              "copy of an online video."},
                "title": {"type": "string", "description": "Optional title override."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "Answer a question from the ingested videos, with timestamped "
                       "citations and deep links. Use for any question about a video, talk, "
                       "or topic in the knowledge base — e.g. 'what did they say about X?', "
                       "'summarize the main argument'. Refuses (rather than guessing) when "
                       "the corpus holds no evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question to answer."},
                "modality": {"type": "string", "enum": ["any", "speech", "visual"],
                             "description": "'speech' = what was said, 'visual' = what was "
                                            "shown (slides/screen/charts), 'any' (default) "
                                            "= both."},
                "video_id": {"type": "string",
                             "description": "Restrict to one video (ids come from "
                                            "list_videos or an ingest result)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_videos",
        "description": "List the videos already in the knowledge base (video_id, title, "
                       "source URL, duration, ingest date). Use to check whether a video is "
                       "already ingested or to find a video_id for search_knowledge.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_claim_provenance",
        "description": "Resolve a claim_id (from search/synthesis results) to its exact "
                       "source: video, time span, and deep link. Use when the user asks "
                       "where a claim came from or wants to verify it.",
        "inputSchema": {
            "type": "object",
            "properties": {"claim_id": {"type": "string"}},
            "required": ["claim_id"],
        },
    },
    {
        "name": "synthesize_topic",
        "description": "Synthesize what ALL ingested videos say about one topic — consensus "
                       "points, contradictions, citations. Use for cross-video questions "
                       "like 'what do my videos say about X?'.",
        "inputSchema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
    {
        "name": "find_contradictions",
        "description": "Find points where the ingested videos disagree with each other, "
                       "optionally restricted to a topic. Use when the user asks about "
                       "conflicts, disagreements, or inconsistencies in the corpus.",
        "inputSchema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
        },
    },
    {
        "name": "consolidate",
        "description": "Rebuild the cross-video layer (topic induction, contradiction/"
                       "agreement detection, consensus, dedup). Run after ingesting "
                       "several new videos or before cross-video questions; single-video "
                       "search_knowledge needs no consolidation. Returns {job_id, state} "
                       "immediately; poll job_status.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "claim_timeline",
        "description": "Trace how a position or number about an entity or topic changed "
                       "across videos over time (ordered, deep-linked evolution steps). Use "
                       "for 'how did their estimate/stance change?' questions.",
        "inputSchema": {
            "type": "object",
            "properties": {"entity": {"type": "string"}, "topic": {"type": "string"}},
        },
    },
    {
        "name": "job_status",
        "description": "Check a background job started by ingest_video or consolidate. "
                       "Returns {state, result, error}; state is 'queued', 'running', "
                       "'succeeded', or 'failed'. Poll a few times at most, then hand "
                       "back to the user — a long video ingest can take several minutes "
                       "and the job keeps running in the background.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
]

VALID_MODALITIES = ("any", "speech", "visual")


class McpServer:
    def __init__(self, mv: Memovox) -> None:
        self.mv = mv

    # -- JSON-RPC dispatch (unit-testable) --------------------------------

    def handle(self, request: dict) -> Optional[dict]:
        """Handle one JSON-RPC request; return a response, or None for notifications."""
        if not isinstance(request, dict):
            # A non-object JSON value (array/number/string/null) is an invalid
            # request — answer with -32600 instead of crashing on ``.get``.
            return _error(None, -32600, "Invalid Request: expected a JSON object")
        method = request.get("method")
        req_id = request.get("id")
        is_notification = "id" not in request

        try:
            if method == "initialize":
                params = request.get("params") or {}
                asked = params.get("protocolVersion")
                version = asked if asked in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
                result = {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "memovox", "version": _version()},
                    "instructions": INSTRUCTIONS,
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
        except KeyError as exc:  # a missing required tool argument -> Invalid params
            if is_notification:
                return None
            return _error(req_id, -32602, f"Missing required parameter: {exc}")
        except Exception as exc:  # surface other dispatch errors as JSON-RPC errors
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
            known = ", ".join(t["name"] for t in TOOLS)
            return _tool_text(f"Unknown tool: {name}. Available tools: {known}",
                              is_error=True)
        try:
            return handler(args)
        except KeyError:
            raise  # missing required argument -> -32602 Invalid params (see handle())
        except Exception as exc:
            # Tool *execution* failures belong in the result with isError (MCP spec,
            # tools): the model sees the message and can self-correct, whereas a
            # -32603 protocol error is opaque to it.
            return _tool_text(f"{type(exc).__name__}: {exc}", is_error=True)

    # -- tools ------------------------------------------------------------

    def _tool_ingest_video(self, args: dict) -> dict:
        # Non-blocking like consolidate: a real ingest (download + ASR + NLI) runs
        # minutes, but MCP clients cancel long calls (Claude Desktop at 240 s) and
        # the result is lost even though the ingest finishes. Enqueue + job_status.
        url = args["url"]
        # Validate up front — an immediate, actionable error beats enqueueing a job
        # whose failure the model only discovers after polling job_status.
        if not isinstance(url, str) or not url.strip():
            return _tool_text("'url' must be a non-empty string: a video/audio URL or a "
                              "local file path.", is_error=True)
        url = url.strip()
        if url.lower().startswith("file://"):
            # Models routinely write file:///path for local files — accept it.
            from urllib.parse import unquote, urlparse
            url = unquote(urlparse(url).path) or url
        if url.lower().startswith(("http://", "https://")):
            if self.mv.settings.local_only:
                return _tool_text(
                    f"local_only is set: refusing to acquire the remote source {url!r}. "
                    "Ingest a downloaded local file instead, or unset local_only.",
                    is_error=True)
            # Schemes are case-insensitive (RFC 3986) but the pipeline's URL
            # detection is not — canonicalize so HTTPS://... doesn't fail late
            # as a "missing local file".
            scheme, rest = url.split("://", 1)
            url = scheme.lower() + "://" + rest
        elif "://" in url:
            scheme = url.split("://", 1)[0]
            return _tool_text(
                f"Unsupported URL scheme {scheme!r} — use an https:// URL or a plain "
                "local file path.", is_error=True)
        else:
            path = pathlib.Path(url).expanduser()
            if not path.is_file():
                detail = "is a directory, not a file" if path.is_dir() else "was not found"
                return _tool_text(
                    f"Local file {url!r} {detail}. Pass the path of an existing media or "
                    "transcript file, or a full web URL (https://...).", is_error=True)
        handle = self.mv.enqueue_ingest(
            url, source_url=args.get("source_url"), title=args.get("title"))
        handle["hint"] = (
            "Ingestion runs in the background and can take several minutes for a long "
            "video. Tell the user it is underway, then poll job_status with this job_id "
            "a few times; once it succeeds, answer questions with search_knowledge. If "
            "it is still running after a few polls, stop and tell the user it continues "
            "in the background — the job_id stays valid across turns.")
        return _tool_json(handle)

    def _tool_search_knowledge(self, args: dict) -> dict:
        videos = self.mv.list_videos()
        if not videos:
            return _empty_corpus()
        modality = args.get("modality", "any")
        if modality not in VALID_MODALITIES:
            return _tool_text(
                f"Unknown modality {modality!r} — use one of {', '.join(VALID_MODALITIES)}.",
                is_error=True)
        video_id = args.get("video_id")
        if video_id and all(v.video_id != video_id for v in videos):
            return _tool_text(
                f"Unknown video_id {video_id!r} — call list_videos for the available ids, "
                "or omit video_id to search the whole knowledge base.",
                is_error=True)
        answer = self.mv.ask(args["query"], video_id=video_id, modality=modality)
        return _tool_json(answer.to_dict())

    def _tool_list_videos(self, args: dict) -> dict:
        videos = [v.to_dict() for v in self.mv.list_videos()]
        out = {"count": len(videos), "videos": videos}
        if not videos:
            out["hint"] = "No videos ingested yet — use ingest_video to add one."
        return _tool_json(out)

    def _tool_get_claim_provenance(self, args: dict) -> dict:
        prov = self.mv.get_provenance(args["claim_id"])
        if prov is None:
            return _tool_text(
                f"No claim {args['claim_id']!r} — claim ids come from search_knowledge / "
                "synthesize_topic citations.", is_error=True)
        return _tool_json(prov)

    def _tool_synthesize_topic(self, args: dict) -> dict:
        if not self.mv.list_videos():
            return _empty_corpus()
        syn = self.mv.synthesize(args["topic"])
        return _tool_json(syn.to_dict())

    def _tool_find_contradictions(self, args: dict) -> dict:
        if not self.mv.list_videos():
            return _empty_corpus()
        pairs = self.mv.contradictions(topic=args.get("topic"))
        return _tool_json([p.to_dict() for p in pairs])

    def _tool_consolidate(self, args: dict) -> dict:
        # M3.3: non-blocking — enqueue + return a handle so the JSON-RPC loop never
        # freezes on a long consolidation. Resolve completion via job_status.
        handle = self.mv.enqueue_consolidate()
        handle["hint"] = ("Consolidation runs in the background — poll job_status with "
                          "this job_id until state is 'succeeded'.")
        return _tool_json(handle)

    def _tool_job_status(self, args: dict) -> dict:
        job = self.mv.job_status(args.get("job_id", ""))
        if job is None:
            return _tool_text(
                f"No job {args.get('job_id')!r}. Job ids come from ingest_video / "
                "consolidate responses; if the id was lost, re-run that tool.",
                is_error=True)
        job["hint"] = _job_hint(job)
        return _tool_json(job)

    def _tool_claim_timeline(self, args: dict) -> dict:
        # M3.1: reuse loom/evolution via the SDK (no new ordering logic)
        if not self.mv.list_videos():
            return _empty_corpus()
        return _tool_json(self.mv.evolution(entity=args.get("entity"), topic=args.get("topic")))


def _version() -> str:
    from .. import __version__

    return __version__


def _job_hint(job: dict) -> str:
    """Next-step guidance embedded in every job_status payload — the model acts on
    it directly instead of guessing what a state means."""
    state, kind = job.get("state"), job.get("kind")
    if state in ("queued", "running"):
        return ("Still working. Poll job_status a few more times; if it is still not "
                "finished, stop and tell the user the job continues in the background "
                "(a long video ingest takes several minutes) — this job_id stays valid, "
                "so check it again when they next ask.")
    if state == "failed":
        return ("The job failed — report the error to the user. Fixing the source "
                "(path/URL) and re-running the tool usually resolves it.")
    if kind == "ingest":
        return ("Ingest complete — the video is now queryable with search_knowledge "
                "(its video_id is in result). After ingesting several videos, run "
                "consolidate to refresh cross-video topics and contradictions.")
    return ("Done — synthesize_topic and find_contradictions answers now reflect the "
            "whole corpus.")


def _empty_corpus() -> dict:
    return _tool_text(
        "The knowledge base is empty — no videos have been ingested yet. Ingest one "
        "first with ingest_video (a video/audio URL or local file), poll job_status "
        "until it succeeds, then ask again.", is_error=True)


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
            response = _error(None, -32700, "Parse error")
        else:
            try:
                response = server.handle(request)
            except Exception as exc:  # a handler bug must never kill the loop
                print(f"mcp: unhandled error: {exc}", file=sys.stderr)
                response = _error(None, -32603, "Internal error")
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
