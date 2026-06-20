"""Framework-agnostic request handlers (M3.3).

Each ``route_*`` is a PURE function ``(mv, ...) -> (status, payload, content_type)``
with no HTTP-server coupling, so the stdlib ``http.server`` handler AND the optional
FastAPI app call the SAME logic — one source of truth for request semantics, locked
by a JSON-parity test. ``payload`` is a JSON-serializable object for
``application/json`` routes, or a ``str`` for the Markdown export.
"""

from __future__ import annotations

import json as _json
from http import HTTPStatus

import math

from ..errors import AcquisitionError, DemuxError, IngestionError
from ..loom import LoomStore
from ..util import deep_link

JSON = "application/json"
MARKDOWN = "text/markdown; charset=utf-8"


def _bad_request(msg: str):
    return (HTTPStatus.BAD_REQUEST, {"error": msg}, JSON)


def _wrong_type(body, key: str):
    """Return a 400 if ``body[key]`` is present but NOT a usable string, else None. Guards
    the POST routes so a non-string field (``{"query": 123}``, ``{"video_id": ["x"]}``)
    is rejected at the boundary instead of flowing into ``.strip()`` / a SQLite bound
    parameter and surfacing a raw Python/SQLite error in a 500. Also rejects a string that
    json.loads accepts but is not encodable — a LONE SURROGATE (``"\\ud800"``) — which would
    otherwise crash the SQLite parameter bind (UnicodeEncodeError) deep in a query; the
    encode check raises ONLY for lone surrogates (emoji / CJK / accents all pass)."""
    val = body.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        return _bad_request(f"{key!r} must be a string")
    try:
        val.encode("utf-8")
    except UnicodeEncodeError:
        return _bad_request(f"{key!r} contains invalid characters")
    return None


def _finite_float(value, default: float):
    """Parse a query param to a FINITE float (rejecting nan/inf and junk). Returns
    ``default`` when the value is None/empty, or ``None`` when it's unparseable —
    so a route can answer 400 instead of crashing or accepting nan/inf."""
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def route_index(mv):
    return (HTTPStatus.OK, {"name": "memovox", "endpoints": [
        "POST /ingest", "POST /query", "POST /synthesize", "POST /consolidate",
        "GET /clip", "GET /export/{id}", "GET /graph/contradictions",
        "GET /timeline", "GET /videos", "GET /job/{id}"]}, JSON)


def route_videos(mv):
    return (HTTPStatus.OK, [v.to_dict() for v in mv.list_videos()], JSON)


def route_clip(mv, params):
    from ..augur.stitch import stitch_clips
    from ..augur.types import Citation

    video_id = params.get("video")
    if not video_id:
        return (HTTPStatus.BAD_REQUEST, {"error": "missing 'video'"}, JSON)
    t_start = _finite_float(params.get("t_start"), 0.0)
    t_end = _finite_float(params.get("t_end"), t_start)
    if t_start is None or t_end is None:
        return (HTTPStatus.BAD_REQUEST, {"error": "t_start/t_end must be finite numbers"}, JSON)
    with LoomStore(mv.config) as store:
        video = store.get_video(video_id) if video_id else None
        if not video:
            return (HTTPStatus.NOT_FOUND, {"error": "unknown video"}, JSON)
        moments = [m for m in store.moments_for_video(video_id)
                   if m.t_end_s >= t_start and m.t_start_s <= t_end]
    cits = [Citation(index=i, video_id=video_id, moment_id=m.moment_id,
                     t_start_s=m.t_start_s, t_end_s=m.t_end_s, title=video.title)
            for i, m in enumerate(moments, start=1)]
    clips = stitch_clips(cits, videos={video_id: video},
                         merge_gap_s=mv.settings.clip_merge_gap_s)
    return (HTTPStatus.OK, {
        "video_id": video_id, "t_start_s": t_start, "t_end_s": t_end,
        "deep_link": deep_link(video.source_url, t_start),
        "moments": [m.to_dict() for m in moments],
        "clips": [c.to_dict() for c in clips],
    }, JSON)


def route_timeline(mv, params):
    entity, topic = params.get("entity"), params.get("topic")
    if not entity and not topic:
        return (HTTPStatus.BAD_REQUEST, {"error": "provide 'entity' or 'topic'"}, JSON)
    return (HTTPStatus.OK, mv.evolution(entity=entity, topic=topic), JSON)


def route_export(mv, video_id, params):
    fmt = params.get("format") or "json"
    try:
        content = mv.export(video_id, fmt=fmt)
    except KeyError:
        return (HTTPStatus.NOT_FOUND, {"error": "unknown video"}, JSON)
    except ValueError as exc:
        # bad client-supplied format (mv.export raises ValueError) -> 400, not a 500 that
        # leaks the raw exception via do_GET's catch-all (and FastAPI parity).
        return _bad_request(str(exc))
    if fmt == "json":
        return (HTTPStatus.OK, _json.loads(content), JSON)
    return (HTTPStatus.OK, content, MARKDOWN)


def route_contradictions(mv, params):
    pairs = mv.contradictions(topic=params.get("topic"))
    return (HTTPStatus.OK, [p.to_dict() for p in pairs], JSON)


def route_job_status(mv, job_id):
    job = mv.job_status(job_id)
    if job is None:
        return (HTTPStatus.NOT_FOUND, {"error": "unknown job"}, JSON)
    return (HTTPStatus.OK, job, JSON)


def route_ingest(mv, body):
    for key in ("source", "source_url", "title"):
        err = _wrong_type(body, key)
        if err:
            return err
    if not body.get("source"):
        return _bad_request("missing 'source'")
    try:
        report = mv.ingest(body["source"], source_url=body.get("source_url"),
                           title=body.get("title"))
    except (AcquisitionError, DemuxError, IngestionError) as exc:
        # a bad client-supplied source (missing file, bad URL, unsupported type, ffmpeg
        # failure) is a CLIENT error -> 400, not a catch-all 500 that leaks the internal
        # exception (mirrors the route_export ValueError->400 fix; keeps FastAPI parity).
        # IngestionError covers the local_only egress refusal (pipeline.ingest raises it
        # for an http(s) source when private): an EXPECTED, client-driven refusal that the
        # MCP ingest_video tool already returns cleanly — REST/FastAPI must match, not 500.
        return _bad_request(str(exc))
    return (HTTPStatus.OK, report.to_dict(), JSON)


def route_query(mv, body):
    # Accept 'question' as an alias for 'query' — the CLI (ask), SDK (.ask) and MCP
    # (search_knowledge) all phrase it as a question, so a client need not guess.
    for key in ("query", "question", "video_id"):
        err = _wrong_type(body, key)
        if err:
            return err
    query = body.get("query") or body.get("question")
    if not query:
        return _bad_request("missing 'query' (or 'question')")
    answer = mv.ask(query, video_id=body.get("video_id"))
    return (HTTPStatus.OK, answer.to_dict(), JSON)


def route_synthesize(mv, body):
    err = _wrong_type(body, "topic")
    if err:
        return err
    if not body.get("topic"):
        return _bad_request("missing 'topic'")
    return (HTTPStatus.OK, mv.synthesize(body["topic"]).to_dict(), JSON)


def route_consolidate(mv, body):
    # M3.3: non-blocking — enqueue a job and return its handle immediately.
    return (HTTPStatus.OK, mv.enqueue_consolidate(), JSON)
