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

from ..loom import LoomStore
from ..util import deep_link

JSON = "application/json"
MARKDOWN = "text/markdown; charset=utf-8"


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
    if not body.get("source"):
        return (HTTPStatus.BAD_REQUEST, {"error": "missing 'source'"}, JSON)
    report = mv.ingest(body["source"], source_url=body.get("source_url"),
                       title=body.get("title"))
    return (HTTPStatus.OK, report.to_dict(), JSON)


def route_query(mv, body):
    # Accept 'question' as an alias for 'query' — the CLI (ask), SDK (.ask) and MCP
    # (search_knowledge) all phrase it as a question, so a client need not guess.
    query = body.get("query") or body.get("question")
    if not query:
        return (HTTPStatus.BAD_REQUEST, {"error": "missing 'query' (or 'question')"}, JSON)
    answer = mv.ask(query, video_id=body.get("video_id"))
    return (HTTPStatus.OK, answer.to_dict(), JSON)


def route_synthesize(mv, body):
    if not body.get("topic"):
        return (HTTPStatus.BAD_REQUEST, {"error": "missing 'topic'"}, JSON)
    return (HTTPStatus.OK, mv.synthesize(body["topic"]).to_dict(), JSON)


def route_consolidate(mv, body):
    # M3.3: non-blocking — enqueue a job and return its handle immediately.
    return (HTTPStatus.OK, mv.enqueue_consolidate(), JSON)
