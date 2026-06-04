"""Optional FastAPI app behind the [serve] extra (M3.3).

``fastapi`` is imported ONLY inside ``build_app`` (never at module load), so a bare
stdlib install never imports it. The app mounts the SAME ``routes.py`` pure
functions the stdlib ``http.server`` handler uses, so a JSON-parity test proves the
two servers return byte-identical responses. ``uvicorn`` is the production runner.
"""

from __future__ import annotations

import importlib.util

from ..errors import BackendUnavailable
from . import routes


def is_available() -> bool:
    return importlib.util.find_spec("fastapi") is not None


def build_app(mv):
    """Build a FastAPI app mounting routes.py. Raises BackendUnavailable if the
    [serve] extra is not installed (never an ImportError crash)."""
    if not is_available():
        raise BackendUnavailable(
            "FastAPI is not installed. Install it with: pip install 'memovox[serve]'."
        )
    from fastapi import FastAPI, Request  # type: ignore
    from fastapi.responses import JSONResponse, PlainTextResponse  # type: ignore

    app = FastAPI(title="memovox", version="0.1")

    def _respond(result):
        status, payload, content_type = result
        if content_type == routes.JSON:
            return JSONResponse(content=payload, status_code=int(status))
        return PlainTextResponse(content=payload, status_code=int(status),
                                 media_type=content_type)

    @app.get("/")
    def _index():
        return _respond(routes.route_index(mv))

    @app.get("/videos")
    def _videos():
        return _respond(routes.route_videos(mv))

    @app.get("/clip")
    def _clip(request: Request):
        return _respond(routes.route_clip(mv, dict(request.query_params)))

    @app.get("/timeline")
    def _timeline(request: Request):
        return _respond(routes.route_timeline(mv, dict(request.query_params)))

    @app.get("/export/{video_id}")
    def _export(video_id: str, request: Request):
        return _respond(routes.route_export(mv, video_id, dict(request.query_params)))

    @app.get("/graph/contradictions")
    def _contradictions(request: Request):
        return _respond(routes.route_contradictions(mv, dict(request.query_params)))

    @app.get("/job/{job_id}")
    def _job(job_id: str):
        return _respond(routes.route_job_status(mv, job_id))

    @app.post("/ingest")
    async def _ingest(request: Request):
        return _respond(routes.route_ingest(mv, await request.json()))

    @app.post("/query")
    async def _query(request: Request):
        return _respond(routes.route_query(mv, await request.json()))

    @app.post("/synthesize")
    async def _synthesize(request: Request):
        return _respond(routes.route_synthesize(mv, await request.json()))

    @app.post("/consolidate")
    async def _consolidate(request: Request):
        body = await request.json() if await request.body() else {}
        return _respond(routes.route_consolidate(mv, body))

    return app
