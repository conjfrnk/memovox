"""Optional FastAPI app behind the [serve] extra (M3.3).

``fastapi`` is imported ONLY inside ``build_app`` (never at module load), so a bare
stdlib install never imports it. The app mounts the SAME ``routes.py`` pure
functions the stdlib ``http.server`` handler uses, so a JSON-parity test proves the
two servers return byte-identical responses. ``uvicorn`` is the production runner.
"""

# NB: deliberately NO ``from __future__ import annotations`` here. fastapi is imported
# only inside build_app, so the route handlers' ``request: Request`` annotations must
# evaluate EAGERLY at def-time (where Request is in local scope) to a real class.
# Stringized (PEP 563) annotations would be resolved by FastAPI via module globals,
# where Request does not exist -> PydanticUndefinedAnnotation. (W5.10)

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
    from starlette.concurrency import run_in_threadpool  # type: ignore

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

    async def _json_body(request: Request) -> dict:
        # Malformed / empty / non-object body -> {} (the route then answers a clean
        # 400), matching the stdlib server's _body() — keeps JSON parity, never 500s.
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - any JSON decode failure
            return {}
        return body if isinstance(body, dict) else {}

    # The route_* functions are SYNCHRONOUS and do heavy blocking work (route_ingest runs
    # the multi-minute download+ASR+NLI pipeline; ask/synthesize do SQLite + embedding +
    # rerank). Calling them inline in an `async def` would run that work ON the single
    # asyncio event-loop thread, freezing EVERY other request (even GET / health) for the
    # whole duration. run_in_threadpool offloads them so the loop stays responsive — the GET
    # handlers are already sync `def` (FastAPI threadpools those automatically); this gives
    # the POST handlers the same isolation. (The stdlib ThreadingHTTPServer is immune — one
    # OS thread per request — so this only matters for the uvicorn/FastAPI production runner.)
    @app.post("/ingest")
    async def _ingest(request: Request):
        body = await _json_body(request)
        return _respond(await run_in_threadpool(routes.route_ingest, mv, body))

    @app.post("/query")
    async def _query(request: Request):
        body = await _json_body(request)
        return _respond(await run_in_threadpool(routes.route_query, mv, body))

    @app.post("/synthesize")
    async def _synthesize(request: Request):
        body = await _json_body(request)
        return _respond(await run_in_threadpool(routes.route_synthesize, mv, body))

    @app.post("/consolidate")
    async def _consolidate(request: Request):
        body = await _json_body(request)
        return _respond(await run_in_threadpool(routes.route_consolidate, mv, body))

    return app
