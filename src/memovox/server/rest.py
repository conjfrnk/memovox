"""REST API (spec §8), implemented on the standard-library http.server so it
needs no web framework. FastAPI/uvicorn are the documented production option
(``pip install "memovox[serve]"``); this keeps the default free and dependency-free.

Endpoints:
    POST /ingest                {source, source_url?, title?}
    POST /query                 {query, video_id?}
    GET  /clip?video&t_start&t_end
    GET  /export/{video_id}?format=md|json
    GET  /graph/contradictions?topic=
    GET  /videos
    GET  /                       health/index
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from ..sdk import Memovox
from . import routes


def make_handler(mv: Memovox):
    class Handler(BaseHTTPRequestHandler):
        server_version = "memovox/0.1"

        def log_message(self, *args):  # quiet
            return

        def _send(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except (TypeError, ValueError):
                return {}  # malformed Content-Length -> empty body, never crash
            if length <= 0:
                return {}
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}
            return data if isinstance(data, dict) else {}  # non-object body -> {}

        # -- adapter: parse → call the pure route → serialize ------------

        def _respond(self, result):
            status, payload, content_type = result
            if content_type == routes.JSON:
                return self._send(payload, status)
            body = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # -- GET ----------------------------------------------------------

        def do_GET(self):
            parsed = urlsplit(self.path)
            path = parsed.path
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                if path == "/":
                    return self._respond(routes.route_index(mv))
                if path == "/videos":
                    return self._respond(routes.route_videos(mv))
                if path == "/clip":
                    return self._respond(routes.route_clip(mv, params))
                if path == "/timeline":
                    return self._respond(routes.route_timeline(mv, params))
                if path.startswith("/export/"):
                    return self._respond(routes.route_export(mv, path[len("/export/"):], params))
                if path == "/graph/contradictions":
                    return self._respond(routes.route_contradictions(mv, params))
                if path.startswith("/job/"):
                    return self._respond(routes.route_job_status(mv, path[len("/job/"):]))
                return self._send({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # pragma: no cover - defensive
                self._send({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        # -- POST ---------------------------------------------------------

        def do_POST(self):
            parsed = urlsplit(self.path)
            data = self._body()
            try:
                if parsed.path == "/ingest":
                    return self._respond(routes.route_ingest(mv, data))
                if parsed.path == "/query":
                    return self._respond(routes.route_query(mv, data))
                if parsed.path == "/synthesize":
                    return self._respond(routes.route_synthesize(mv, data))
                if parsed.path == "/consolidate":
                    return self._respond(routes.route_consolidate(mv, data))
                return self._send({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # pragma: no cover - defensive
                self._send({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def serve(mv: Memovox, *, host: str = "127.0.0.1", port: int = 8808) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(mv))
    print(f"memovox REST API on http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
