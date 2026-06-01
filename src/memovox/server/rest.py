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

from ..loom import LoomStore
from ..sdk import Memovox
from ..util import deep_link


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
            length = int(self.headers.get("Content-Length", 0) or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except ValueError:
                return {}

        # -- GET ----------------------------------------------------------

        def do_GET(self):
            parsed = urlsplit(self.path)
            path = parsed.path
            q = parse_qs(parsed.query)
            try:
                if path == "/":
                    return self._send({"name": "memovox", "endpoints": [
                        "POST /ingest", "POST /query", "GET /clip", "GET /export/{id}",
                        "GET /graph/contradictions", "GET /videos"]})
                if path == "/videos":
                    return self._send([v.to_dict() for v in mv.list_videos()])
                if path == "/clip":
                    return self._clip(q)
                if path.startswith("/export/"):
                    return self._export(path[len("/export/"):], q)
                if path == "/graph/contradictions":
                    pairs = mv.contradictions(topic=(q.get("topic", [None])[0]))
                    return self._send([p.to_dict() for p in pairs])
                return self._send({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # pragma: no cover - defensive
                self._send({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _clip(self, q):
            video_id = q.get("video", [None])[0]
            t_start = float(q.get("t_start", [0])[0])
            t_end = float(q.get("t_end", [t_start])[0])
            with LoomStore(mv.config) as store:
                video = store.get_video(video_id) if video_id else None
                if not video:
                    return self._send({"error": "unknown video"}, HTTPStatus.NOT_FOUND)
                overlapping = [
                    m.to_dict() for m in store.moments_for_video(video_id)
                    if m.t_end_s >= t_start and m.t_start_s <= t_end
                ]
            self._send({
                "video_id": video_id, "t_start_s": t_start, "t_end_s": t_end,
                "deep_link": deep_link(video.source_url, t_start), "moments": overlapping,
            })

        def _export(self, video_id, q):
            fmt = q.get("format", ["json"])[0]
            try:
                content = mv.export(video_id, fmt=fmt)
            except KeyError:
                return self._send({"error": "unknown video"}, HTTPStatus.NOT_FOUND)
            if fmt == "json":
                self._send(json.loads(content))
            else:
                body = content.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # -- POST ---------------------------------------------------------

        def do_POST(self):
            parsed = urlsplit(self.path)
            data = self._body()
            try:
                if parsed.path == "/ingest":
                    if not data.get("source"):
                        return self._send({"error": "missing 'source'"}, HTTPStatus.BAD_REQUEST)
                    report = mv.ingest(data["source"], source_url=data.get("source_url"),
                                       title=data.get("title"))
                    return self._send(report.to_dict())
                if parsed.path == "/query":
                    if not data.get("query"):
                        return self._send({"error": "missing 'query'"}, HTTPStatus.BAD_REQUEST)
                    answer = mv.ask(data["query"], video_id=data.get("video_id"))
                    return self._send(answer.to_dict())
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
