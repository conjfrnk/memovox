"""M3.3 W5 — optional FastAPI app behind [serve], JSON-parity with the stdlib server.

Skips cleanly when fastapi is absent (the bare/free machine). The unavailable path
must raise BackendUnavailable, never crash on import.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import Memovox
from memovox.errors import BackendUnavailable
from memovox.server import fastapi_app, routes

VTT = "WEBVTT\n\n00:00:10.000 --> 00:00:22.000\nThe recommended chunk size is 512 tokens.\n"


class FastApiParityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.mv = Memovox(store=self.dir / "store", llm_backend="none")
        vtt = self.dir / "t.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        self.mv.ingest(str(vtt), source_url="https://youtu.be/abc123")

    def tearDown(self):
        self._tmp.cleanup()

    def test_unavailable_is_clean(self):
        if fastapi_app.is_available():
            self.skipTest("fastapi installed")
        with self.assertRaises(BackendUnavailable):
            fastapi_app.build_app(self.mv)

    def test_stdlib_fastapi_json_parity(self):
        if not fastapi_app.is_available():
            self.skipTest("fastapi not installed (free path)")
        from fastapi.testclient import TestClient

        client = TestClient(fastapi_app.build_app(self.mv))
        # GET routes: FastAPI response == the pure route payload (which the stdlib
        # handler also returns verbatim).
        for path, route_call in [
            ("/videos", lambda: routes.route_videos(self.mv)),
            ("/clip?video=yt:abc123&t_start=0&t_end=60",
             lambda: routes.route_clip(self.mv, {"video": "yt:abc123", "t_start": "0", "t_end": "60"})),
        ]:
            resp = client.get(path)
            _, pure, _ = route_call()
            self.assertEqual(resp.json(), pure, path)
        # POST /query parity
        resp = client.post("/query", json={"query": "chunk size?"})
        _, pure, _ = routes.route_query(self.mv, {"query": "chunk size?"})
        self.assertEqual(resp.json(), pure)


if __name__ == "__main__":
    unittest.main()
