"""M3.3 W3 — server/routes.py pure functions + a live-server JSON parity tripwire.

The stdlib http.server handler is now a thin adapter over routes.py; this locks
that the refactor is byte-identical (the handler returns exactly what the pure
function produces).
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import Memovox
from memovox.server import routes
from memovox.server.rest import make_handler

VTT = "WEBVTT\n\n00:00:10.000 --> 00:00:22.000\nThe recommended chunk size is 512 tokens.\n"


class RoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.mv = Memovox(store=self.dir / "store", llm_backend="none")
        vtt = self.dir / "t.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        self.mv.ingest(str(vtt), source_url="https://youtu.be/abc123")

    def tearDown(self):
        self._tmp.cleanup()

    def test_pure_functions_return_status_payload(self):
        st, payload, ct = routes.route_videos(self.mv)
        self.assertEqual(st, 200)
        self.assertEqual(ct, routes.JSON)
        self.assertTrue(any(v["video_id"] == "yt:abc123" for v in payload))

        st, payload, ct = routes.route_query(self.mv, {"query": "chunk size?"})
        self.assertEqual(st, 200)
        self.assertIn("citations", payload)

        st, payload, ct = routes.route_export(self.mv, "yt:abc123", {"format": "md"})
        self.assertEqual(ct, routes.MARKDOWN)
        self.assertIsInstance(payload, str)

        st, payload, _ = routes.route_query(self.mv, {})  # missing query
        self.assertEqual(st, 400)

    def test_handler_adapter_matches_pure_function(self):
        # The stdlib handler is a thin adapter: driving do_GET must produce exactly
        # what the pure route returns (the refactor parity tripwire), no socket.
        Handler = make_handler(self.mv)
        h = Handler.__new__(Handler)
        captured = {}
        h._send = lambda obj, status=200: captured.update(payload=obj, status=status)
        h.path = "/videos"
        h.do_GET()
        _, pure, _ = routes.route_videos(self.mv)
        self.assertEqual(captured["payload"], pure)
        self.assertEqual(captured["status"], 200)


if __name__ == "__main__":
    unittest.main()
