"""M2.3 — REST /clip returns a stitched superset (legacy keys + additive clips)."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import Memovox
from memovox.server.rest import make_handler

VTT = """WEBVTT

00:00:10.000 --> 00:00:22.000
The recommended chunk size is 512 tokens for retrieval.

00:00:23.000 --> 00:00:35.000
That choice keeps latency low while preserving recall.
"""


class RestClipTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.mv = Memovox(store=self.dir / "store", llm_backend="none")
        vtt = self.dir / "t.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        self.mv.ingest(str(vtt), source_url="https://youtu.be/abc123")

    def tearDown(self):
        self._tmp.cleanup()

    def _clip(self, q):
        return self._get("/clip", q)  # /clip is now a routes.route_clip adapter

    def _get(self, path, q):
        Handler = make_handler(self.mv)
        h = Handler.__new__(Handler)
        captured = {}
        h._send = lambda obj, status=None: captured.update(payload=obj, status=status)
        # drive the do_GET dispatch via the closure's path handling
        h.path = path + ("?" + "&".join(f"{k}={v[0]}" for k, v in q.items()) if q else "")
        h.do_GET()
        return captured["payload"]

    def test_timeline_endpoint_reuses_evolution(self):
        # M3.1: /timeline returns ordered evolution steps (reuses loom/evolution)
        payload = self._get("/timeline", {"topic": ["chunk size"]})
        self.assertIsInstance(payload, list)  # ordered EvolutionStep dicts (possibly empty)

    def test_index_lists_timeline(self):
        payload = self._get("/", {})
        self.assertIn("GET /timeline", payload["endpoints"])

    def test_clip_endpoint_returns_stitched_superset(self):
        payload = self._clip({"video": ["yt:abc123"], "t_start": ["0"], "t_end": ["60"]})
        # legacy keys unchanged
        for key in ("video_id", "t_start_s", "t_end_s", "deep_link", "moments"):
            self.assertIn(key, payload)
        # additive stitched clips with a ranged deep link
        self.assertIn("clips", payload)
        self.assertTrue(payload["clips"])
        self.assertIn("youtube.com/watch?v=abc123", payload["clips"][0]["deep_link"])
        # the two adjacent moments merge into one window
        self.assertEqual(len(payload["clips"]), 1)


if __name__ == "__main__":
    unittest.main()
