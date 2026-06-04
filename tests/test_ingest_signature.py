"""M0.3 W5 — the canonical pipeline.ingest() signature + word-precision on the
free path (a word-bearing JSON fixture narrows a claim span below its cue)."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import pipeline
from memovox.config import Config, Settings
from memovox.loom.store import LoomStore

_FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "eval" / "fixtures"
_FREE = dict(embed_backend="hashing", nli_backend="lexical", asr_backend="captions",
             llm_backend="none", vlm_backend="none", ocr_backend="none", entity_backend="none")

VTT = """WEBVTT

00:00:10.000 --> 00:00:20.000
The recommended chunk size is 512 tokens for retrieval.
"""


class IngestSignatureTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.config = Config(store=self.dir / "store", settings=Settings(**_FREE)).ensure()

    def tearDown(self):
        self._tmp.cleanup()

    def test_keyword_only_seams_accepted_and_published_at_override(self):
        vtt = self.dir / "talk.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        report = pipeline.ingest(
            self.config, str(vtt), source_url="https://youtu.be/abc123",
            published_at="2024-01-02", visual_result=None, modality=None,
        )
        self.assertEqual(report.status, "ingested")
        with LoomStore(self.config) as store:
            self.assertEqual(store.get_video(report.video_id).published_at, "2024-01-02")


class WordPrecisionFreePathTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.config = Config(store=self.dir / "store", settings=Settings(**_FREE)).ensure()

    def tearDown(self):
        self._tmp.cleanup()

    def test_word_bearing_fixture_narrows_a_claim_below_its_cue(self):
        report = pipeline.ingest(self.config, str(_FIXTURES / "words_clip.json"),
                                 source_url="https://x/words")
        self.assertGreaterEqual(report.n_claims_committed, 1)
        with LoomStore(self.config) as store:
            claims = store.claims_for_video(report.video_id)
            # the cue spans 0..12s; word tightening must pull at least one claim
            # window strictly inside that cue (word-precision on the free path).
            self.assertTrue(any(c.t_end_s < 12.0 for c in claims),
                            f"expected a word-narrowed span; got {[(c.t_start_s, c.t_end_s) for c in claims]}")


if __name__ == "__main__":
    unittest.main()
