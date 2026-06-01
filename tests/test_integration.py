import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import Memovox

VTT_RAG = """WEBVTT

00:00:10.000 --> 00:00:20.000
Welcome. The recommended chunk size is 512 tokens for retrieval.

00:00:20.000 --> 00:00:30.000
Hybrid retrieval combines dense and sparse search for the best recall.
"""


def write_vtt(dirpath, name, text):
    p = pathlib.Path(dirpath) / name
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.mv = Memovox(store=self.dir / "store", llm_backend="none")

    def tearDown(self):
        self._tmp.cleanup()

    def test_ingest_ask_export_idempotent(self):
        vtt = write_vtt(self.dir, "rag.en.vtt", VTT_RAG)
        report = self.mv.ingest(vtt, source_url="https://youtu.be/abc123")
        self.assertEqual(report.status, "ingested")
        self.assertEqual(report.video_id, "yt:abc123")
        self.assertGreaterEqual(report.n_moments, 1)
        self.assertGreaterEqual(report.n_claims_committed, 1)
        self.assertEqual(report.asr_backend, "captions")
        self.assertEqual(report.embed_backend, "hashing")

        ans = self.mv.ask("what chunk size is recommended?")
        self.assertFalse(ans.low_evidence)
        self.assertTrue(ans.citations)
        self.assertIn("[1]", ans.text)
        self.assertTrue(ans.citations[0].deep_link.startswith("https://youtu.be/abc123?t="))
        self.assertIn("chunk size", ans.citations[0].snippet.lower())

        md = self.mv.export("yt:abc123", fmt="md")
        self.assertIn("youtu.be/abc123", md)
        self.assertIn("Claims:", md)

        # digest file written (human-readable substrate)
        digests = list((self.dir / "store" / "digests").glob("*.md"))
        self.assertEqual(len(digests), 1)

        # idempotency: re-ingesting the same content is a no-op
        report2 = self.mv.ingest(vtt, source_url="https://youtu.be/abc123")
        self.assertEqual(report2.status, "unchanged")
        self.assertEqual(len(self.mv.list_videos()), 1)

    def test_provenance_lookup(self):
        vtt = write_vtt(self.dir, "rag.en.vtt", VTT_RAG)
        self.mv.ingest(vtt, source_url="https://youtu.be/abc123")
        # first committed claim id is deterministic
        prov = self.mv.get_provenance("yt:abc123#m0000.c00")
        self.assertIsNotNone(prov)
        self.assertIn("deep_link", prov["provenance"])

    def test_cross_corpus_contradiction(self):
        a = write_vtt(self.dir, "a.en.vtt",
                      "WEBVTT\n\n00:00:01.000 --> 00:00:09.000\n"
                      "Scaling laws hold well beyond the current regime.\n")
        b = write_vtt(self.dir, "b.en.vtt",
                      "WEBVTT\n\n00:00:01.000 --> 00:00:09.000\n"
                      "Scaling laws do not hold well beyond the current regime.\n")
        self.mv.ingest(a, source_url="https://youtu.be/aaa")
        self.mv.ingest(b, source_url="https://youtu.be/bbb")

        pairs = self.mv.contradictions(topic="scaling laws")
        self.assertTrue(pairs, "expected at least one contradiction pair")
        self.assertEqual(pairs[0].relation, "CONTRADICTS")
        self.assertNotEqual(pairs[0].claim_a.video_id, pairs[0].claim_b.video_id)


if __name__ == "__main__":
    unittest.main()
