import contextlib
import io
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.cli import main

VTT = """WEBVTT

00:00:10.000 --> 00:00:22.000
The recommended chunk size is 512 tokens for retrieval augmented generation.
"""

VTT_ENTITY = """WEBVTT

00:00:02.000 --> 00:00:12.000
The Transformer architecture is the foundation of modern language models.
"""


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(argv)
    return code, out.getvalue()


class TestCLI(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.store = str(self.dir / "store")
        self.vtt = self.dir / "talk.en.vtt"
        self.vtt.write_text(VTT, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _ingest(self):
        return run(["--store", self.store, "--llm", "none", "ingest", str(self.vtt),
                    "--source-url", "https://youtu.be/abc123"])

    def test_ingest_then_ask_list_stats(self):
        code, out = self._ingest()
        self.assertEqual(code, 0)
        self.assertIn("ingested", out)
        self.assertIn("yt:abc123", out)

        code, out = run(["--store", self.store, "--llm", "none", "ask", "what chunk size?"])
        self.assertEqual(code, 0)
        self.assertIn("[1]", out)
        self.assertIn("youtu.be/abc123", out)
        # M2.3: a stitched Clips: block with a ranged deep link
        self.assertIn("Clips:", out)
        self.assertIn("youtube.com/watch?v=abc123", out)

        code, out = run(["--store", self.store, "--llm", "none", "ask", "what chunk size?", "--json"])
        self.assertIn("\"clips\"", out)

        code, out = run(["--store", self.store, "list"])
        self.assertIn("yt:abc123", out)

        code, out = run(["--store", self.store, "stats"])
        self.assertIn("videos", out)

        code, out = run(["--store", self.store, "export", "--video", "yt:abc123", "--format", "md"])
        self.assertIn("youtu.be/abc123", out)

    def test_subscribe_list_unsubscribe(self):
        url = "https://www.youtube.com/@chan"
        code, out = run(["--store", self.store, "subscribe", url])
        self.assertEqual(code, 0)
        self.assertIn("Subscribed", out)
        # idempotent
        run(["--store", self.store, "subscribe", url])
        code, out = run(["--store", self.store, "subscriptions"])
        self.assertEqual(out.count(url), 1)  # listed once, no dup
        run(["--store", self.store, "unsubscribe", url])
        code, out = run(["--store", self.store, "subscriptions"])
        self.assertIn("No subscriptions", out)

    def test_worker_once_drains_queue(self):
        from memovox import Memovox
        mv = Memovox(store=self.store, llm_backend="none")
        mv.ingest(str(self.vtt), source_url="https://youtu.be/abc123")
        job = mv.enqueue_consolidate()
        code, out = run(["--store", self.store, "--llm", "none", "worker", "--once"])
        self.assertEqual(code, 0)
        self.assertEqual(mv.job_status(job["job_id"])["state"], "succeeded")

    def test_worker_default_concurrency_is_one(self):
        from memovox.cli import build_parser
        args = build_parser().parse_args(["worker"])
        self.assertEqual(args.concurrency, 1)

    def test_sync_prints_structured_summary(self):
        from unittest import mock
        from memovox.stentor.acquire import EnumeratedEntry
        run(["--store", self.store, "subscribe", "https://www.youtube.com/@chan"])
        entry = EnumeratedEntry("yt:abc123", str(self.vtt), "talk")
        with mock.patch("memovox.stentor.enumerate_source", return_value=[entry]):
            code, out = run(["--store", self.store, "--llm", "none", "sync"])
        self.assertEqual(code, 0)
        self.assertIn("1 new", out)
        self.assertIn("[new]", out)

    def test_metrics_command(self):
        self._ingest()
        code, out = run(["--store", self.store, "metrics"])
        self.assertEqual(code, 0)
        self.assertNotIn("Traceback", out)
        self.assertIn("ledger", out.lower())          # cumulative ledger surfaced
        self.assertIn("claims", out.lower())           # a per-video stage row
        self.assertIn("yt:abc123", out)                # per-video table keyed by id

    def test_stats_includes_metrics_summary(self):
        self._ingest()
        code, out = run(["--store", self.store, "stats"])
        self.assertEqual(code, 0)
        self.assertIn("ledger", out.lower())           # new metrics summary line

    def test_evolution_by_entity(self):
        self.vtt.write_text(VTT_ENTITY, encoding="utf-8")
        self._ingest()
        code, out = run(["--store", self.store, "--llm", "none",
                         "evolution", "--entity", "Transformer"])
        self.assertEqual(code, 0)
        self.assertIn("Transformer", out)
        self.assertIn("youtu.be/abc123", out)

    def test_evolution_requires_entity_or_topic(self):
        # argparse rejects the missing required group with a usage error (exit 2).
        with self.assertRaises(SystemExit) as cm:
            run(["--store", self.store, "evolution"])
        self.assertEqual(cm.exception.code, 2)

    def test_consolidate_runs(self):
        self._ingest()
        code, out = run(["--store", self.store, "--llm", "none", "consolidate"])
        self.assertEqual(code, 0)
        self.assertIn("topics induced", out)
        self.assertIn("claims superseded", out)

    def test_synthesize_runs(self):
        self._ingest()
        code, out = run(["--store", self.store, "--llm", "none", "synthesize", "chunk", "size"])
        self.assertEqual(code, 0)
        # Single-source corpus: valid synthesis output, no crash.
        code, out = run(["--store", self.store, "--llm", "none", "synthesize", "chunk", "--json"])
        self.assertIn("consensus_points", out)

    def test_backends_runs(self):
        code, out = run(["--store", self.store, "backends"])
        self.assertEqual(code, 0)
        self.assertIn("hashing", out)

    def test_no_command_prints_help(self):
        code, _ = run(["--store", self.store])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
