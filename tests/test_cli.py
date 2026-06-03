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

        code, out = run(["--store", self.store, "list"])
        self.assertIn("yt:abc123", out)

        code, out = run(["--store", self.store, "stats"])
        self.assertIn("videos", out)

        code, out = run(["--store", self.store, "export", "--video", "yt:abc123", "--format", "md"])
        self.assertIn("youtu.be/abc123", out)

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

    def test_backends_runs(self):
        code, out = run(["--store", self.store, "backends"])
        self.assertEqual(code, 0)
        self.assertIn("hashing", out)

    def test_no_command_prints_help(self):
        code, _ = run(["--store", self.store])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
