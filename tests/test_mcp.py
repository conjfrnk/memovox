import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import Memovox
from memovox.server.mcp import McpServer

VTT = """WEBVTT

00:00:10.000 --> 00:00:22.000
The recommended chunk size is 512 tokens for retrieval.
"""


class TestMcp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.mv = Memovox(store=self.dir / "store", llm_backend="none")
        vtt = self.dir / "t.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        self.mv.ingest(str(vtt), source_url="https://youtu.be/abc123")
        self.server = McpServer(self.mv)

    def tearDown(self):
        self._tmp.cleanup()

    def test_initialize(self):
        resp = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(resp["result"]["serverInfo"]["name"], "memovox")
        self.assertIn("protocolVersion", resp["result"])

    def test_notification_returns_none(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_tools_list(self):
        resp = self.server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertIn("search_knowledge", names)
        self.assertIn("ingest_video", names)
        self.assertEqual(len(names), 5)

    def test_tools_call_search(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "search_knowledge", "arguments": {"query": "chunk size?"}},
        })
        text = resp["result"]["content"][0]["text"]
        self.assertIn("citations", text)
        self.assertIn("abc123", text)

    def test_tools_call_synthesize_topic(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "synthesize_topic", "arguments": {"topic": "chunk size"}},
        })
        text = resp["result"]["content"][0]["text"]
        # Real synthesis payload (not the old ask() shim): structured fields.
        self.assertIn("consensus_points", text)
        self.assertIn("contradictions", text)

    def test_unknown_method_errors(self):
        resp = self.server.handle({"jsonrpc": "2.0", "id": 9, "method": "bogus"})
        self.assertEqual(resp["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
