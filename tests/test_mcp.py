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
        self.assertIn("consolidate", names)
        self.assertIn("claim_timeline", names)
        self.assertIn("job_status", names)
        self.assertEqual(len(names), 8)

    def test_claim_timeline_tool(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 12, "method": "tools/call",
            "params": {"name": "claim_timeline", "arguments": {"topic": "chunk size"}},
        })
        self.assertNotIn("error", resp)
        self.assertIsInstance(resp["result"]["content"][0]["text"], str)  # ordered steps JSON

    def test_tools_call_search(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "search_knowledge", "arguments": {"query": "chunk size?"}},
        })
        text = resp["result"]["content"][0]["text"]
        self.assertIn("citations", text)
        self.assertIn("abc123", text)

    def test_search_knowledge_threads_modality(self):
        from unittest import mock

        captured = {}
        real_ask = self.mv.ask

        def _spy(query, **kwargs):
            captured.update(kwargs)
            return real_ask(query, **kwargs)

        with mock.patch.object(self.mv, "ask", side_effect=_spy):
            self.server.handle({
                "jsonrpc": "2.0", "id": 9, "method": "tools/call",
                "params": {"name": "search_knowledge",
                           "arguments": {"query": "the slide", "modality": "visual"}},
            })
        self.assertEqual(captured.get("modality"), "visual")  # threaded, not dropped

    def test_search_knowledge_includes_clips(self):
        # M2.3: clips flow through to_dict() with no schema change
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 11, "method": "tools/call",
            "params": {"name": "search_knowledge", "arguments": {"query": "chunk size?"}},
        })
        text = resp["result"]["content"][0]["text"]
        self.assertIn("clips", text)
        self.assertIn("youtube.com/watch?v=abc123", text)  # ranged deep link

    def test_tools_call_synthesize_topic(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "synthesize_topic", "arguments": {"topic": "chunk size"}},
        })
        text = resp["result"]["content"][0]["text"]
        # Real synthesis payload (not the old ask() shim): structured fields.
        self.assertIn("consensus_points", text)
        self.assertIn("contradictions", text)

    def test_consolidate_is_nonblocking_and_job_status_resolves(self):
        import json
        # M3.3: consolidate enqueues and returns a job handle (does NOT block).
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {"name": "consolidate", "arguments": {}},
        })
        handle = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("job_id", handle)
        self.assertIn(handle["state"], ("queued", "running", "succeeded"))
        # drain the queue, then job_status resolves the result
        from memovox.serving.jobs import JobWorker
        JobWorker(self.mv, once=True).drain()
        st = self.server.handle({
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {"name": "job_status", "arguments": {"job_id": handle["job_id"]}},
        })
        status = json.loads(st["result"]["content"][0]["text"])
        self.assertEqual(status["state"], "succeeded")
        self.assertIn("topics", status["result"])

    def test_ingest_video_is_nonblocking_and_job_status_resolves(self):
        import json
        import time
        # ingest_video must enqueue + return a job handle immediately: a real ingest
        # (download + ASR + NLI) outlasts MCP client timeouts (Claude Desktop cancels
        # at 240 s), so the inline path loses the result even when the ingest succeeds.
        vtt = self.dir / "t2.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 13, "method": "tools/call",
            "params": {"name": "ingest_video",
                       "arguments": {"url": str(vtt), "source_url": "https://youtu.be/def456"}},
        })
        handle = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("job_id", handle)
        self.assertNotIn("n_moments", handle)  # a handle, not an inline report
        self.assertIn(handle["state"], ("queued", "running", "succeeded"))
        # the production path: the auto-spawned worker drains the queue; poll to terminal
        status = None
        deadline = time.time() + 30
        while time.time() < deadline:
            st = self.server.handle({
                "jsonrpc": "2.0", "id": 14, "method": "tools/call",
                "params": {"name": "job_status", "arguments": {"job_id": handle["job_id"]}},
            })
            status = json.loads(st["result"]["content"][0]["text"])
            if status["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "succeeded")
        self.assertEqual(status["result"]["video_id"], "yt:def456")
        self.assertEqual(status["result"]["status"], "ingested")

    def test_unknown_method_errors(self):
        resp = self.server.handle({"jsonrpc": "2.0", "id": 9, "method": "bogus"})
        self.assertEqual(resp["error"]["code"], -32601)

    def test_non_object_request_is_invalid_not_crash(self):
        # a JSON array/number/string is -32600, never an AttributeError DoS
        for bad in ([1, 2, 3], 42, "hello", None):
            resp = self.server.handle(bad)
            self.assertEqual(resp["error"]["code"], -32600)

    def test_missing_required_arg_is_invalid_params(self):
        # search_knowledge needs 'query' — a missing arg is -32602, not -32603
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {"name": "search_knowledge", "arguments": {}},
        })
        self.assertEqual(resp["error"]["code"], -32602)

    def test_serve_stdio_survives_malformed_and_nonobject_lines(self):
        import io
        from memovox.server.mcp import serve_stdio
        stdin = io.StringIO('not json\n[1,2,3]\n{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
        stdout = io.StringIO()
        serve_stdio(self.mv, stdin=stdin, stdout=stdout)  # must not raise
        lines = [l for l in stdout.getvalue().splitlines() if l.strip()]
        self.assertEqual(len(lines), 3)  # parse-error, invalid-request, ping result


if __name__ == "__main__":
    unittest.main()
