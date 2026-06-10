import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import Memovox
from memovox.server.mcp import McpServer, SUPPORTED_PROTOCOL_VERSIONS

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
        self.mv.close()  # stop the auto-spawned job worker BEFORE deleting its store
        self._tmp.cleanup()

    def test_initialize(self):
        resp = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(resp["result"]["serverInfo"]["name"], "memovox")
        self.assertIn("protocolVersion", resp["result"])

    def test_initialize_carries_model_facing_instructions(self):
        # The instructions string is what makes "watch this video" route here
        # without the user naming memovox — it must exist and name the entry tools.
        resp = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        instructions = resp["result"]["instructions"]
        self.assertIn("ingest_video", instructions)
        self.assertIn("search_knowledge", instructions)
        self.assertIn("job_status", instructions)

    def test_initialize_echoes_supported_client_version(self):
        for version in SUPPORTED_PROTOCOL_VERSIONS:
            resp = self.server.handle({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": version},
            })
            self.assertEqual(resp["result"]["protocolVersion"], version)

    def test_initialize_offers_newest_version_on_unknown(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        })
        self.assertEqual(resp["result"]["protocolVersion"], SUPPORTED_PROTOCOL_VERSIONS[0])

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
        self.assertIn("list_videos", names)
        self.assertEqual(len(names), 9)

    def test_list_videos_tool(self):
        import json
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 20, "method": "tools/call",
            "params": {"name": "list_videos", "arguments": {}},
        })
        payload = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["videos"][0]["video_id"], "yt:abc123")

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

    def test_ingest_video_missing_local_file_is_immediate_tool_error(self):
        # Bad input must fail at call time with an actionable message, not enqueue
        # a job whose failure only surfaces after polling job_status.
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 21, "method": "tools/call",
            "params": {"name": "ingest_video", "arguments": {"url": str(self.dir / "nope.vtt")}},
        })
        self.assertNotIn("error", resp)  # tool error, not a protocol error
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("not found", resp["result"]["content"][0]["text"])

    def test_ingest_video_blank_url_is_tool_error(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 22, "method": "tools/call",
            "params": {"name": "ingest_video", "arguments": {"url": "   "}},
        })
        self.assertTrue(resp["result"]["isError"])

    def _wait_terminal(self, job_id):
        import json
        import time
        deadline = time.time() + 30
        while time.time() < deadline:
            st = self.server.handle({
                "jsonrpc": "2.0", "id": 99, "method": "tools/call",
                "params": {"name": "job_status", "arguments": {"job_id": job_id}},
            })
            status = json.loads(st["result"]["content"][0]["text"])
            if status["state"] in ("succeeded", "failed"):
                return status
            time.sleep(0.1)
        self.fail(f"job {job_id} did not reach a terminal state")

    def test_ingest_video_accepts_file_uri(self):
        import json
        # Models routinely hand local files over as file:///path URIs.
        vtt = self.dir / "t4.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 32, "method": "tools/call",
            "params": {"name": "ingest_video", "arguments": {"url": vtt.as_uri()}},
        })
        self.assertFalse(resp["result"]["isError"])
        handle = json.loads(resp["result"]["content"][0]["text"])
        status = self._wait_terminal(handle["job_id"])
        self.assertEqual(status["state"], "succeeded")

    def test_ingest_video_rejects_unsupported_scheme_immediately(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 33, "method": "tools/call",
            "params": {"name": "ingest_video", "arguments": {"url": "ftp://example.com/a.mp4"}},
        })
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("scheme", resp["result"]["content"][0]["text"].lower())

    def test_ingest_video_directory_is_tool_error(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 34, "method": "tools/call",
            "params": {"name": "ingest_video", "arguments": {"url": str(self.dir)}},
        })
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("directory", resp["result"]["content"][0]["text"])

    def test_ingest_video_uppercase_scheme_hits_local_only_guard(self):
        # RFC 3986: schemes are case-insensitive — HTTPS:// must not slip past
        # the local_only refusal and fail late as a "missing local file".
        with Memovox(store=self.dir / "store-lo2", llm_backend="none", local_only=True) as mv:
            server = McpServer(mv)
            resp = server.handle({
                "jsonrpc": "2.0", "id": 35, "method": "tools/call",
                "params": {"name": "ingest_video", "arguments": {"url": "HTTPS://youtu.be/zzz"}},
            })
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("local_only", resp["result"]["content"][0]["text"])

    def test_ingest_video_local_only_refuses_remote_immediately(self):
        with Memovox(store=self.dir / "store-lo", llm_backend="none", local_only=True) as mv:
            server = McpServer(mv)
            resp = server.handle({
                "jsonrpc": "2.0", "id": 23, "method": "tools/call",
                "params": {"name": "ingest_video", "arguments": {"url": "https://youtu.be/zzz"}},
            })
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("local_only", resp["result"]["content"][0]["text"])

    def test_ingest_handle_and_job_status_carry_next_step_hints(self):
        import json
        import time
        vtt = self.dir / "t3.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 24, "method": "tools/call",
            "params": {"name": "ingest_video", "arguments": {"url": str(vtt)}},
        })
        handle = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("job_status", handle["hint"])  # tells the model what to do next
        # Poll to a terminal state (and assert hints along the way) so the background
        # worker is quiet before tearDown removes the temp store.
        status = None
        deadline = time.time() + 30
        while time.time() < deadline:
            st = self.server.handle({
                "jsonrpc": "2.0", "id": 25, "method": "tools/call",
                "params": {"name": "job_status", "arguments": {"job_id": handle["job_id"]}},
            })
            status = json.loads(st["result"]["content"][0]["text"])
            self.assertTrue(status["hint"])  # every state maps to a hint
            if status["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "succeeded")
        self.assertIn("search_knowledge", status["hint"])  # succeeded-ingest hint

    def test_job_status_unknown_id_is_tool_error(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 26, "method": "tools/call",
            "params": {"name": "job_status", "arguments": {"job_id": "nope"}},
        })
        self.assertNotIn("error", resp)
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("ingest_video", resp["result"]["content"][0]["text"])

    def test_search_unknown_video_id_points_to_list_videos(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 27, "method": "tools/call",
            "params": {"name": "search_knowledge",
                       "arguments": {"query": "x", "video_id": "yt:doesnotexist"}},
        })
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("list_videos", resp["result"]["content"][0]["text"])

    def test_search_unknown_modality_is_tool_error(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 28, "method": "tools/call",
            "params": {"name": "search_knowledge",
                       "arguments": {"query": "x", "modality": "audio"}},
        })
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("speech", resp["result"]["content"][0]["text"])

    def test_query_tools_explain_empty_corpus(self):
        with Memovox(store=self.dir / "store-empty", llm_backend="none") as mv:
            server = McpServer(mv)
            for name, arguments in (
                ("search_knowledge", {"query": "anything"}),
                ("synthesize_topic", {"topic": "anything"}),
                ("find_contradictions", {}),
                ("claim_timeline", {"topic": "anything"}),
            ):
                resp = server.handle({
                    "jsonrpc": "2.0", "id": 29, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                })
                self.assertTrue(resp["result"]["isError"], name)
                self.assertIn("ingest_video", resp["result"]["content"][0]["text"], name)

    def test_tool_execution_error_is_iserror_result_not_protocol_error(self):
        # MCP: execution failures belong in the result (isError) where the model can
        # read them and self-correct — not in an opaque -32603 protocol error.
        from unittest import mock
        with mock.patch.object(self.mv, "synthesize", side_effect=RuntimeError("backend exploded")):
            resp = self.server.handle({
                "jsonrpc": "2.0", "id": 30, "method": "tools/call",
                "params": {"name": "synthesize_topic", "arguments": {"topic": "x"}},
            })
        self.assertNotIn("error", resp)
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("backend exploded", resp["result"]["content"][0]["text"])

    def test_unknown_tool_lists_available_tools(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 31, "method": "tools/call",
            "params": {"name": "bogus_tool", "arguments": {}},
        })
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("ingest_video", resp["result"]["content"][0]["text"])

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
