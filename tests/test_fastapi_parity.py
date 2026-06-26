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

        # wall_ms is an explicitly VOLATILE per-call timing field, so two independent
        # ask() invocations never share it; parity is asserted on the SUBSTANTIVE
        # payload with wall_ms scrubbed (both servers emit the same volatile field).
        def scrub(obj):
            if isinstance(obj, dict):
                return {k: scrub(v) for k, v in obj.items() if k != "wall_ms"}
            if isinstance(obj, list):
                return [scrub(v) for v in obj]
            return obj

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
            self.assertEqual(scrub(resp.json()), scrub(pure), path)
        # POST /query parity (substantive payload; volatile wall_ms scrubbed)
        resp = client.post("/query", json={"query": "chunk size?"})
        _, pure, _ = routes.route_query(self.mv, {"query": "chunk size?"})
        self.assertEqual(scrub(resp.json()), scrub(pure))

    def test_blocking_post_does_not_freeze_event_loop(self):
        # A blocking POST route (route_ingest/ask run for seconds-to-minutes) must be
        # offloaded so it does not monopolize the single asyncio event-loop thread and
        # starve every other request — including the trivial GET / health probe.
        if not fastapi_app.is_available():
            self.skipTest("fastapi not installed (free path)")
        try:
            import asyncio
            import time

            import httpx
        except ImportError:
            self.skipTest("httpx not installed")

        orig = routes.route_query
        SLEEP = 0.8
        try:
            def slow_query(mv, body):
                time.sleep(SLEEP)  # stand-in for the real multi-second ask()/ingest()
                return orig(mv, body)
            routes.route_query = slow_query
            app = fastapi_app.build_app(self.mv)

            async def main():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://t") as cl:
                    t0 = time.monotonic()

                    async def do_post():
                        await cl.post("/query", json={"query": "hi"})
                        return time.monotonic() - t0

                    async def do_get():
                        await asyncio.sleep(0.05)  # ensure the POST is in-flight first
                        await cl.get("/")
                        return time.monotonic() - t0

                    return await asyncio.gather(do_post(), do_get())

            post_dt, get_dt = asyncio.run(main())
            # If the loop were blocked, GET would serialize BEHIND the ~0.8s POST. Offloaded,
            # it returns promptly while the POST is still sleeping in the threadpool.
            self.assertLess(get_dt, SLEEP * 0.6,
                            f"GET / was starved by the blocking POST (get={get_dt:.2f}s, post={post_dt:.2f}s)")
        finally:
            routes.route_query = orig


if __name__ == "__main__":
    unittest.main()
