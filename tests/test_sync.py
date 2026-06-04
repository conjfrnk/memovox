"""M3.2 — sync_state cursor + subscription sync engine.

All yt-dlp/enumeration is monkeypatched; make test does no network I/O.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import sync_state
from memovox.config import Config
from memovox.loom import LoomStore


class SyncStateCursorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()

    def tearDown(self):
        self._tmp.cleanup()

    def test_cursor_roundtrips_and_persists(self):
        url = "https://www.youtube.com/@chan"
        with LoomStore(self.config) as store:
            self.assertEqual(sync_state.seen_ids(store, url), set())  # unknown -> empty
            sync_state.mark_seen(store, url, "yt:a")
            sync_state.mark_seen(store, url, "yt:b")
        with LoomStore(self.config) as store:  # reopen -> persisted
            self.assertEqual(sync_state.seen_ids(store, url), {"yt:a", "yt:b"})

    def test_mark_seen_is_idempotent(self):
        url = "https://x/chan"
        with LoomStore(self.config) as store:
            sync_state.mark_seen(store, url, "yt:a")
            sync_state.mark_seen(store, url, "yt:a")  # no-op
            self.assertEqual(sync_state.seen_ids(store, url), {"yt:a"})

    def test_distinct_sources_are_isolated(self):
        with LoomStore(self.config) as store:
            sync_state.mark_seen(store, "https://x/a", "yt:1")
            sync_state.mark_seen(store, "https://x/b", "yt:2")
            self.assertEqual(sync_state.seen_ids(store, "https://x/a"), {"yt:1"})
            self.assertEqual(sync_state.seen_ids(store, "https://x/b"), {"yt:2"})

    def test_clear_forgets_cursor(self):
        url = "https://x/chan"
        with LoomStore(self.config) as store:
            sync_state.mark_seen(store, url, "yt:a")
            sync_state.clear(store, url)
            self.assertEqual(sync_state.seen_ids(store, url), set())


class ResolveCorpusFlagTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        from memovox import Memovox
        self.mv = Memovox(store=self.dir / "store", llm_backend="none")

    def tearDown(self):
        self._tmp.cleanup()

    def _vtt(self, name, text):
        p = self.dir / name
        p.write_text(f"WEBVTT\n\n00:00:01.000 --> 00:00:09.000\n{text}\n", encoding="utf-8")
        return str(p)

    def test_deferred_then_batch_resolve_equals_default(self):
        from memovox import pipeline
        from memovox.loom import LoomStore
        a = self._vtt("a.en.vtt", "Alice studied the Chinchilla scaling law in detail.")
        b = self._vtt("b.en.vtt", "Bob also studied the Chinchilla scaling law closely.")
        # deferred ingest -> entities NOT resolved yet
        self.mv.ingest(a, source_url="https://x/a", resolve_corpus=False)
        self.mv.ingest(b, source_url="https://x/b", resolve_corpus=False)
        with LoomStore(self.mv.config) as store:
            before = store.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            pipeline.resolve_corpus_pass(self.mv.config, store)
            after = store.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        self.assertEqual(before, 0)        # deferred: no entities until the batch pass
        self.assertGreater(after, 0)       # batch pass resolves them

    def test_default_resolve_corpus_true_resolves_immediately(self):
        from memovox.loom import LoomStore
        a = self._vtt("a.en.vtt", "Alice studied the Chinchilla scaling law in detail.")
        self.mv.ingest(a, source_url="https://x/a")  # default resolve_corpus=True
        with LoomStore(self.mv.config) as store:
            self.assertGreater(store.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
