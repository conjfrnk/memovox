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


if __name__ == "__main__":
    unittest.main()
