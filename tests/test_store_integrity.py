"""Storage-layer integrity (round 15): write-atomicity + index/source parity.

Covers three defects the adversarial stress panel surfaced on the store-write path —
untouched by the prior query-surface / ReDoS / citation hardening:

* INSERT OR REPLACE on moments/claims CASCADE-wiped derived rows (vectors/claims/mentions).
* moments_fts was created-but-never-backfilled across a no-fts5 -> fts5 store transition.
* a crashed ingest left a partial video that is_unchanged() masked as "unchanged" forever.
"""

from __future__ import annotations

import pathlib
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.config import Config
from memovox.loom.models import Claim, Moment, Video
from memovox.loom.store import LoomStore


def _cfg(tmp):
    return Config(store=pathlib.Path(tmp) / "s").ensure()


class ReaddPreservesChildrenTest(unittest.TestCase):
    """[8] re-adding a moment/claim must UPDATE in place, not REPLACE (cascade-delete)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = LoomStore(_cfg(self._tmp.name))
        self.store.upsert_video(Video(video_id="v", source_url="u", title="t",
                                      content_hash="h", pipeline_version="p"))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_readd_moment_keeps_vectors_and_claims(self):
        self.store.add_moment(Moment(moment_id="v#m0", video_id="v", t_start_s=0, t_end_s=1,
                                     transcript="x", index=0), [1.0, 0.0])
        self.store.add_claim(Claim(claim_id="v#m0.c0", moment_id="v#m0", video_id="v",
                                   text="c", status="committed"))
        self.assertEqual(len(self.store.moment_vectors()), 1)
        self.assertEqual(len(self.store.list_claims(status=None)), 1)
        # re-add the moment WITHOUT an embedding (e.g. a metadata-only update)
        self.store.add_moment(Moment(moment_id="v#m0", video_id="v", t_start_s=0, t_end_s=1,
                                     transcript="x", index=0, topic_id="t1"))
        self.assertEqual(len(self.store.moment_vectors()), 1, "vector must survive a re-add")
        self.assertEqual(len(self.store.list_claims(status=None)), 1, "claims must survive")
        self.assertEqual(self.store.get_moment("v#m0").topic_id, "t1", "update applied")

    def test_readd_claim_keeps_mentions(self):
        self.store.add_moment(Moment(moment_id="v#m0", video_id="v", t_start_s=0, t_end_s=1,
                                     transcript="x", index=0))
        self.store.add_claim(Claim(claim_id="v#m0.c0", moment_id="v#m0", video_id="v",
                                   text="c", status="committed"))
        from memovox.loom.models import Entity
        self.store.upsert_entity(Entity(entity_id="e:1", canonical_name="Thing", type="concept"))
        self.store.link_mention("v#m0.c0", "e:1")
        n_before = self.store.conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
        self.assertEqual(n_before, 1)
        # re-add the claim -> mentions must NOT be cascade-wiped
        self.store.add_claim(Claim(claim_id="v#m0.c0", moment_id="v#m0", video_id="v",
                                   text="c updated", status="committed"))
        n_after = self.store.conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
        self.assertEqual(n_after, 1, "mentions must survive a claim re-add")


class FtsBackfillTest(unittest.TestCase):
    """[4] moments_fts must be backfilled from moments on a no-fts5 -> fts5 transition."""

    def test_reopen_backfills_empty_fts_index(self):
        tmp = tempfile.TemporaryDirectory()
        cfg = _cfg(tmp.name)
        store = LoomStore(cfg)
        if not store.fts:
            store.close(); tmp.cleanup()
            self.skipTest("fts5 not available in this SQLite build")
        store.upsert_video(Video(video_id="v", source_url="u", title="t",
                                 content_hash="h", pipeline_version="p"))
        store.add_moment(Moment(moment_id="v#m0", video_id="v", t_start_s=0, t_end_s=1,
                                transcript="quantum entanglement physics", index=0), [1.0, 0.0])
        self.assertTrue(store.lexical_search("quantum"))
        store.close()
        # simulate a store originally built WITHOUT fts5: index absent, backfill flag unset
        raw = sqlite3.connect(str(cfg.db_path))
        raw.execute("DROP TABLE IF EXISTS moments_fts")
        raw.execute("DELETE FROM meta WHERE key = 'fts_backfilled'")
        raw.commit(); raw.close()
        store2 = LoomStore(cfg)  # _migrate recreates moments_fts + backfills it
        try:
            self.assertTrue(store2.lexical_search("quantum"),
                            "lexical search must find the moment after FTS backfill")
            self.assertGreater(store2.doc_freq("quantum"), 0, "doc_freq must reflect backfill")
        finally:
            store2.close()
            tmp.cleanup()


class PartialIngestMarkerTest(unittest.TestCase):
    """[1] a partial (crashed) ingest must not be masked as 'unchanged' forever."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        from memovox import Memovox
        self.mv = Memovox(store=self.dir / "store", llm_backend="none")
        self.vtt = self.dir / "t.en.vtt"
        self.vtt.write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:09.000\nThe chunk size is 512 tokens.\n",
            encoding="utf-8")

    def tearDown(self):
        self.mv.close()
        self._tmp.cleanup()

    def test_completed_ingest_sets_marker(self):
        rep = self.mv.ingest(str(self.vtt), source_url="https://youtu.be/abc123")
        with LoomStore(self.mv.config) as store:
            self.assertTrue(store.is_ingest_complete(rep.video_id))

    def test_partial_video_is_repaired_not_masked(self):
        rep = self.mv.ingest(str(self.vtt), source_url="https://youtu.be/abc123")
        vid = rep.video_id
        # simulate a crash AFTER the video row + some moments were committed but BEFORE the
        # completion marker: clear the marker and wipe the claims to make it partial.
        with LoomStore(self.mv.config) as store:
            store.conn.execute("DELETE FROM meta WHERE key = ?",
                               ("ingest_complete:" + vid,))
            store.conn.execute("DELETE FROM claims WHERE video_id = ?", (vid,))
            store.conn.commit()
            self.assertTrue(store.is_unchanged(store.get_video(vid)))   # hash still matches
            self.assertFalse(store.is_ingest_complete(vid))            # ...but incomplete
            self.assertEqual(len(store.list_claims(status=None)), 0)   # genuinely partial
        # re-ingest without force: must NOT short-circuit to 'unchanged' — it rebuilds.
        rep2 = self.mv.ingest(str(self.vtt), source_url="https://youtu.be/abc123")
        self.assertEqual(rep2.status, "replaced")
        with LoomStore(self.mv.config) as store:
            self.assertTrue(store.is_ingest_complete(vid))
            self.assertGreater(len(store.list_claims(status=None)), 0, "claims rebuilt")

    def test_legacy_videos_backfilled_complete_on_migration(self):
        rep = self.mv.ingest(str(self.vtt), source_url="https://youtu.be/abc123")
        vid = rep.video_id
        # simulate a pre-marker (legacy) store: drop the marker AND the backfill flag.
        with LoomStore(self.mv.config) as store:
            store.conn.execute("DELETE FROM meta WHERE key IN (?, 'ingest_complete_backfilled')",
                               ("ingest_complete:" + vid,))
            store.conn.commit()
        with LoomStore(self.mv.config) as store:  # _migrate runs the one-time backfill
            self.assertTrue(store.is_ingest_complete(vid),
                            "legacy video must be assumed complete after migration backfill")


if __name__ == "__main__":
    unittest.main()
