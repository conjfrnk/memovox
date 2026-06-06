"""Bulletproofing: data-integrity — complete redaction, no dangling graph, and
schema-version safety (from the hardening audit, data-integrity dimension)."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.config import Config
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.models import Claim


class DeleteVideoCompletenessTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _count(self, sql, *p):
        return self.store.conn.execute(sql, p).fetchone()[0]

    def test_delete_video_leaves_no_dangling_graph(self):
        s = self.store
        for v in ("yt:a", "yt:b"):
            s.upsert_video(Video(v, f"https://youtu.be/{v}", v))
            s.add_moment(Moment(f"{v}#m0", v, 0.0, 5.0, "scaling laws hold", "spk", index=0))
            s.add_claim(Claim(f"{v}#m0.c0", f"{v}#m0", v, "scaling laws hold", subject="x"))
        # a CROSS-VIDEO contradiction edge stamped with yt:b (points INTO yt:a's claim)
        s.add_edge("yt:b#m0.c0", "CONTRADICTS", "yt:a#m0.c0",
                   src_type="Claim", dst_type="Claim", video_id="yt:b")
        # a SAME_AS speaker edge stamped with yt:a whose endpoints are BOTH speakers
        # (neither a moment nor a claim) — must still be removed on delete.
        s.add_edge("yt:a:spk_0", "SAME_AS", "spk:alice",
                   src_type="Speaker", dst_type="Speaker", video_id="yt:a")
        # entity mentioned only by yt:a, and a topic on yt:a's moment
        s.conn.execute("INSERT INTO entities (entity_id, canonical_name) VALUES ('ent:solo','Solo')")
        s.conn.execute("INSERT INTO mentions (claim_id, entity_id) VALUES ('yt:a#m0.c0','ent:solo')")
        s.conn.execute("INSERT INTO topics (topic_id, label, moment_count) VALUES ('topic:z','z',1)")
        s.set_moment_topic("yt:a#m0", "topic:z")
        s.conn.commit()
        # watermark advanced as if consolidated
        s.set_meta("consolidation_watermark", "999")

        self.assertTrue(s.delete_video("yt:a"))

        # the cross-video edge into yt:a's deleted claim is gone (no dangle)
        self.assertEqual(self._count("SELECT COUNT(*) FROM edges WHERE dst='yt:a#m0.c0'"), 0)
        # the SAME_AS speaker edge stamped with yt:a is gone (both endpoints speakers)
        self.assertEqual(self._count("SELECT COUNT(*) FROM edges WHERE rel='SAME_AS' AND video_id='yt:a'"), 0)
        # the now-mention-less entity and member-less topic are GC'd
        self.assertEqual(self._count("SELECT COUNT(*) FROM entities WHERE entity_id='ent:solo'"), 0)
        self.assertEqual(self._count("SELECT COUNT(*) FROM topics WHERE topic_id='topic:z'"), 0)
        # watermark reset (rowid-reuse safety)
        self.assertEqual(s.get_meta("consolidation_watermark"), "0")
        # yt:b is untouched
        self.assertIsNotNone(s.get_video("yt:b"))
        self.assertEqual(self._count("SELECT COUNT(*) FROM claims WHERE video_id='yt:b'"), 1)


class LocalOnlyEntityEgressTest(unittest.TestCase):
    def test_local_only_forces_offline_null_linker(self):
        # under local_only, entity linking must NOT select the network-egressing
        # WikidataLinker (whose is_available() probes wikidata.org).
        from memovox.backends import get_entity_linker
        from memovox.backends.entity_link import NullLinker
        from memovox.config import Config, Settings
        cfg = Config(settings=Settings(local_only=True))
        self.assertIsInstance(get_entity_linker("auto", config=cfg), NullLinker)
        self.assertIsInstance(get_entity_linker("wikidata", config=cfg), NullLinker)


class SchemaVersionGuardTest(unittest.TestCase):
    def test_refuses_store_from_the_future(self):
        import sqlite3

        from memovox.loom.store import SCHEMA_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(store=pathlib.Path(tmp) / "s").ensure()
            LoomStore(config).close()  # create at current version
            con = sqlite3.connect(str(config.db_path))
            con.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
            con.commit(); con.close()
            with self.assertRaises(RuntimeError):
                LoomStore(config)  # newer schema -> refuse, don't down-stamp


if __name__ == "__main__":
    unittest.main()
