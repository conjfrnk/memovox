"""M3.1 — decay & versioning.

Recency is a default-OFF retrieval signal: ``decay_enabled=False`` is byte-identical
to today, and an all-undated corpus stays byte-identical even when ON (every recency
multiplier is 1.0). Supersede lineage is a first-class read; nothing is deleted (§2).
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.augur.retrieve import retrieve
from memovox.backends.embed import HashingEmbedder
from memovox.config import Config, Settings
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.models import Claim


class DecayTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _add(self, vid, url, title, published_at, text):
        self.store.upsert_video(Video(vid, url, title, published_at=published_at))
        m = Moment(f"{vid}#m0000", vid, 0.0, 10.0, text, "spk", index=0)
        self.store.add_moment(m, self.emb.embed_one(m.text_for_embedding()))
        return m.moment_id

    def test_recency_weight_shared_model(self):
        from memovox.loom.consensus import _RECENCY_HALFLIFE_DAYS, recency_weight
        self.assertEqual(recency_weight(None, "2026-01-01", default=1.0), 1.0)  # missing -> default
        self.assertEqual(recency_weight("2026-01-01", "2026-01-01"), 1.0)       # same day
        # one half-life older -> half the weight
        w = recency_weight("2025-01-01", "2026-01-01", halflife=365.0)
        self.assertAlmostEqual(w, 0.5, places=2)

    def test_decay_off_is_byte_identical(self):
        q = "scaling laws and model performance over the years"
        self._add("yt:a", "https://youtu.be/a", "a", "2024-01-01",
                  "scaling laws and model performance discussed at length")
        self._add("yt:b", "https://youtu.be/b", "b", "2026-01-01",
                  "scaling laws and model performance revisited recently")
        base = retrieve(self.store, q, embedder=self.emb, settings=Settings(decay_enabled=False))
        again = retrieve(self.store, q, embedder=self.emb, settings=Settings())  # default OFF
        self.assertEqual(base, again)

    def test_decay_on_reweights_recent_first(self):
        q = "scaling laws model performance"
        old = self._add("yt:a", "https://youtu.be/a", "a", "2020-01-01",
                        "scaling laws model performance the old account")
        new = self._add("yt:b", "https://youtu.be/b", "b", "2026-01-01",
                        "scaling laws model performance the recent account")
        off = [m for m, _ in retrieve(self.store, q, embedder=self.emb,
                                      settings=Settings(decay_enabled=False))]
        on = [m for m, _ in retrieve(self.store, q, embedder=self.emb,
                                     settings=Settings(decay_enabled=True))]
        self.assertEqual(set(off), set(on))           # same set, only re-weighted
        self.assertEqual(on[0], new)                  # newer ranks first under decay
        self.assertLess(on.index(new), on.index(old))

    def test_undated_corpus_byte_identical_even_when_on(self):
        q = "scaling laws model performance"
        self._add("yt:a", "https://youtu.be/a", "a", None, "scaling laws model performance a")
        self._add("yt:b", "https://youtu.be/b", "b", None, "scaling laws model performance b")
        off = retrieve(self.store, q, embedder=self.emb, settings=Settings(decay_enabled=False))
        on = retrieve(self.store, q, embedder=self.emb, settings=Settings(decay_enabled=True))
        self.assertEqual(off, on)  # all undated -> every multiplier 1.0 -> identical

    def test_claim_history_preserves_all_versions(self):
        self.store.upsert_video(Video("yt:a", "https://youtu.be/a", "a"))
        self.store.add_moment(Moment("yt:a#m0", "yt:a", 0.0, 5.0, "x", "spk", index=0))
        for i, cid in enumerate(["yt:a#m0.c0", "yt:a#m0.c1", "yt:a#m0.c2"]):
            self.store.add_claim(Claim(cid, "yt:a#m0", "yt:a", f"v{i}", subject="x"))
        self.store.supersede_claim("yt:a#m0.c0", "yt:a#m0.c1")
        self.store.supersede_claim("yt:a#m0.c1", "yt:a#m0.c2")
        for anchor in ("yt:a#m0.c0", "yt:a#m0.c1", "yt:a#m0.c2"):
            hist = self.store.claim_history(anchor)
            self.assertEqual([c.claim_id for c in hist],
                             ["yt:a#m0.c0", "yt:a#m0.c1", "yt:a#m0.c2"])  # full lineage from any id
        self.assertEqual(self.store.claim_history("nope"), [])

    def test_superseded_only_moment_demoted(self):
        q = "the foundational claim about retrieval"
        keep = self._add("yt:a", "https://youtu.be/a", "a", None,
                         "the foundational claim about retrieval stands")
        # a moment whose only claim is superseded
        self.store.upsert_video(Video("yt:b", "https://youtu.be/b", "b"))
        mb = Moment("yt:b#m0000", "yt:b", 0.0, 10.0,
                    "the foundational claim about retrieval is outdated", "spk", index=0)
        self.store.add_moment(mb, self.emb.embed_one(mb.text_for_embedding()))
        self.store.add_claim(Claim("yt:b#m0000.c0", "yt:b#m0000", "yt:b", "old", subject="x"))
        self.store.add_claim(Claim("yt:b#m0000.c1", "yt:b#m0000", "yt:b", "new", subject="x"))
        self.store.supersede_claim("yt:b#m0000.c0", "yt:b#m0000.c1")
        # now supersede the survivor too -> moment fully superseded
        self.store.add_claim(Claim("yt:b#m0000.c2", "yt:b#m0000", "yt:b", "newer", subject="x"))
        self.store.supersede_claim("yt:b#m0000.c1", "yt:b#m0000.c2")
        self.store.conn.execute("UPDATE claims SET status='superseded' WHERE claim_id='yt:b#m0000.c2'")
        self.store.conn.commit()
        off = [m for m, _ in retrieve(self.store, q, embedder=self.emb,
                                      settings=Settings(decay_enabled=False))]
        on = [m for m, _ in retrieve(self.store, q, embedder=self.emb,
                                     settings=Settings(decay_enabled=True))]
        self.assertIn("yt:b#m0000", off)       # present without decay
        self.assertNotIn("yt:b#m0000", on)     # demoted (excluded) under decay
        self.assertIn(keep, on)


if __name__ == "__main__":
    unittest.main()
