"""W2 — cross-corpus claim clustering + consensus scoring (Phase 3, spec §4.7).

Two pieces:
  * ``cluster_claims`` groups semantically-equivalent committed claims ACROSS
    the corpus (content-token Jaccard, free + deterministic) and emits a
    provenanced cross-video ``SUPPORTS`` edge per agreeing pair — the agreement
    half of "contradiction & agreement detection".
  * ``score_consensus`` turns a cluster into a [0,1] confidence estimate weighted
    by source count × recency × speaker authority (salience proxy).
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.embed import HashingEmbedder
from memovox.config import Config
from memovox.loom import Claim, LoomStore, Moment, Video
from memovox.loom.consensus import ClaimCluster, cluster_claims, score_consensus


def _claim(cid, vid, text, *, salience=0.5, moment=None):
    return Claim(claim_id=cid, moment_id=moment or f"{vid}#m0000", video_id=vid,
                 text=text, subject=text.split(".")[0], salience=salience,
                 t_start_s=0.0, t_end_s=10.0, speaker_id="spk_0")


class TestScoreConsensus(unittest.TestCase):
    """score_consensus in isolation — pure function over a constructed cluster."""

    def _cluster(self, claims, dates):
        return ClaimCluster(claims=claims, dates=dates)

    def test_more_sources_scores_higher(self):
        one = self._cluster([_claim("a.c0", "vid:a", "x")], {"vid:a": None})
        two = self._cluster([_claim("a.c0", "vid:a", "x"),
                             _claim("b.c0", "vid:b", "x")],
                            {"vid:a": None, "vid:b": None})
        self.assertGreater(score_consensus(two), score_consensus(one))

    def test_higher_authority_scores_higher(self):
        low = self._cluster([_claim("a.c0", "vid:a", "x", salience=0.1)], {"vid:a": None})
        high = self._cluster([_claim("a.c0", "vid:a", "x", salience=0.9)], {"vid:a": None})
        self.assertGreater(score_consensus(high), score_consensus(low))

    def test_newer_date_scores_higher(self):
        old = self._cluster([_claim("a.c0", "vid:a", "x")], {"vid:a": "2020-01-01"})
        new = self._cluster([_claim("a.c0", "vid:a", "x")], {"vid:a": "2026-01-01"})
        ref = "2026-01-01"
        self.assertGreater(score_consensus(new, reference_date=ref),
                           score_consensus(old, reference_date=ref))

    def test_recency_neutral_without_dates(self):
        # No publish dates (the free golden path) -> still a real, deterministic
        # score in [0,1]; recency term is neutral, not a crash or a zero.
        c = self._cluster([_claim("a.c0", "vid:a", "x")], {"vid:a": None})
        s = score_consensus(c)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_score_in_unit_range(self):
        c = self._cluster([_claim("a.c0", "vid:a", "x", salience=1.0),
                           _claim("b.c0", "vid:b", "x", salience=1.0),
                           _claim("c.c0", "vid:c", "x", salience=1.0)],
                          {"vid:a": "2026-01-01", "vid:b": "2026-01-01", "vid:c": "2026-01-01"})
        s = score_consensus(c, reference_date="2026-01-01")
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


class ClusterTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        for vid, pub in (("vid:a", "2024-01-01"), ("vid:b", "2026-01-01")):
            self.store.upsert_video(Video(video_id=vid, source_url=f"https://x/{vid}",
                                          title=vid, content_hash=vid, published_at=pub))

    def _add(self, cid, vid, text, *, salience=0.5):
        # A moment is required (claims FK to moments).
        mid = f"{vid}#m{cid[-1]}"
        self.store.add_moment(Moment(mid, vid, 0.0, 10.0, text, "spk_0", index=int(cid[-1])),
                              self.emb.embed_one(text))
        c = Claim(claim_id=cid, moment_id=mid, video_id=vid, text=text,
                  subject=text, salience=salience, t_start_s=0.0, t_end_s=10.0,
                  speaker_id="spk_0", status="committed")
        self.store.add_claim(c)
        return c

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class TestClusterClaims(ClusterTestBase):
    def test_equivalent_claims_cluster_across_videos(self):
        self._add("a.0", "vid:a", "Scaling laws hold beyond current compute budgets.")
        self._add("b.0", "vid:b", "Scaling laws hold beyond current compute budgets.")
        self._add("a.1", "vid:a", "The chef cooked delicious pasta for dinner tonight.")
        clusters = cluster_claims(self.store)
        # The two equivalent scaling claims form one 2-source cluster.
        multi = [c for c in clusters if c.support_count >= 2]
        self.assertEqual(len(multi), 1)
        self.assertEqual({cl.video_id for cl in multi[0].claims}, {"vid:a", "vid:b"})
        # The cooking claim is its own singleton cluster.
        self.assertTrue(any(c.support_count == 1 and "pasta" in c.representative.lower()
                            for c in clusters))

    def test_supports_edge_emitted_cross_video(self):
        a = self._add("a.0", "vid:a", "Scaling laws hold beyond current compute budgets.")
        b = self._add("b.0", "vid:b", "Scaling laws hold beyond current compute budgets.")
        cluster_claims(self.store, write_edges=True)
        out = self.store.neighbors(a.claim_id, rel="SUPPORTS")
        self.assertIn(b.claim_id, {e["dst"] for e in out} |
                      {e["src"] for e in self.store.neighbors(a.claim_id, rel="SUPPORTS",
                                                              direction="in")})
        # provenance present on the edge
        edges = self.store.edges(rel="SUPPORTS")
        self.assertTrue(edges)
        self.assertEqual(edges[0]["src_type"], "Claim")
        self.assertEqual(edges[0]["dst_type"], "Claim")

    def test_no_supports_within_single_video(self):
        # Two equivalent claims in the SAME video cluster, but agreement (SUPPORTS)
        # is a CROSS-corpus signal — no within-video SUPPORTS edge is emitted.
        self._add("a.0", "vid:a", "Scaling laws hold beyond current compute budgets.")
        self._add("a.1", "vid:a", "Scaling laws hold beyond current compute budgets.")
        cluster_claims(self.store, write_edges=True)
        self.assertEqual(self.store.edges(rel="SUPPORTS"), [])

    def test_consensus_populated_on_returned_clusters(self):
        self._add("a.0", "vid:a", "Scaling laws hold beyond current compute budgets.", salience=0.8)
        self._add("b.0", "vid:b", "Scaling laws hold beyond current compute budgets.", salience=0.8)
        clusters = cluster_claims(self.store)
        multi = [c for c in clusters if c.support_count >= 2][0]
        self.assertGreater(multi.consensus, 0.0)
        self.assertLessEqual(multi.consensus, 1.0)

    def test_idempotent(self):
        self._add("a.0", "vid:a", "Scaling laws hold beyond current compute budgets.")
        self._add("b.0", "vid:b", "Scaling laws hold beyond current compute budgets.")
        cluster_claims(self.store, write_edges=True)
        n1 = len(self.store.edges(rel="SUPPORTS"))
        cluster_claims(self.store, write_edges=True)
        n2 = len(self.store.edges(rel="SUPPORTS"))
        self.assertEqual(n1, n2)


if __name__ == "__main__":
    unittest.main()
