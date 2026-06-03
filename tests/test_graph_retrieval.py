"""Graph-expansion retrieval leg (W3.2, spec §5).

``retrieve(..., use_graph=True)`` adds a GRAPH leg: from the dense/lexical seed
moments, follow claim->claim edges (SUPPORTS/CONTRADICTS/ELABORATES) to surface
related moments that share NO query terms. The default (``use_graph=False``)
preserves the dense+lexical-only behavior so no existing retrieval test changes.

The store is built by hand with the free hashing embedder so the test is fully
hermetic (no network, no model downloads).
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.augur import retrieve
from memovox.augur.traverse import expand
from memovox.backends.embed import HashingEmbedder
from memovox.config import Config, Settings
from memovox.loom import Claim, LoomStore, Moment, Video

# Moment A's transcript carries the query term ("photosynthesis"); moment B's
# transcript shares NO term with the query, so it can ONLY be reached through the
# claim->claim graph edge linking cA <-> cB.
QUERY = "photosynthesis"
A_MID = "yt:a#m0000"
B_MID = "yt:b#m0000"
A_CID = "yt:a#m0000.c00"
B_CID = "yt:b#m0000.c00"


class _GraphStoreBase(unittest.TestCase):
    rel = "CONTRADICTS"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)

        self.store.upsert_video(Video("yt:a", "https://youtu.be/a", "talk a"))
        self.store.upsert_video(Video("yt:b", "https://youtu.be/b", "talk b"))

        a = Moment(A_MID, "yt:a", 0.0, 30.0,
                   "Photosynthesis converts sunlight into chemical energy.",
                   "spk_0", index=0)
        # No shared term with QUERY="photosynthesis": purely graph-reachable.
        b = Moment(B_MID, "yt:b", 0.0, 30.0,
                   "Mitochondria release stored fuel during cellular respiration.",
                   "spk_1", index=0)
        # A is seeded by dense + lexical (embedding + FTS). B is given NO
        # embedding and shares no query term, so neither the dense nor lexical
        # leg can surface it — its ONLY route into the result set is the graph
        # edge. (The hashing embedder happens to score cos(q, B) == 0, but a
        # zero-similarity row still rides along in vector_search's top_k pool;
        # withholding the embedding makes "graph-only" unambiguous and robust.)
        self.store.add_moment(a, self.emb.embed_one(a.text_for_embedding()))
        self.store.add_moment(b)

        self.store.add_claim(Claim(A_CID, A_MID, "yt:a",
                                   "Photosynthesis stores energy in glucose.",
                                   subject="photosynthesis", t_start_s=0.0, t_end_s=30.0))
        self.store.add_claim(Claim(B_CID, B_MID, "yt:b",
                                   "Respiration breaks glucose back down.",
                                   subject="respiration", t_start_s=0.0, t_end_s=30.0))
        # Edge direction: cA -[rel]-> cB. A seed on the SRC side reaches cB via the
        # OUT edge; a seed on the DST side must reach cA via the IN edge.
        self.store.add_edge(A_CID, self.rel, B_CID,
                            src_type="Claim", dst_type="Claim")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class ExpandTest(_GraphStoreBase):
    def test_follows_edge_from_source_seed(self):
        # Seed is on the SRC side of the edge -> reach B via the OUT edge.
        out = dict(expand(self.store, [A_MID], rels=[self.rel]))
        self.assertIn(B_MID, out)
        self.assertNotIn(A_MID, out)  # seeds are excluded

    def test_follows_edge_from_dest_seed(self):
        # Seed is on the DST side of the edge -> reach A via the IN edge.
        out = dict(expand(self.store, [B_MID], rels=[self.rel]))
        self.assertIn(A_MID, out)
        self.assertNotIn(B_MID, out)

    def test_hop_scoring_is_positive_and_bounded(self):
        out = dict(expand(self.store, [A_MID], rels=[self.rel], hops=1))
        self.assertGreater(out[B_MID], 0.0)
        self.assertLessEqual(out[B_MID], 1.0)

    def test_unrelated_relation_finds_nothing(self):
        self.assertEqual(expand(self.store, [A_MID], rels=["ELABORATES"]), [])


class GraphRetrieveTest(_GraphStoreBase):
    def _settings(self):
        return Settings(top_k=10)

    def test_graph_off_omits_only_linked_moment(self):
        ids = {mid for mid, _ in retrieve(
            self.store, QUERY, embedder=self.emb, settings=self._settings())}
        self.assertIn(A_MID, ids)
        self.assertNotIn(B_MID, ids)

    def test_graph_on_surfaces_only_linked_moment(self):
        ids = {mid for mid, _ in retrieve(
            self.store, QUERY, embedder=self.emb, settings=self._settings(),
            use_graph=True)}
        self.assertIn(A_MID, ids)
        self.assertIn(B_MID, ids)  # surfaced purely via the graph edge

    def test_video_scope_blocks_cross_video_edge(self):
        # A single-video query must stay in-video: B lives in yt:b, so even with
        # the graph leg on, scoping to yt:a must NOT surface it.
        ids = {mid for mid, _ in retrieve(
            self.store, QUERY, embedder=self.emb, settings=self._settings(),
            video_id="yt:a", use_graph=True)}
        self.assertIn(A_MID, ids)
        self.assertNotIn(B_MID, ids)


class MultiHopTest(unittest.TestCase):
    """A 3-moment chain A -> B -> C exercising the BFS frontier advance.

    cA SUPPORTS cB and cB SUPPORTS cC (one claim per moment). From seed A,
    ``hops=2`` must reach BOTH B (1 hop) and C (2 hops), and C must score lower
    than B because it sits a hop farther out.
    """

    REL = "SUPPORTS"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.store.upsert_video(Video("yt:c", "https://youtu.be/c", "chain"))

        self.mids = [f"yt:c#m{i:04d}" for i in range(3)]
        self.cids = [f"{m}.c00" for m in self.mids]
        texts = ["alpha", "bravo", "charlie"]
        for i, (mid, cid) in enumerate(zip(self.mids, self.cids)):
            self.store.add_moment(Moment(mid, "yt:c", float(i), float(i) + 1.0,
                                         texts[i], "spk_0", index=i))
            self.store.add_claim(Claim(cid, mid, "yt:c", texts[i],
                                       t_start_s=float(i), t_end_s=float(i) + 1.0))
        # cA -> cB -> cC
        self.store.add_edge(self.cids[0], self.REL, self.cids[1],
                            src_type="Claim", dst_type="Claim")
        self.store.add_edge(self.cids[1], self.REL, self.cids[2],
                            src_type="Claim", dst_type="Claim")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_two_hops_reach_b_and_c_with_decaying_score(self):
        out = dict(expand(self.store, [self.mids[0]], rels=[self.REL], hops=2))
        self.assertIn(self.mids[1], out)   # B at hop 1
        self.assertIn(self.mids[2], out)   # C at hop 2
        self.assertNotIn(self.mids[0], out)  # seed excluded
        self.assertLess(out[self.mids[2]], out[self.mids[1]])  # farther hop scores lower

    def test_one_hop_stops_at_b(self):
        # With hops=1 the frontier never advances past B, so C is unreachable.
        out = dict(expand(self.store, [self.mids[0]], rels=[self.REL], hops=1))
        self.assertIn(self.mids[1], out)
        self.assertNotIn(self.mids[2], out)

    def test_seed_with_no_edges_returns_empty(self):
        # C is a leaf for OUT edges and a sink with one IN edge; a moment whose
        # claim has NO edges at all yields nothing. Add an isolated moment.
        iso_mid = "yt:c#m0009"
        self.store.add_moment(Moment(iso_mid, "yt:c", 9.0, 10.0, "lonely",
                                     "spk_0", index=9))
        self.store.add_claim(Claim(f"{iso_mid}.c00", iso_mid, "yt:c", "lonely",
                                   t_start_s=9.0, t_end_s=10.0))
        self.assertEqual(expand(self.store, [iso_mid], rels=[self.REL], hops=2), [])


if __name__ == "__main__":
    unittest.main()
