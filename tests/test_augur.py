import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import augur
from memovox.augur import plan, rrf_fuse
from memovox.backends.embed import HashingEmbedder
from memovox.config import Config, Settings
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.models import Claim


class TestPlanner(unittest.TestCase):
    def test_intents(self):
        self.assertEqual(plan("how did his view change over time?").strategy, "temporal")
        self.assertTrue(plan("where do these sources contradict?").contradiction)
        self.assertEqual(plan("how to install the package").strategy, "procedure")
        self.assertEqual(plan("show me the slide with the diagram").strategy, "visual")
        self.assertEqual(plan("what is attention").strategy, "hybrid")


class TestDecompose(unittest.TestCase):
    def test_single_clause_yields_one_subquery(self):
        from memovox.augur.planner import decompose
        qp = decompose("what is attention")
        self.assertEqual(len(qp.subqueries), 1)
        self.assertEqual(qp.subqueries[0].text, "what is attention")
        self.assertEqual(qp.subqueries[0].strategy, "hybrid")
        self.assertEqual(qp.strategy, "hybrid")  # top-level mirrors the first sub-query

    def test_multipart_splits_on_comma_and(self):
        from memovox.augur.planner import decompose
        qp = decompose("What was the optimal context length, and which model family "
                       "reused the Chinchilla token ratio?")
        self.assertEqual(len(qp.subqueries), 2)
        self.assertIn("context length", qp.subqueries[0].text)
        self.assertIn("model family", qp.subqueries[1].text)

    def test_noun_phrase_and_is_not_split(self):
        from memovox.augur.planner import decompose
        # bare "X and Y" noun-phrase list must NOT split
        qp = decompose("explain scaling laws and compute budgets")
        self.assertEqual(len(qp.subqueries), 1)

    def test_multiple_questions_split(self):
        from memovox.augur.planner import decompose
        qp = decompose("What is the chunk size? Which model reused Chinchilla?")
        self.assertEqual(len(qp.subqueries), 2)

    def test_per_clause_strategy(self):
        from memovox.augur.planner import decompose
        qp = decompose("What changed over time, and what does talk_b dispute?")
        self.assertEqual(qp.subqueries[0].strategy, "temporal")
        self.assertEqual(qp.subqueries[1].strategy, "contradiction")


class TestRRF(unittest.TestCase):
    def test_fusion_prefers_consensus(self):
        a = [("x", 1.0), ("y", 0.5)]
        b = [("y", 1.0), ("z", 0.5)]
        fused = dict(rrf_fuse([a, b], k=60, top_k=3))
        self.assertGreater(fused["y"], fused["x"])
        self.assertGreater(fused["y"], fused["z"])


class TestAsk(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.store.upsert_video(Video("yt:abc", "https://youtu.be/abc", "RAG talk"))
        m0 = Moment("yt:abc#m0000", "yt:abc", 100.0, 130.0,
                    "The recommended chunk size is 512 tokens for retrieval augmented generation.",
                    "spk_0", index=0)
        m1 = Moment("yt:abc#m0001", "yt:abc", 130.0, 160.0,
                    "We trained the model on a cluster of GPUs for several weeks.", "spk_0", index=1)
        for m in (m0, m1):
            self.store.add_moment(m, self.emb.embed_one(m.text_for_embedding()))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_answer_has_citation_and_deeplink(self):
        ans = augur.ask(self.store, "what chunk size is recommended?",
                        embedder=self.emb, settings=Settings(top_k=4))
        self.assertFalse(ans.low_evidence)
        self.assertTrue(ans.citations)
        top = ans.citations[0]
        self.assertEqual(top.moment_id, "yt:abc#m0000")
        self.assertTrue(top.deep_link.startswith("https://youtu.be/abc?t=100"))
        self.assertIn("chunk size", top.snippet.lower())
        self.assertIn("[1]", ans.text)

    def test_empty_store_low_evidence(self):
        with tempfile.TemporaryDirectory() as t2:
            store2 = LoomStore(Config(store=pathlib.Path(t2) / "s").ensure())
            ans = augur.ask(store2, "anything?", embedder=self.emb)
            self.assertTrue(ans.low_evidence)
            self.assertEqual(ans.citations, [])
            store2.close()


class TestGraphLegThroughAsk(unittest.TestCase):
    """M1.2 W1: the §5 graph-retrieval leg fires END-TO-END through mv.ask(), not
    just via hand-built retrieve() unit stores. A gold moment reachable ONLY by a
    CONTRADICTS edge (no embedding, no shared query terms) is surfaced when the
    planner routes the question to the contradiction strategy, and is ABSENT when
    it does not. (NLI-derived edges connect lexically-similar claims, so a robust
    graph-only proof uses a hand-placed edge between dissimilar claims.)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.store.upsert_video(Video("yt:a", "https://youtu.be/a", "talk a"))
        self.store.upsert_video(Video("yt:b", "https://youtu.be/b", "talk b"))
        # SEED (has embedding + matches the question terms)
        a = Moment("yt:a#m0000", "yt:a", 0.0, 30.0,
                   "Photosynthesis stores chemical energy in glucose during the light reactions.",
                   "spk_0", index=0)
        # GOLD (NO embedding, shares no query term incl. stopwords — reachable ONLY
        # by the edge; FTS OR-matches on shared stopwords, so b must avoid them too)
        b = Moment("yt:b#m0000", "yt:b", 0.0, 30.0,
                   "Cellular respiration releases stored fuel within mitochondria.",
                   "spk_1", index=0)
        self.store.add_moment(a, self.emb.embed_one(a.text_for_embedding()))
        self.store.add_moment(b)  # no embedding
        self.store.add_claim(Claim("yt:a#m0000.c00", "yt:a#m0000", "yt:a",
                                   "Photosynthesis stores energy in glucose.",
                                   subject="photosynthesis", t_start_s=0.0, t_end_s=30.0))
        self.store.add_claim(Claim("yt:b#m0000.c00", "yt:b#m0000", "yt:b",
                                   "Respiration breaks glucose back down.",
                                   subject="respiration", t_start_s=0.0, t_end_s=30.0))
        self.store.add_edge("yt:a#m0000.c00", "CONTRADICTS", "yt:b#m0000.c00",
                            src_type="Claim", dst_type="Claim", video_id="yt:a")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _cited(self, answer):
        return {c.moment_id for c in answer.citations}

    def test_graph_leg_surfaces_gold_when_routed_to_contradiction(self):
        # trigger word "contradicts" -> planner routes contradiction -> graph leg ON.
        # Stopword-free query so FTS cannot spuriously OR-match the gold.
        ans = augur.ask(self.store, "contradict photosynthesis glucose energy claim",
                        embedder=self.emb, settings=Settings(top_k=10))
        self.assertIn("yt:b#m0000", self._cited(ans))  # graph-only gold surfaced

    def test_gold_absent_without_contradiction_routing(self):
        # no trigger word -> hybrid strategy -> graph leg OFF -> gold unreachable
        ans = augur.ask(self.store, "describe photosynthesis glucose energy claim",
                        embedder=self.emb, settings=Settings(top_k=10))
        self.assertNotIn("yt:b#m0000", self._cited(ans))


class TestVisualLeg(unittest.TestCase):
    QUERY = "the loss curve diagram"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.settings = Settings(top_k=8)
        self.store.upsert_video(Video("v", "https://x/v", "talk"))
        # text decoys that match the query lexically (the text legs).
        for i in range(1, 4):
            mid = f"v#m{i:04d}"
            text = f"discussion {i} of the loss curve diagram and training dynamics"
            m = Moment(mid, "v", float(i), float(i) + 1, text, "spk_0", index=i)
            self.store.add_moment(m, self.emb.embed_one(m.text_for_embedding()))
        # the slide is VISUAL-ONLY: no text vector (embedding=None) and unrelated
        # text, so neither the dense nor lexical leg can ever surface it — only the
        # visual signature can. This is "knowledge that exists nowhere in the audio".
        slide = Moment("v#m0000", "v", 0.0, 1.0, "completely unrelated chatter xyz",
                       "spk_0", index=0)
        self.store.add_moment(slide, None, visual_embedding=[1.0, 0.0, 0.0, 0.0])
        self.vsig = [0.96, 0.04, 0.0, 0.0]  # image query close to the slide's signature

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_visual_leg_off_by_default_byte_identical(self):
        from memovox.augur.retrieve import retrieve
        base = retrieve(self.store, self.QUERY, embedder=self.emb, settings=self.settings)
        off = retrieve(self.store, self.QUERY, embedder=self.emb, settings=self.settings,
                       use_visual=False, visual_query_vec=self.vsig)
        self.assertEqual(base, off)  # default OFF == passing use_visual=False

    def test_visual_query_fuses_visual_leg(self):
        from memovox.augur.retrieve import retrieve
        off = [m for m, _ in retrieve(self.store, self.QUERY, embedder=self.emb,
                                      settings=self.settings)]
        on = [m for m, _ in retrieve(self.store, self.QUERY, embedder=self.emb,
                                     settings=self.settings, use_visual=True,
                                     visual_query_vec=self.vsig)]
        # without the visual leg the slide is below the text top-k; the leg pulls it in.
        self.assertNotIn("v#m0000", off)
        self.assertIn("v#m0000", on)

    def test_empty_visual_query_skips_leg_gracefully(self):
        from memovox.augur.retrieve import retrieve
        base = retrieve(self.store, self.QUERY, embedder=self.emb, settings=self.settings)
        on = retrieve(self.store, self.QUERY, embedder=self.emb, settings=self.settings,
                      use_visual=True, visual_query_vec=None)  # no image query
        self.assertEqual(base, on)  # leg skipped when no visual query vector


class TestStrategyDrivenRetrieval(unittest.TestCase):
    """The planner's strategy must change WHICH moments come back and HOW
    citations are ordered — not just the decorative ``Answer.strategy`` label."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _add_moment(self, mid, vid, t0, t1, text, idx):
        m = Moment(mid, vid, t0, t1, text, "spk_0", index=idx)
        self.store.add_moment(m, self.emb.embed_one(m.text_for_embedding()))
        return m

    def _build_contradiction_corpus(self):
        """A two-sided contradiction plus enough unrelated distractors that B
        falls OUTSIDE the dense+lexical baseline's top_k — so B can only re-enter
        the result set via the graph leg following A's CONTRADICTS edge.

        Moment A shares the query's scaling/compute terms; moment B shares NONE
        of them (and isn't lexically/densely close enough to beat the
        distractors); A and B carry claims joined by a CONTRADICTS edge.
        """
        self.store.upsert_video(Video("yt:va", "https://youtu.be/va", "Talk A"))
        self.store.upsert_video(Video("yt:vb", "https://youtu.be/vb", "Talk B"))
        self.store.upsert_video(Video("yt:vd", "https://youtu.be/vd", "Distractors"))
        self._add_moment("yt:va#m0000", "yt:va", 10.0, 20.0,
                         "what do they disagree about scaling laws and compute predictions", 0)
        self._add_moment("yt:vb#m0000", "yt:vb", 30.0, 40.0,
                         "totally orthogonal gardening botanical photosynthesis chlorophyll leaves", 0)
        for i in range(5):
            self._add_moment(f"yt:vd#m000{i}", "yt:vd", 100.0 + i * 10, 110.0 + i * 10,
                             f"cooking recipes for soup number {i} unrelated kitchen", i)
        cA = Claim("yt:va#m0000#c0", "yt:va#m0000", "yt:va",
                   "Scaling laws hold predictably.", t_start_s=10.0)
        cB = Claim("yt:vb#m0000#c0", "yt:vb#m0000", "yt:vb",
                   "Empirical curves bend sharply near the frontier.", t_start_s=30.0)
        self.store.add_claim(cA)
        self.store.add_claim(cB)
        self.store.add_edge(cA.claim_id, "CONTRADICTS", cB.claim_id,
                            src_type="Claim", dst_type="Claim")

    # -- contradiction routing -------------------------------------------- #

    def test_contradiction_routes_through_graph_and_cites_both_sides(self):
        self._build_contradiction_corpus()
        ans = augur.ask(self.store, "what do they disagree about scaling?",
                        embedder=self.emb, settings=Settings(top_k=5))
        self.assertEqual(ans.strategy, "contradiction")
        cited = {c.moment_id for c in ans.citations}
        # Both sides of the CONTRADICTS edge are cited: A via lexical/dense, B
        # ONLY reachable through the graph leg (it's outside the baseline top_k).
        self.assertIn("yt:va#m0000", cited)
        self.assertIn("yt:vb#m0000", cited)

    def test_graph_leg_is_strategy_gated_not_always_on(self):
        # SAME store/edge, but a hybrid (non-contradiction) query that retrieves A
        # must NOT surface B — proving the graph leg is gated on the contradiction
        # strategy, not always on. The SAME store yields DIFFERENT moments
        # depending on the planner's strategy.
        self._build_contradiction_corpus()
        ans = augur.ask(self.store, "tell me about scaling laws and compute predictions",
                        embedder=self.emb, settings=Settings(top_k=5))
        self.assertEqual(ans.strategy, "hybrid")
        cited = {c.moment_id for c in ans.citations}
        self.assertIn("yt:va#m0000", cited)
        self.assertNotIn("yt:vb#m0000", cited)

    # -- temporal ordering ------------------------------------------------ #

    def test_temporal_orders_citations_by_published_at(self):
        # Three videos with DISTINCT published_at, inserted out of chronological
        # order so a stable sort can't accidentally produce the right order.
        chrono = [
            ("yt:v2020", "2020-01-01", 50.0),
            ("yt:v2018", "2018-01-01", 50.0),
            ("yt:v2022", "2022-01-01", 50.0),
        ]
        for vid, pub, t0 in chrono:
            self.store.upsert_video(Video(vid, f"https://youtu.be/{vid[3:]}",
                                          f"Talk {vid}", published_at=pub))
            self._add_moment(f"{vid}#m0000", vid, t0, t0 + 10.0,
                             "The recommended approach changed considerably this year.", 0)

        ans = augur.ask(self.store, "how did the recommended approach change over time?",
                        embedder=self.emb, settings=Settings(top_k=5))
        self.assertEqual(ans.strategy, "temporal")
        # Citations ordered ascending by published_at: 2018, 2020, 2022.
        order = [c.video_id for c in ans.citations]
        self.assertEqual(order, ["yt:v2018", "yt:v2020", "yt:v2022"])
        # Indices are re-assigned 1..n in the temporal order.
        self.assertEqual([c.index for c in ans.citations], list(range(1, len(order) + 1)))
        # Synthesis still cites every retrieved citation in the new order.
        for c in ans.citations:
            self.assertIn(f"[{c.index}]", ans.text)

    def test_temporal_missing_published_at_sorts_last(self):
        # Pins the missing-last branch of the sort key (`_published_at(c) == ""`):
        # a video with published_at=None must sort AFTER all dated videos while the
        # dated ones stay in ascending order, with c.index 1..n over the full set.
        dated = [
            ("yt:v2020", "2020-01-01", 50.0),
            ("yt:v2018", "2018-01-01", 50.0),
            ("yt:v2022", "2022-01-01", 50.0),
        ]
        for vid, pub, t0 in dated:
            self.store.upsert_video(Video(vid, f"https://youtu.be/{vid[3:]}",
                                          f"Talk {vid}", published_at=pub))
            self._add_moment(f"{vid}#m0000", vid, t0, t0 + 10.0,
                             "The recommended approach changed considerably this year.", 0)
        # 4th video with NO publish date.
        self.store.upsert_video(Video("yt:vnone", "https://youtu.be/vnone",
                                      "Talk undated", published_at=None))
        self._add_moment("yt:vnone#m0000", "yt:vnone", 50.0, 60.0,
                         "The recommended approach changed considerably this year.", 0)

        ans = augur.ask(self.store, "how did the recommended approach change over time?",
                        embedder=self.emb, settings=Settings(top_k=5))
        self.assertEqual(ans.strategy, "temporal")
        order = [c.video_id for c in ans.citations]
        # All four moments retrieved; dated ones ascending, undated one LAST.
        self.assertEqual(order, ["yt:v2018", "yt:v2020", "yt:v2022", "yt:vnone"])
        # Indices re-assigned 1..n across the full set (including the undated one).
        self.assertEqual([c.index for c in ans.citations], list(range(1, len(order) + 1)))
        for c in ans.citations:
            self.assertIn(f"[{c.index}]", ans.text)


if __name__ == "__main__":
    unittest.main()
