import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.embed import HashingEmbedder
from memovox.config import Config
from memovox.loom import Claim, LoomStore, Moment, Video


class LoomTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.video = Video(
            video_id="yt:abc", source_url="https://youtu.be/abc", title="Test talk",
            content_hash="hash1", pipeline_version="0.1.0-phase0",
        )
        self.store.upsert_video(self.video)
        self.m1 = Moment("yt:abc#m0000", "yt:abc", 0.0, 30.0,
                         "neural networks learn via backpropagation and gradients", "spk_0", index=0)
        self.m2 = Moment("yt:abc#m0001", "yt:abc", 30.0, 60.0,
                         "the chef prepared a delicious italian pasta dinner", "spk_0", index=1)
        self.store.add_moment(self.m1, self.emb.embed_one(self.m1.text_for_embedding()))
        self.store.add_moment(self.m2, self.emb.embed_one(self.m2.text_for_embedding()))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class TestVideosMoments(LoomTestBase):
    def test_get_and_list_video(self):
        self.assertEqual(self.store.get_video("yt:abc").title, "Test talk")
        self.assertEqual(len(self.store.list_videos()), 1)

    def test_moments_ordered(self):
        moments = self.store.moments_for_video("yt:abc")
        self.assertEqual([m.index for m in moments], [0, 1])

    def test_visual_embedding_stored_and_counted(self):
        vm = Moment(
            "yt:abc#m0009", "yt:abc", 90.0, 120.0, "a slide is shown", "spk_0",
            visual_caption="bar chart", ocr_text="Loss curve", index=9,
        )
        self.store.add_moment(
            vm, self.emb.embed_one(vm.text_for_embedding()),
            visual_embedding=[0.5, 0.25, 0.75],
        )
        self.assertEqual(self.store.stats()["visual_vectors"], 1)
        self.assertEqual(self.store.get_visual_vector("yt:abc#m0009"), [0.5, 0.25, 0.75])

    def test_visual_embedding_cascades_on_video_delete(self):
        vm = Moment("yt:abc#m0009", "yt:abc", 90.0, 120.0, "slide", "spk_0",
                    ocr_text="Loss curve", index=9)
        self.store.add_moment(vm, self.emb.embed_one("slide Loss curve"),
                              visual_embedding=[0.5, 0.25, 0.75])
        self.store.delete_video("yt:abc")
        self.assertEqual(self.store.stats()["visual_vectors"], 0)

    def test_idempotency_unchanged(self):
        same = Video("yt:abc", "https://youtu.be/abc", "Test talk",
                     content_hash="hash1", pipeline_version="0.1.0-phase0")
        self.assertTrue(self.store.is_unchanged(same))
        changed = Video("yt:abc", None, "Test talk", content_hash="hash2",
                        pipeline_version="0.1.0-phase0")
        self.assertFalse(self.store.is_unchanged(changed))


class TestSearch(LoomTestBase):
    def test_lexical_search(self):
        hits = self.store.lexical_search("backpropagation")
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "yt:abc#m0000")

    def test_vector_search_ranks_relevant_first(self):
        q = self.emb.embed_one("deep learning neural network gradients")
        hits = self.store.vector_search(q, top_k=2)
        self.assertEqual(hits[0][0], "yt:abc#m0000")

    def test_vector_search_dim_mismatch_skipped(self):
        # A 4-dim query cannot match 256-dim stored vectors -> no results, no crash.
        self.assertEqual(self.store.vector_search([0.1, 0.2, 0.3, 0.4]), [])


class TestClaimsGraph(LoomTestBase):
    def test_claim_roundtrip(self):
        claim = Claim(
            claim_id="yt:abc#m0000.c00", moment_id="yt:abc#m0000", video_id="yt:abc",
            text="neural networks learn via backpropagation", subject="neural networks",
            predicate="learn via", object="backpropagation", claim_type="FACT",
            entailment_score=0.9, t_start_s=0.0, t_end_s=30.0, speaker_id="spk_0",
            qualifiers={"hedge": False},
        )
        self.store.add_claim(claim)
        got = self.store.get_claim("yt:abc#m0000.c00")
        self.assertEqual(got.object, "backpropagation")
        self.assertEqual(got.qualifiers, {"hedge": False})
        self.assertEqual(len(self.store.claims_for_video("yt:abc")), 1)

    def test_edges_neighbors(self):
        self.store.add_edge("spk_0", "STATES", "yt:abc#m0000.c00",
                            src_type="Speaker", dst_type="Claim", video_id="yt:abc")
        out = self.store.neighbors("spk_0", rel="STATES")
        self.assertEqual(out[0]["dst"], "yt:abc#m0000.c00")
        # idempotent: inserting the same edge again does not duplicate
        self.store.add_edge("spk_0", "STATES", "yt:abc#m0000.c00", video_id="yt:abc")
        self.assertEqual(len(self.store.neighbors("spk_0", rel="STATES")), 1)


class TestDeleteCascade(LoomTestBase):
    def test_delete_video_cascades(self):
        self.store.add_claim(Claim("yt:abc#m0000.c00", "yt:abc#m0000", "yt:abc", "x"))
        self.assertTrue(self.store.delete_video("yt:abc"))
        self.assertIsNone(self.store.get_moment("yt:abc#m0000"))
        self.assertEqual(self.store.moments_for_video("yt:abc"), [])
        self.assertEqual(self.store.lexical_search("backpropagation"), [])
        self.assertEqual(self.store.vector_search(self.emb.embed_one("neural")), [])
        stats = self.store.stats()
        self.assertEqual(stats["moments"], 0)
        self.assertEqual(stats["claims"], 0)


if __name__ == "__main__":
    unittest.main()
