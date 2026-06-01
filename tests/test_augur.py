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


class TestPlanner(unittest.TestCase):
    def test_intents(self):
        self.assertEqual(plan("how did his view change over time?").strategy, "temporal")
        self.assertTrue(plan("where do these sources contradict?").contradiction)
        self.assertEqual(plan("how to install the package").strategy, "procedure")
        self.assertEqual(plan("show me the slide with the diagram").strategy, "visual")
        self.assertEqual(plan("what is attention").strategy, "hybrid")


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


if __name__ == "__main__":
    unittest.main()
