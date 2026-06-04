"""M2.1 — the cross-encoder rerank stage (spec §5/§3).

The free default is a deterministic IDENTITY reranker: same candidate set, order
unchanged, so the eval gates stay byte-identical. A cross-encoder is an opt-in
is_available-gated upgrade.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends import Reranker, get_reranker
from memovox.backends.rerank import CrossEncoderReranker, IdentityReranker
from memovox.config import Settings
from memovox.errors import BackendUnavailable


class RerankerInterfaceTest(unittest.TestCase):
    def test_auto_and_none_resolve_to_free_fallback(self):
        self.assertTrue(get_reranker("auto").is_available())
        self.assertIsInstance(get_reranker("none"), IdentityReranker)
        self.assertIsInstance(get_reranker(""), IdentityReranker)
        self.assertIsInstance(get_reranker("identity"), IdentityReranker)

    def test_unknown_backend_raises(self):
        with self.assertRaises(BackendUnavailable):
            get_reranker("bogus")

    def test_identity_rerank_is_unchanged(self):
        r = get_reranker("auto")
        self.assertIsInstance(r, Reranker)
        cands = [("m2", 0.9), ("m1", 0.8), ("m3", 0.7)]
        self.assertEqual(r.rerank("any query", cands), cands)  # same ids, order, scores
        self.assertFalse(r.needs_text)

    def test_settings_default_and_env(self):
        import os
        from unittest import mock
        self.assertEqual(Settings().rerank_backend, "auto")
        with mock.patch.dict(os.environ, {"MEMOVOX_RERANK_BACKEND": "none"}):
            self.assertEqual(Settings.from_env().rerank_backend, "none")


class CrossEncoderGatingTest(unittest.TestCase):
    def test_cross_encoder_is_available_is_classmethod_bool(self):
        self.assertIsInstance(CrossEncoderReranker.is_available(), bool)

    def test_auto_falls_back_to_identity_when_cross_encoder_absent(self):
        if not CrossEncoderReranker.is_available():
            self.assertIsInstance(get_reranker("auto"), IdentityReranker)
            with self.assertRaises(BackendUnavailable):
                get_reranker("cross-encoder")
        self.assertTrue(CrossEncoderReranker.needs_text)


class RerankWiringTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from memovox.backends.embed import HashingEmbedder
        from memovox.config import Config
        from memovox.loom import LoomStore, Moment, Video
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.store.upsert_video(Video("v", "https://x/v", "talk"))
        for i in range(3):
            text = f"chunk {i} about retrieval augmented generation and models"
            m = Moment(f"v#m{i:04d}", "v", float(i), float(i) + 1, text, "spk_0", index=i)
            self.store.add_moment(m, self.emb.embed_one(m.text_for_embedding()))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_identity_default_is_byte_identical_to_no_rerank(self):
        from memovox import augur
        from memovox.config import Settings
        q = "retrieval augmented generation models"
        base = augur.ask(self.store, q, embedder=self.emb, settings=Settings(top_k=5),
                         reranker=None)
        idn = augur.ask(self.store, q, embedder=self.emb, settings=Settings(top_k=5),
                        reranker=get_reranker("identity"))
        self.assertEqual([c.moment_id for c in base.citations],
                         [c.moment_id for c in idn.citations])
        self.assertEqual([c.index for c in base.citations], [c.index for c in idn.citations])

    def test_reranker_reorders_citations_with_contiguous_indices(self):
        from memovox import augur
        from memovox.backends.base import Reranker
        from memovox.config import Settings

        class ReverseReranker(Reranker):
            name = "reverse"
            needs_text = False

            @classmethod
            def is_available(cls):
                return True

            def rerank(self, query, candidates, *, texts=None):
                return list(reversed(candidates))

        q = "retrieval augmented generation models"
        base = augur.ask(self.store, q, embedder=self.emb, settings=Settings(top_k=5))
        rev = augur.ask(self.store, q, embedder=self.emb, settings=Settings(top_k=5),
                        reranker=ReverseReranker())
        self.assertEqual([c.moment_id for c in rev.citations],
                         list(reversed([c.moment_id for c in base.citations])))
        # [n] indices stay contiguous 1..N over the new order
        self.assertEqual([c.index for c in rev.citations],
                         list(range(1, len(rev.citations) + 1)))


if __name__ == "__main__":
    unittest.main()
