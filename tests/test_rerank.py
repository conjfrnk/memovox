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


if __name__ == "__main__":
    unittest.main()
