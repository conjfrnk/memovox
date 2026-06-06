import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends import (
    backend_status,
    get_embedder,
    get_nli,
    get_visual_embedder,
)
from memovox.backends.embed import HashingEmbedder
from memovox.backends.nli import LexicalNLI
from memovox.errors import BackendUnavailable


class TestVisualEmbedder(unittest.TestCase):
    def test_signature_visual_embedder_is_free_and_deterministic(self):
        emb = get_visual_embedder("auto")
        self.assertEqual(emb.space, "visual_sig")
        sig = [0.1, 0.2, 0.3, 0.4]
        a, b = emb.embed_image(sig), emb.embed_image(sig)
        self.assertEqual(a, b)              # deterministic
        self.assertEqual(a, sig)            # the signature IS the free embedding

    def test_signature_embedder_handles_raw_bytes(self):
        emb = get_visual_embedder("auto")
        vec = emb.embed_image(bytes([0, 128, 255, 64]))
        self.assertEqual(vec, [0.0, 128 / 255.0, 1.0, 64 / 255.0])

    def test_unknown_and_unavailable_visual_embedders(self):
        with self.assertRaises(BackendUnavailable):
            get_visual_embedder("nope")
        # ColPali is opt-in; absent on a bare machine
        from memovox.backends.visual_embed import ColPaliVisualEmbedder
        if not ColPaliVisualEmbedder.is_available():
            with self.assertRaises(BackendUnavailable):
                get_visual_embedder("colpali")

    def test_visual_embedder_in_backend_status(self):
        self.assertIn("visual_embed", backend_status())


class TestNamedVisualDefaults(unittest.TestCase):
    def test_surya_and_qwen_fall_back_to_free_when_absent(self):
        from memovox.backends import get_ocr, get_vlm
        from memovox.backends.base import OCRBackend, VLMBackend
        from memovox.backends.ocr import SuryaOCR
        from memovox.backends.vlm import Qwen25VL

        # auto never raises (free Null/Tesseract fallback), and never resolves Surya/Qwen
        self.assertIsInstance(get_ocr("auto"), OCRBackend)
        self.assertNotIsInstance(get_ocr("auto"), SuryaOCR)
        self.assertIsInstance(get_vlm("auto"), VLMBackend)
        self.assertNotIsInstance(get_vlm("auto"), Qwen25VL)
        # a forced request for the named default raises when its dep is absent
        if not SuryaOCR.is_available():
            with self.assertRaises(BackendUnavailable):
                get_ocr("surya")
        if not Qwen25VL.is_available():
            with self.assertRaises(BackendUnavailable):
                get_vlm("qwen2.5-vl")


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class TestHashingEmbedder(unittest.TestCase):
    def setUp(self):
        self.emb = HashingEmbedder(dim=256)

    def test_deterministic(self):
        a = self.emb.embed_one("the quick brown fox")
        b = self.emb.embed_one("the quick brown fox")
        self.assertEqual(a, b)

    def test_dimension_and_normalized(self):
        v = self.emb.embed_one("hello world knowledge graph")
        self.assertEqual(len(v), 256)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=5)

    def test_similarity_ordering(self):
        base = self.emb.embed_one("neural networks learn representations from data")
        near = self.emb.embed_one("neural networks learn representations from training data")
        far = self.emb.embed_one("the chef cooked a delicious italian pasta dinner")
        self.assertGreater(_cosine(base, near), _cosine(base, far))

    def test_empty_text(self):
        v = self.emb.embed_one("")
        self.assertEqual(len(v), 256)
        self.assertEqual(sum(x * x for x in v), 0.0)


class TestLexicalNLI(unittest.TestCase):
    def setUp(self):
        self.nli = LexicalNLI()

    def test_entailment_when_contained(self):
        premise = "The recommended chunk size is 512 tokens for retrieval."
        hypothesis = "recommended chunk size is 512 tokens"
        res = self.nli.classify(premise, hypothesis)
        self.assertEqual(res.label, "entailment")
        self.assertGreaterEqual(res.entail, 0.5)

    def test_contradiction_on_negation_flip(self):
        premise = "Scaling laws do not hold beyond this regime."
        hypothesis = "scaling laws hold beyond this regime"
        res = self.nli.classify(premise, hypothesis)
        self.assertEqual(res.label, "contradiction")

    def test_neutral_when_unrelated(self):
        res = self.nli.classify("Cats are mammals.", "Quantum entanglement is nonlocal.")
        self.assertEqual(res.label, "neutral")


class TestRegistry(unittest.TestCase):
    def test_auto_embedder_resolves_to_available_backend(self):
        # "auto" resolves by is_available() (package presence); reading .name
        # checks the chosen backend without loading/downloading a model.
        emb = get_embedder("auto")
        self.assertIn(emb.name, {"hashing", "sentence-transformers"})
        # The free fallback is importable and usable offline at the real dim.
        free = get_embedder("hashing")
        self.assertEqual(free.name, "hashing")
        self.assertEqual(len(free.embed_one("x")), 256)

    def test_auto_nli_resolves_to_available_backend(self):
        # "auto" resolves by is_available() (package presence); reading .name
        # checks the chosen backend without loading/downloading a model.
        nli = get_nli("auto")
        self.assertIn(nli.name, {"lexical", "deberta-nli"})
        # The free fallback is importable and produces a valid label offline.
        free = get_nli("lexical")
        self.assertEqual(free.name, "lexical")
        res = free.classify("a b c", "a b")
        self.assertIn(res.label, {"entailment", "neutral", "contradiction"})

    def test_status_shape(self):
        status = backend_status()
        self.assertIn("asr", status)
        self.assertTrue(status["embed"]["hashing"])
        self.assertTrue(status["nli"]["lexical"])


class TestOllamaAvailability(unittest.TestCase):
    """Ollama is reached over HTTP, so availability must reflect a REACHABLE
    server, not merely the binary on PATH. Otherwise `backends` reports it
    available, auto-selects it, and every ask/synthesize/claim-extraction makes a
    failing connect before degrading."""

    def test_binary_present_but_server_down_is_unavailable(self):
        from unittest import mock

        from memovox.backends.llm import OllamaLLM
        from memovox.backends.vlm import OllamaVLM

        with mock.patch("shutil.which", return_value="/usr/local/bin/ollama"), \
                mock.patch("memovox.backends.llm._ping", return_value=False), \
                mock.patch("memovox.backends.vlm._ping", return_value=False):
            self.assertFalse(OllamaLLM.is_available())
            self.assertFalse(OllamaVLM.is_available())

    def test_reachable_server_is_available(self):
        from unittest import mock

        from memovox.backends.llm import OllamaLLM

        with mock.patch("shutil.which", return_value=None), \
                mock.patch("memovox.backends.llm._ping", return_value=True):
            self.assertTrue(OllamaLLM.is_available())


if __name__ == "__main__":
    unittest.main()
