import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends import backend_status, get_embedder, get_nli
from memovox.backends.embed import HashingEmbedder
from memovox.backends.nli import LexicalNLI


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
    def test_auto_embedder_is_available(self):
        emb = get_embedder("auto")
        self.assertTrue(emb.embed(["a", "b"]))

    def test_auto_nli_is_available(self):
        nli = get_nli("auto")
        self.assertIsNotNone(nli.classify("a b c", "a b"))

    def test_status_shape(self):
        status = backend_status()
        self.assertIn("asr", status)
        self.assertTrue(status["embed"]["hashing"])
        self.assertTrue(status["nli"]["lexical"])


if __name__ == "__main__":
    unittest.main()
