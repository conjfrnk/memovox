"""Tests for the eval harness (W0.3).

Two layers:
  * focused unit tests for each pure-stdlib metric on tiny synthetic inputs
    (so the metrics themselves are trustworthy, independent of the pipeline);
  * an end-to-end run over the golden corpus, asserting the gates that are
    meaningful *today* (retrieval hit_rate, groundedness) and the *presence*
    of the "later" metrics (entity_f1, der, contradiction).

Written as ``unittest.TestCase`` subclasses so both ``make test``
(unittest discover) and ``pytest -q`` collect them.
"""

import math
import unittest

from eval.harness import (
    _answer_groundedness,
    _check_thresholds,
    clustering_f1,
    contradiction_pr,
    groundedness,
    hit_rate,
    mrr,
    ndcg,
    run_eval,
)


class TestRetrievalMetrics(unittest.TestCase):
    def test_hit_rate_basic(self):
        # Two queries; first has a relevant id in top-k, second does not.
        per_query = [
            (["a", "b", "c"], {"c"}),   # hit (c in top-3)
            (["x", "y"], {"z"}),        # miss
        ]
        self.assertAlmostEqual(hit_rate(per_query, k=3), 0.5)

    def test_hit_rate_respects_k(self):
        # Relevant id is at rank 4; with k=3 it's a miss, with k=4 a hit.
        per_query = [(["a", "b", "c", "d"], {"d"})]
        self.assertEqual(hit_rate(per_query, k=3), 0.0)
        self.assertEqual(hit_rate(per_query, k=4), 1.0)

    def test_hit_rate_empty(self):
        self.assertEqual(hit_rate([], k=5), 0.0)
        self.assertEqual(hit_rate([([], set())], k=5), 0.0)

    def test_mrr(self):
        per_query = [
            (["a", "b", "c"], {"b"}),   # first relevant at rank 2 -> 1/2
            (["x", "y"], {"x"}),        # rank 1 -> 1/1
            (["p", "q"], {"z"}),        # none -> 0
        ]
        self.assertAlmostEqual(mrr(per_query), (0.5 + 1.0 + 0.0) / 3)

    def test_mrr_empty(self):
        self.assertEqual(mrr([]), 0.0)

    def test_ndcg_perfect_vs_worse(self):
        # Single relevant item at rank 1 -> nDCG = 1.0.
        perfect = [(["a", "b", "c"], {"a"})]
        self.assertAlmostEqual(ndcg(perfect, k=3), 1.0)
        # Relevant item at rank 2: DCG = 1/log2(3), IDCG = 1 -> 1/log2(3).
        worse = [(["a", "b", "c"], {"b"})]
        self.assertAlmostEqual(ndcg(worse, k=3), 1.0 / math.log2(3))

    def test_ndcg_empty(self):
        self.assertEqual(ndcg([], k=5), 0.0)
        self.assertEqual(ndcg([([], set())], k=5), 0.0)


class TestClusteringF1(unittest.TestCase):
    def test_perfect_match(self):
        gold = [{"a", "b"}, {"c"}]
        pred = [{"a", "b"}, {"c"}]
        p, r, f1 = clustering_f1(pred, gold)
        self.assertEqual((p, r, f1), (1.0, 1.0, 1.0))

    def test_all_singletons_against_one_cluster(self):
        # Gold says {a,b} same; pred splits them -> 1 gold pair, 0 found.
        gold = [{"a", "b"}]
        pred = [{"a"}, {"b"}]
        p, r, f1 = clustering_f1(pred, gold)
        # No predicted same-cluster pairs at all -> precision defined as 0,
        # recall 0 (the one gold pair was missed).
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)

    def test_known_split(self):
        # gold: {a,b,c} -> pairs {ab, ac, bc} (3 positive pairs)
        # pred: {a,b},{c} -> predicts only {ab} as same-cluster.
        # precision = 1/1 = 1.0 ; recall = 1/3 ; f1 = 2*1*(1/3)/(1+1/3)=0.5
        gold = [{"a", "b", "c"}]
        pred = [{"a", "b"}, {"c"}]
        p, r, f1 = clustering_f1(pred, gold)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0 / 3.0)
        self.assertAlmostEqual(f1, 0.5)

    def test_empty(self):
        self.assertEqual(clustering_f1([], []), (0.0, 0.0, 0.0))


class TestContradictionPR(unittest.TestCase):
    def test_perfect(self):
        found = [("talk_a", "talk_b")]
        gold = [("talk_a", "talk_b")]
        out = contradiction_pr(found, gold)
        self.assertEqual((out["precision"], out["recall"], out["f1"]), (1.0, 1.0, 1.0))

    def test_order_insensitive(self):
        found = [("talk_b", "talk_a")]
        gold = [("talk_a", "talk_b")]
        out = contradiction_pr(found, gold)
        self.assertEqual(out["recall"], 1.0)
        self.assertEqual(out["precision"], 1.0)

    def test_false_positive(self):
        found = [("talk_a", "talk_b"), ("talk_a", "talk_c")]
        gold = [("talk_a", "talk_b")]
        out = contradiction_pr(found, gold)
        self.assertAlmostEqual(out["precision"], 0.5)
        self.assertAlmostEqual(out["recall"], 1.0)

    def test_empty(self):
        out = contradiction_pr([], [])
        self.assertEqual((out["precision"], out["recall"], out["f1"]), (0.0, 0.0, 0.0))
        # Gold present but nothing found -> recall 0, no crash.
        out2 = contradiction_pr([], [("talk_a", "talk_b")])
        self.assertEqual(out2["recall"], 0.0)


class _StubNLI:
    """Deterministic NLI stub: entailed iff hypothesis tokens subset of premise.

    Uses the same word tokenizer as production so punctuation / bracket markers
    are ignored, matching how the real lexical NLI treats content words.
    """

    def classify(self, premise, hypothesis):
        from memovox.backends.base import NLIResult
        from memovox.util import tokenize

        p = set(tokenize(premise))
        h = tokenize(hypothesis)
        if h and all(w in p for w in h):
            return NLIResult("entailment", 1.0, 0.0, 0.0)
        return NLIResult("neutral", 0.0, 1.0, 0.0)


class TestGroundedness(unittest.TestCase):
    def test_all_grounded(self):
        # Each sentence's words appear in the premise -> 100% grounded.
        sentences = ["the cat sat", "on the mat"]
        premise = "the cat sat on the mat"
        self.assertEqual(groundedness(sentences, premise, _StubNLI()), 1.0)

    def test_half_grounded(self):
        sentences = ["the cat sat", "dogs fly away"]
        premise = "the cat sat on the mat"
        self.assertAlmostEqual(groundedness(sentences, premise, _StubNLI()), 0.5)

    def test_empty(self):
        self.assertEqual(groundedness([], "premise", _StubNLI()), 0.0)
        self.assertEqual(groundedness(["a"], "", _StubNLI()), 0.0)


class _FakeCitation:
    def __init__(self, index, snippet):
        self.index = index
        self.snippet = snippet
        self.moment_id = f"m{index}"


class _FakeAnswer:
    def __init__(self, text, citations):
        self.text = text
        self.citations = citations


class TestAnswerGroundedness(unittest.TestCase):
    """The conservative per-citation grounding used over the corpus.

    The synthesizer places the ``[n]`` marker before each sentence's period, so
    ``split_sentences`` yields one sentence per citation (e.g. ``"Cats sit [1]."``).
    """

    def test_cited_sentence_grounded_by_its_own_span(self):
        ans = _FakeAnswer(
            "Cats sit [1]. Dogs run [2].",
            [_FakeCitation(1, "the cats sit on the mat"),
             _FakeCitation(2, "the dogs run fast in the park")],
        )
        # Both sentences are entailed by the span they cite -> 1.0.
        self.assertEqual(_answer_groundedness(ans, _StubNLI(), store=None, threshold=0.5), 1.0)

    def test_uncited_sentence_is_not_grounded(self):
        # Second sentence carries NO [n] marker -> counts as not grounded,
        # even though its words appear in citation [1]'s span.
        ans = _FakeAnswer(
            "Cats sit [1]. Cats sit.",
            [_FakeCitation(1, "the cats sit on the mat")],
        )
        self.assertEqual(_answer_groundedness(ans, _StubNLI(), store=None, threshold=0.5), 0.5)

    def test_sentence_not_entailed_by_its_span(self):
        ans = _FakeAnswer(
            "Elephants fly south [1].",
            [_FakeCitation(1, "the cat sat on the mat")],
        )
        self.assertEqual(_answer_groundedness(ans, _StubNLI(), store=None, threshold=0.5), 0.0)

    def test_no_citations_at_all(self):
        ans = _FakeAnswer("Cats sit. Dogs run.", [])
        self.assertEqual(_answer_groundedness(ans, _StubNLI(), store=None, threshold=0.5), 0.0)


class TestThresholdGates(unittest.TestCase):
    def _report(self, hit_rate_v, groundedness_v, contradiction_f1):
        return {
            "retrieval": {"hit_rate": hit_rate_v, "mrr": 0.0, "ndcg": 0.0, "k": 5},
            "groundedness": groundedness_v,
            "entity_f1": 0.0,
            "der": 0.0,
            "contradiction": {"precision": 0.0, "recall": 0.0, "f1": contradiction_f1},
        }

    def test_all_pass(self):
        self.assertEqual(_check_thresholds(self._report(1.0, 1.0, 1.0)), [])

    def test_contradiction_gate_fails(self):
        failures = _check_thresholds(self._report(1.0, 1.0, 0.0))
        self.assertEqual(len(failures), 1)
        self.assertIn("contradiction.f1", failures[0])

    def test_retrieval_and_groundedness_gates_fail(self):
        failures = _check_thresholds(self._report(0.0, 0.0, 1.0))
        self.assertEqual(len(failures), 2)
        self.assertTrue(any("hit_rate" in f for f in failures))
        self.assertTrue(any("groundedness" in f for f in failures))

    def test_entity_f1_and_der_are_ungated(self):
        # entity_f1/der at 0.0 must NOT trip a gate (legit until W2.3/W4.1).
        self.assertEqual(_check_thresholds(self._report(1.0, 1.0, 1.0)), [])


class TestRunEvalGoldenCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # End-to-end on the golden corpus, free stack pinned inside run_eval.
        cls.report = run_eval()

    def test_report_schema(self):
        r = self.report
        self.assertIn("retrieval", r)
        self.assertIn("hit_rate", r["retrieval"])
        self.assertIn("mrr", r["retrieval"])
        self.assertIn("ndcg", r["retrieval"])
        self.assertIn("k", r["retrieval"])
        self.assertIn("groundedness", r)
        self.assertIn("entity_f1", r)
        self.assertIn("der", r)
        self.assertIn("contradiction", r)
        self.assertIn("precision", r["contradiction"])
        self.assertIn("recall", r["contradiction"])
        self.assertIn("f1", r["contradiction"])

    def test_metrics_are_real_numbers(self):
        r = self.report
        for v in (r["retrieval"]["hit_rate"], r["retrieval"]["mrr"],
                  r["retrieval"]["ndcg"], r["groundedness"],
                  r["entity_f1"], r["der"]):
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_gates_today(self):
        # The only gates meaningful at W0.3.
        self.assertGreaterEqual(self.report["retrieval"]["hit_rate"], 0.6)
        self.assertGreaterEqual(self.report["groundedness"], 0.8)

    def test_later_metrics_present_best_effort(self):
        # entity_f1 / der / contradiction must be PRESENT (asserted above) and
        # crash-safe real numbers; no threshold asserted today.
        self.assertIsInstance(self.report["entity_f1"], float)
        self.assertIsInstance(self.report["der"], float)
        self.assertIsInstance(self.report["contradiction"]["f1"], float)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
