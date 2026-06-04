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
    _entity_clusters,
    _speaker_clusters,
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


class TestEntityF1ReadsPersistence(unittest.TestCase):
    """entity_f1 must score the PERSISTED graph, not re-derive it.

    Proves the regression-guard property: when entity resolution did NOT run
    (no entities/mentions persisted), every gold atom collapses to a singleton
    and ``entity_f1`` is 0.0 — so a broken/no-op ``resolve_entities`` cannot
    falsely pass the metric.
    """

    class _StubIngested:
        def __init__(self, mv):
            self.mv = mv
            self.logical_to_store = {"talk_a": "yt:a", "talk_b": "yt:b"}
            self.store_to_logical = {"yt:a": "talk_a", "yt:b": "talk_b"}

    def test_unresolved_store_scores_zero(self):
        import pathlib
        import sys
        import tempfile

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
        from memovox.config import Config
        from memovox.loom import LoomStore

        gold_entities = {
            "canonical": ["Transformer", "Chinchilla", "Llama"],
            "mentions": {
                "talk_a": ["Transformer", "Chinchilla"],
                "talk_b": ["Chinchilla", "Llama"],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(store=pathlib.Path(tmp) / "store").ensure()
            # Create the schema (empty store: no entities, no mentions) then close.
            LoomStore(config).close()

            class _FakeMv:
                pass

            mv = _FakeMv()
            mv.config = config
            ing = self._StubIngested(mv)

            pred, gold = _entity_clusters(ing, gold_entities)
            # Gold has the cross-talk Chinchilla pair; pred has only singletons.
            _, _, f1 = clustering_f1(pred, gold)
            self.assertEqual(f1, 0.0)


class TestDERReadsPersistence(unittest.TestCase):
    """der must score the PERSISTED canonical_id, not re-derive resolution.

    Proves the regression-guard property (the W2.3 entity_f1 lesson applied to
    speakers): when cross-video speaker resolution did NOT run (per-video
    speakers persisted with no canonical_id), the two same-named Dr. Lee
    speakers stay self-canonical singletons and ``der`` is 0.0 — so a
    broken/no-op ``resolve_speakers`` cannot falsely pass the metric.
    """

    class _StubIngested:
        def __init__(self, mv):
            self.mv = mv
            self.logical_to_store = {"talk_a": "vid:aaaa", "talk_b": "vid:bbbb"}
            self.store_to_logical = {"vid:aaaa": "talk_a", "vid:bbbb": "talk_b"}

    def test_unresolved_store_scores_zero(self):
        import pathlib
        import sys
        import tempfile

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
        from memovox.config import Config
        from memovox.loom import LoomStore, Speaker

        gold_speakers = {
            "identities": {
                "talk_a:Dr. Lee": "person:lee",
                "talk_b:Dr. Lee": "person:lee",
                "talk_b:Prof. Kim": "person:kim",
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(store=pathlib.Path(tmp) / "store").ensure()
            with LoomStore(config) as store:
                # Per-video speakers persisted WITHOUT a canonical_id (resolution
                # did not run): the cross-talk Dr. Lee pair is not unified.
                store.upsert_speaker(Speaker("vid:aaaa:Dr. Lee", "Dr. Lee"))
                store.upsert_speaker(Speaker("vid:bbbb:Dr. Lee", "Dr. Lee"))
                store.upsert_speaker(Speaker("vid:bbbb:Prof. Kim", "Prof. Kim"))

            class _FakeMv:
                pass

            mv = _FakeMv()
            mv.config = config
            ing = self._StubIngested(mv)

            pred, gold = _speaker_clusters(ing, gold_speakers)
            # Gold has the cross-talk Dr. Lee pair; pred has only singletons.
            _, _, der = clustering_f1(pred, gold)
            self.assertEqual(der, 0.0)


class TestThresholdGates(unittest.TestCase):
    def _report(self, hit_rate_v, groundedness_v, contradiction_f1, synthesis_g=1.0,
                entity_f1=1.0, der=1.0):
        return {
            "retrieval": {"hit_rate": hit_rate_v, "mrr": 0.0, "ndcg": 0.0, "k": 5},
            "groundedness": groundedness_v,
            "entity_f1": entity_f1,
            "der": der,
            "contradiction": {"precision": 0.0, "recall": 0.0, "f1": contradiction_f1},
            "synthesis": {"groundedness": synthesis_g, "contradiction_surfaced": True,
                          "consensus_points": 0},
        }

    def test_all_pass(self):
        self.assertEqual(_check_thresholds(self._report(1.0, 1.0, 1.0)), [])

    def test_contradiction_gate_fails(self):
        failures = _check_thresholds(self._report(1.0, 1.0, 0.0))
        self.assertEqual(len(failures), 1)
        self.assertIn("contradiction.f1", failures[0])

    def test_synthesis_gate_fails(self):
        failures = _check_thresholds(self._report(1.0, 1.0, 1.0, synthesis_g=0.0))
        self.assertEqual(len(failures), 1)
        self.assertIn("synthesis.groundedness", failures[0])

    def test_retrieval_and_groundedness_gates_fail(self):
        failures = _check_thresholds(self._report(0.0, 0.0, 1.0))
        self.assertEqual(len(failures), 2)
        self.assertTrue(any("hit_rate" in f for f in failures))
        self.assertTrue(any("groundedness" in f for f in failures))

    def test_entity_f1_and_der_are_gated(self):
        # M1.2 W9: promoted to gates (threshold 0.5) after talk_c verification.
        self.assertEqual(_check_thresholds(self._report(1.0, 1.0, 1.0)), [])  # 1.0 passes
        fails = _check_thresholds(self._report(1.0, 1.0, 1.0, entity_f1=0.3, der=0.2))
        self.assertTrue(any("entity_f1" in f for f in fails))
        self.assertTrue(any("der" in f for f in fails))


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

    def test_observability_block_structural_invariants(self):
        # M0.1 W8: ungated block, but the corpus-size-INDEPENDENT structural facts
        # are hard invariants (not magnitude thresholds, which would flake CI).
        obs = self.report["observability"]
        for stage in ("asr", "visual", "moments", "embed", "claims", "resolve", "digest"):
            self.assertIn(stage, obs["stages_present"])
        self.assertTrue(obs["all_status_ok"])
        self.assertTrue(obs["wall_ms_nonneg"])
        self.assertTrue(obs["counters_reconcile"])     # committed+unsupported == claims
        self.assertIn("retrieve", obs["ask_stages"])
        self.assertIn("synthesize", obs["ask_stages"])
        self.assertGreater(obs["forced_cap_dropped"], 0)  # forced-small cap fires
        self.assertTrue(obs["ok"])

    def test_parity_block_present_and_perfect(self):
        # M0.2 W2: free-path retrieval is byte/rank-identical to the recorded golden.
        p = self.report["parity"]
        self.assertTrue(p["recorded"])      # eval/golden/parity.json exists
        self.assertEqual(p["score"], 1.0)   # every query's vector+lexical top-k matches
        self.assertEqual(p["mismatches"], [])

    def test_incremental_equivalence_present_and_perfect(self):
        # M0.2 W5/W6: incremental consolidation == a single full pass on the golden.
        self.assertEqual(self.report["incremental_equivalence"], 1.0)

    def test_span_unchanged_perfect(self):
        # M0.3 W5: free-path claim spans are byte-identical to the recorded baseline.
        su = self.report["span_unchanged"]
        self.assertTrue(su["recorded"])
        self.assertEqual(su["score"], 1.0)
        self.assertEqual(su["drifted"], [])

    def test_span_accuracy_present_ungated(self):
        # M0.3 W5: word-precision signal computed + printed (M1.2 owns gating).
        self.assertIn("span_accuracy", self.report)
        self.assertIn("tightened_fraction", self.report["span_accuracy"])

    def test_multimodal_block_present_and_shows_lift(self):
        # M1.1 W7: UNGATED multimodal block — the VISUAL leg surfaces an on-screen-
        # only moment that transcript-only retrieval misses. The target moment's OWN
        # transcript does not contain the query term, so the lift is genuinely visual.
        # Pinned to the demonstrated values so a regression that breaks the lift fails.
        mm = self.report["multimodal"]
        self.assertEqual(mm["transcript_only"], 0.0)  # text retrieval misses the slide
        self.assertEqual(mm["tri_modal"], 1.0)        # the visual leg surfaces it
        self.assertEqual(mm["delta"], 1.0)            # a real, non-vacuous lift

    def test_multimodal_is_ungated(self):
        # discipline (a): the multimodal block never participates in --assert-thresholds.
        failures = _check_thresholds(self.report)
        self.assertFalse(any("multimodal" in f for f in failures))

    def test_m1_2_metrics_present_and_ungated(self):
        # M1.2 ungated additions: span_accuracy (mean_iou), topic_f1, keyframe
        # efficiency, claim granularity. None participate in the gate.
        self.assertIsInstance(self.report["topic_f1"], float)
        self.assertLess(self.report["keyframe_efficiency"]["ratio"], 1.0)  # adaptive < uniform
        self.assertGreaterEqual(self.report["claim_granularity"]["claims_per_moment"], 0.0)
        self.assertIn("mean_iou", self.report["span_accuracy"])
        self.assertEqual(self.report["span_accuracy"]["mean_iou"], 1.0)  # citations match gold spans
        failures = _check_thresholds(self.report)
        for k in ("topic_f1", "keyframe", "claim_granularity", "span_accuracy"):
            self.assertFalse(any(k in f for f in failures))

    def test_plan_block_subquery_recall(self):
        # M2.2 W5: the agentic planner covers every clause of the 2-part golden item.
        p = self.report["plan"]
        self.assertEqual(p["multipart_items"], 1)
        self.assertEqual(p["subquery_recall"], 1.0)
        self.assertFalse(any("plan" in f for f in _check_thresholds(self.report)))

    def test_clip_block_coverage_and_invariants(self):
        # M2.3 W6: clip coverage >= 0.3 over the golden clip items, invariants hold.
        c = self.report["clip"]
        self.assertGreaterEqual(c["coverage"], 0.3)
        self.assertTrue(c["non_overlap"])
        self.assertTrue(c["idempotent"])
        self.assertGreaterEqual(c["items"], 3)

    def test_rerank_block_off_equals_today(self):
        # M2.1 W4: the free identity reranker is a no-op — rerank mrr/ndcg equal both
        # the no-rerank baseline AND the retrieval block (the off==today guard).
        rr = self.report["rerank"]
        self.assertEqual(rr["mrr"], rr["no_rerank_mrr"])
        self.assertEqual(rr["ndcg"], rr["no_rerank_ndcg"])
        self.assertEqual(rr["mrr"], self.report["retrieval"]["mrr"])
        self.assertEqual(rr["ndcg"], self.report["retrieval"]["ndcg"])
        self.assertFalse(any("rerank" in f for f in _check_thresholds(self.report)))

    def test_observability_is_ungated(self):
        # discipline (a): observability never participates in --assert-thresholds.
        failures = _check_thresholds(self.report)
        bad = dict(self.report)
        bad["observability"] = dict(self.report["observability"], ok=False)
        self.assertEqual(_check_thresholds(bad), failures)
        self.assertFalse(any("observability" in f for f in failures))

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

    def test_entity_f1_moves_after_resolution(self):
        # W2.3: cross-corpus entity resolution unifies the shared golden entity
        # ('Chinchilla', in both talks) into one node spanning both videos, so
        # entity_f1 is now a real cross-video same-cluster signal > 0.
        # Informational only — entity_f1 stays UNGATED in --assert-thresholds.
        self.assertGreater(self.report["entity_f1"], 0.0)

    def test_der_moves_after_speaker_resolution(self):
        # W4.1: cross-video speaker resolution unifies the same named speaker
        # ('Dr. Lee', in both talks) onto one canonical identity spanning both
        # videos, so der is now a real cross-video same-cluster signal > 0.
        # Informational only — der stays UNGATED in --assert-thresholds.
        self.assertGreater(self.report["der"], 0.0)

    def test_synthesis_section_present_and_grounded(self):
        # Phase 3 / spec §5: corpus-level synthesis is grounded (every sentence
        # cited from its own span) and surfaces the seeded cross-talk contradiction.
        syn = self.report["synthesis"]
        self.assertIn("groundedness", syn)
        self.assertGreaterEqual(syn["groundedness"], 0.8)  # the gated value
        self.assertTrue(syn["contradiction_surfaced"])


class TestClipCoverage(unittest.TestCase):
    def test_clip_coverage_iou(self):
        from eval.harness import clip_coverage
        self.assertEqual(clip_coverage([(0.0, 10.0)], [0.0, 10.0]), 1.0)
        self.assertAlmostEqual(clip_coverage([(0.0, 30.0)], [0.0, 10.0]), 1 / 3)
        self.assertAlmostEqual(clip_coverage([(100.0, 110.0), (0.0, 20.0)], [0.0, 10.0]), 0.5)
        self.assertEqual(clip_coverage([], [0.0, 10.0]), 0.0)


class TestSubqueryRecall(unittest.TestCase):
    def test_subquery_recall(self):
        from eval.harness import subquery_recall
        self.assertEqual(subquery_recall([(["a", "b"], {"a"}), (["c"], {"c"})]), 1.0)
        self.assertEqual(subquery_recall([(["a"], {"z"}), (["c"], {"c"})]), 0.5)
        self.assertEqual(subquery_recall([]), 0.0)


class TestSpanIou(unittest.TestCase):
    def test_span_iou(self):
        from eval.harness import span_iou
        self.assertEqual(span_iou((0.0, 10.0), (0.0, 10.0)), 1.0)   # exact
        self.assertEqual(span_iou((0.0, 5.0), (5.0, 10.0)), 0.0)    # disjoint
        self.assertAlmostEqual(span_iou((0.0, 10.0), (5.0, 15.0)), 1 / 3)  # half
        self.assertEqual(span_iou(None, (0.0, 1.0)), 0.0)           # zero-guard
        self.assertEqual(span_iou((1.0, 1.0), (1.0, 1.0)), 0.0)     # zero-length


class TestThinFixtureDiscipline(unittest.TestCase):
    """M-X W2: a metric may be gated only when it is an exact-equivalence invariant
    OR backed by >=3 golden items (else it flakes on a thin corpus). A premature
    gate (new statistical metric, <3 fixtures, not grandfathered) must be caught."""

    def test_no_gate_is_undeclared(self):
        # Every key _check_thresholds actually gates must be declared, so a NEW gate
        # cannot be added without a deliberate thin-fixture choice.
        from eval.harness import _GATE_DECLARATIONS, _check_thresholds

        all_failing = {
            "retrieval": {"hit_rate": 0.0}, "groundedness": 0.0,
            "contradiction": {"f1": 0.0}, "synthesis": {"groundedness": 0.0},
            "entity_f1": 0.0, "der": 0.0,
            "parity": {"score": 0.0}, "incremental_equivalence": 0.0,
            "span_unchanged": {"score": 0.0},
        }
        failures = _check_thresholds(all_failing)
        gated = {f.split()[0] for f in failures}
        self.assertEqual(gated, set(_GATE_DECLARATIONS),
                         "a gate in _check_thresholds is undeclared (or vice versa)")

    def test_all_current_gates_are_eligible(self):
        from eval.harness import _GATE_DECLARATIONS, _gate_eligible

        for metric, decl in _GATE_DECLARATIONS.items():
            self.assertTrue(_gate_eligible(decl),
                            f"{metric} is gated but not eligible (thin fixture, not exact/grandfathered)")

    def test_premature_statistical_gate_is_flagged(self):
        from eval.harness import _gate_eligible

        # a brand-new statistical metric backed by a non-existent (0-item) fixture
        # is NOT gate-eligible — the guard bites.
        self.assertFalse(_gate_eligible({"kind": "statistical", "fixture": "does_not_exist.json"}))
        # exact-equivalence invariants are always eligible regardless of corpus size
        self.assertTrue(_gate_eligible({"kind": "exact"}))


class TestParityHelper(unittest.TestCase):
    def test_parity_detects_reordering(self):
        from eval.harness import parity

        rec = {"q1": {"vector": ["a", "b"], "lexical": ["x"]}}
        same = {"q1": {"vector": ["a", "b"], "lexical": ["x"]}}
        reordered = {"q1": {"vector": ["b", "a"], "lexical": ["x"]}}  # ULP flip / tie change
        self.assertEqual(parity(same, rec), 1.0)
        self.assertLess(parity(reordered, rec), 1.0)  # the gate has teeth

    def test_parity_empty_recorded_is_perfect(self):
        from eval.harness import parity

        self.assertEqual(parity({}, {}), 1.0)


class TestFrozenSettingsSnapshot(unittest.TestCase):
    """M0.1 W8 / discipline (b): the harness pins the default-OFF flags so a future
    default flip fails loudly instead of silently moving a gate number."""

    def test_default_off_flags_match_current_settings(self):
        from eval.harness import _DEFAULT_OFF_FLAGS
        from memovox.config import Settings

        s = Settings()
        for flag, expected in _DEFAULT_OFF_FLAGS.items():
            self.assertEqual(getattr(s, flag), expected,
                             f"default-OFF flag {flag!r} drifted from the frozen snapshot")

    def test_full_settings_snapshot_is_frozen(self):
        # M1.2 W8: the FULL Settings surface (incl. tuning knobs) is pinned, so any
        # default change fails loudly and forces a conscious re-baseline.
        from dataclasses import asdict

        from eval.harness import EVAL_SETTINGS_SNAPSHOT
        from memovox.config import Settings

        self.assertEqual(asdict(Settings()), EVAL_SETTINGS_SNAPSHOT)

    def test_snapshot_pins_m0_2_flags(self):
        from eval.harness import _DEFAULT_OFF_FLAGS

        self.assertIn("vector_prefilter_fts", _DEFAULT_OFF_FLAGS)
        self.assertFalse(_DEFAULT_OFF_FLAGS["vector_prefilter_fts"])

    def test_snapshot_pins_m0_3_asr_flags(self):
        from eval.harness import _DEFAULT_OFF_FLAGS

        for flag in ("asr_device", "asr_compute_type", "asr_allow_cpu"):
            self.assertIn(flag, _DEFAULT_OFF_FLAGS)

    def test_snapshot_pins_planner_agentic(self):
        from eval.harness import _DEFAULT_OFF_FLAGS
        self.assertIn("planner_agentic", _DEFAULT_OFF_FLAGS)
        self.assertFalse(_DEFAULT_OFF_FLAGS["planner_agentic"])

    def test_snapshot_pins_feature_toggles(self):
        from eval.harness import _DEFAULT_OFF_FLAGS

        # M-X W1: the feature toggles whose flip would move a gate must be pinned.
        self.assertEqual(_DEFAULT_OFF_FLAGS.get("visual_enabled"), True)
        self.assertEqual(_DEFAULT_OFF_FLAGS.get("salience_floor"), 0.0)

    def test_every_settings_field_is_pinned_or_allow_listed(self):
        # M-X W1 reflection completeness: a NEWLY ADDED Settings flag that is neither
        # pinned in the snapshot nor explicitly allow-listed fails loudly, forcing a
        # deliberate choice (pin it, or add to _INTENTIONALLY_UNPINNED with a reason).
        import dataclasses

        from eval.harness import _DEFAULT_OFF_FLAGS, _INTENTIONALLY_UNPINNED
        from memovox.config import Settings

        covered = set(_DEFAULT_OFF_FLAGS) | set(_INTENTIONALLY_UNPINNED)
        missing = [f.name for f in dataclasses.fields(Settings) if f.name not in covered]
        self.assertEqual(missing, [],
                         f"unpinned/unlisted Settings flags (pin or allow-list them): {missing}")


class TestNewExactGates(unittest.TestCase):
    """M0.2 W6: parity and incremental_equivalence gate immediately at 1.0 — they
    are exact-equivalence correctness invariants, not statistical metrics."""

    def _base(self):
        return {
            "retrieval": {"hit_rate": 1.0}, "groundedness": 1.0,
            "contradiction": {"f1": 1.0}, "synthesis": {"groundedness": 1.0},
            "entity_f1": 1.0, "der": 1.0,
            "parity": {"score": 1.0}, "incremental_equivalence": 1.0,
        }

    def test_passing_report_has_no_failures(self):
        from eval.harness import _check_thresholds
        self.assertEqual(_check_thresholds(self._base()), [])

    def test_parity_below_one_fails(self):
        from eval.harness import _check_thresholds
        bad = dict(self._base(), parity={"score": 0.8})
        self.assertTrue(any("parity" in f for f in _check_thresholds(bad)))

    def test_incremental_below_one_fails(self):
        from eval.harness import _check_thresholds
        bad = dict(self._base(), incremental_equivalence=0.0)
        self.assertTrue(any("incremental" in f for f in _check_thresholds(bad)))

    def test_span_unchanged_below_one_fails(self):
        from eval.harness import _check_thresholds
        base = dict(self._base(), span_unchanged={"score": 1.0})
        self.assertEqual(_check_thresholds(base), [])
        bad = dict(base, span_unchanged={"score": 0.9})
        self.assertTrue(any("span_unchanged" in f for f in _check_thresholds(bad)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
