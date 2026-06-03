import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import assay
from memovox.assay.claims import epistemic_type, extract_claims
from memovox.assay.verify import verify_claim
from memovox.backends.base import LLMBackend
from memovox.backends.nli import LexicalNLI
from memovox.config import Settings
from memovox.loom.models import Claim, Moment, SegmentRef


class TestEpistemicTyping(unittest.TestCase):
    cases = [
        ("I think transformers are overrated.", "OPINION"),
        ("Backpropagation is defined as reverse-mode differentiation.", "DEFINITION"),
        ("The model has 7 billion parameters.", "FACT"),
        ("For example, ResNet uses skip connections.", "EXAMPLE"),
        ("First, install the package.", "PROCEDURE"),
        ("This will change by 2027.", "PREDICTION"),
        ("Actually, I misspoke earlier.", "CORRECTION"),
    ]

    def test_types(self):
        for sentence, expected in self.cases:
            self.assertEqual(epistemic_type(sentence), expected, sentence)


class TestExtraction(unittest.TestCase):
    def _moment(self, text):
        return Moment("yt:x#m0000", "yt:x", 10.0, 40.0, text, speaker_id="spk_0")

    def test_extracts_claims_with_span(self):
        m = self._moment("Neural networks learn representations. The dataset has 50000 images.")
        claims = extract_claims(m)
        self.assertEqual(len(claims), 2)
        self.assertTrue(all(c.t_start_s == 10.0 and c.t_end_s == 40.0 for c in claims))
        self.assertTrue(all(c.speaker_id == "spk_0" for c in claims))
        self.assertEqual(claims[0].claim_id, "yt:x#m0000.c00")

    def test_skips_questions_and_short(self):
        m = self._moment("What is attention? Yes. Attention weights tokens by relevance here.")
        claims = extract_claims(m)
        texts = [c.text for c in claims]
        self.assertFalse(any(t.endswith("?") for t in texts))
        self.assertTrue(any("Attention weights tokens" in t for t in texts))

    def test_spo_split(self):
        m = self._moment("The recommended chunk size is 512 tokens.")
        claim = extract_claims(m)[0]
        self.assertIn("chunk size", claim.subject.lower())
        self.assertTrue(claim.object)


class TestClaimSourceSpan(unittest.TestCase):
    def test_claim_bound_to_exact_segment_span(self):
        seg_a = SegmentRef(0.0, 10.0, "Gradient descent minimizes the loss function.")
        seg_b = SegmentRef(10.0, 20.0, "The batch size is set to thirty two samples.")
        seg_c = SegmentRef(20.0, 30.0, "Transformers use multi head self attention layers.")
        transcript = " ".join(s.text for s in (seg_a, seg_b, seg_c))
        m = Moment(
            "yt:x#m0000", "yt:x", 0.0, 30.0, transcript,
            speaker_id="spk_0", segments=[seg_a, seg_b, seg_c],
        )
        claims = extract_claims(m)
        target = [c for c in claims if "multi head self attention" in c.text.lower()]
        self.assertEqual(len(target), 1)
        self.assertEqual((target[0].t_start_s, target[0].t_end_s), (20, 30))
        # The other claims localize to their own segments, not the whole Moment.
        gd = [c for c in claims if "gradient descent" in c.text.lower()][0]
        self.assertEqual((gd.t_start_s, gd.t_end_s), (0, 10))

    def test_falls_back_to_full_span_without_segments(self):
        m = Moment("yt:x#m0000", "yt:x", 5.0, 45.0,
                   "Neural networks learn representations from data.")
        claim = extract_claims(m)[0]
        self.assertEqual((claim.t_start_s, claim.t_end_s), (5.0, 45.0))


class TestVerification(unittest.TestCase):
    def test_in_source_claim_committed(self):
        nli = LexicalNLI()
        source = "The recommended chunk size is 512 tokens for retrieval."
        claim = Claim("c1", "m", "v", text="The recommended chunk size is 512 tokens")
        verify_claim(nli, claim, source, threshold=0.5)
        self.assertEqual(claim.status, "committed")
        self.assertGreaterEqual(claim.entailment_score, 0.5)

    def test_fabricated_claim_unsupported(self):
        nli = LexicalNLI()
        source = "The recommended chunk size is 512 tokens for retrieval."
        claim = Claim("c2", "m", "v", text="The author lives in Paris and owns three cats")
        verify_claim(nli, claim, source, threshold=0.5)
        self.assertEqual(claim.status, "unsupported")

    def test_run_pipeline_marks_status(self):
        nli = LexicalNLI()
        m = Moment("yt:x#m0000", "yt:x", 0.0, 30.0,
                   "Gradient descent minimizes the loss function. The batch size is 32.")
        claims = assay.run(m, nli=nli, settings=Settings(entailment_threshold=0.5))
        self.assertTrue(claims)
        self.assertTrue(all(c.status == "committed" for c in claims))
        self.assertTrue(all(c.entailment_score >= 0.5 for c in claims))


class _HallucinatingLLM(LLMBackend):
    """Emits one verbatim-from-span claim and one fabricated claim."""

    is_generative = True

    def complete(self, prompt, **kw):
        return ('[{"text":"The chain rule is central.","type":"FACT"},'
                ' {"text":"Quantum tunneling powers the optimizer.","type":"FACT"}]')


class _ScatterLLM(LLMBackend):
    """Emits a verbatim claim from segment A, plus a fabricated claim that splices
    segment A's subject onto segment B's predicate — every token exists in the
    Moment but the splice is false within either span."""

    is_generative = True

    def complete(self, prompt, **kw):
        return ('[{"text":"The chain rule is central to backprop.","type":"FACT"},'
                ' {"text":"The chain rule exhibits quantum tunneling.","type":"FACT"}]')


class TestGateRejectsHallucination(unittest.TestCase):
    def test_gate_rejects_unsupported_llm_claim(self):
        m = Moment("v#m0", "v", 0, 12, "The chain rule is central.",
                   segments=[SegmentRef(0, 12, "The chain rule is central.")])
        claims = assay.run(m, nli=LexicalNLI(), llm=_HallucinatingLLM())
        by_text = {c.text: c.status for c in claims}
        self.assertEqual(by_text["The chain rule is central."], "committed")
        self.assertEqual(by_text["Quantum tunneling powers the optimizer."], "unsupported")

    def test_gate_rejects_hallucination_when_tokens_scatter_across_other_spans(self):
        # The fix's teeth: the whole-Moment text scatters tokens of a fabricated
        # claim across OTHER segments, so the LEGACY whole-Moment premise overlaps
        # ~1.0 and wrongly COMMITS it. The real claim localizes (W1.2) to a
        # segment that does NOT contain the fabricated claim's discriminating
        # tokens, so verifying against that own span rejects it. This test FAILS
        # under the old whole-Moment premise and PASSES under per-span verify.
        segs = [
            SegmentRef(0, 4, "The chain rule is central to backprop."),
            SegmentRef(4, 8, "Photons exhibit quantum tunneling in barriers."),
        ]
        transcript = " ".join(s.text for s in segs)
        m = Moment("v#m1", "v", 0, 8, transcript, segments=segs)
        claims = assay.run(m, nli=LexicalNLI(), llm=_ScatterLLM())
        by_text = {c.text: c.status for c in claims}
        self.assertEqual(by_text["The chain rule is central to backprop."], "committed")
        self.assertEqual(
            by_text["The chain rule exhibits quantum tunneling."], "unsupported"
        )


if __name__ == "__main__":
    unittest.main()
