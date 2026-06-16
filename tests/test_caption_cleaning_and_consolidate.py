"""Richer-corpus stress-hardening regressions (iter11):

  * caption cleaning — sentence fragments must not become bogus speaker labels,
    and HTML entities (``&nbsp;``/``&#39;``) must be decoded out of claim text;
  * topic-scoped contradiction search must filter by topic BEFORE the max_claims
    cap, so a relevant pair that arrives late in ingest order still gets compared.
"""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.base import Segment
from memovox.backends.nli import LexicalNLI
from memovox.config import Config
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.consolidate import find_contradictions
from memovox.loom.models import Claim
from memovox.backends.base import Segment as _Seg  # noqa: F401 (alias for clarity)
from memovox.stentor.transcript import (
    _looks_like_speaker,
    clean_segments,
    clean_text,
)


class TestSpeakerLabelHeuristic(unittest.TestCase):
    """A leading ``Name:`` becomes a speaker only when it looks like a name, not a
    sentence fragment that happens to carry a colon (the richer-corpus false
    positives: 'But caveat', 'For example', 'And that worked great', ...)."""

    def test_real_labels_accepted(self):
        for name in ["BRADY HARAN", "MATT PARKER", "Rob Wiblin", "Molaison",
                     "NEWSCASTER", "Yanjaa", "YANJAA", "Dr. Lee", "McDonald",
                     "Tony Padilla", "REAGAN"]:
            self.assertTrue(_looks_like_speaker(name), name)

    def test_sentence_fragments_rejected(self):
        for frag in ["But caveat", "For example", "And that worked great",
                     "I have three world records", "So", "Well", "Now",
                     "What I mean is", "this is the part"]:
            self.assertFalse(_looks_like_speaker(frag), frag)

    def test_single_word_section_openers_rejected(self):
        # latent false-positive class the panel flagged: a lone Title-case opener +
        # colon ("Note:", "Today:", "Look:") must NOT be hoisted to a speaker (it would
        # eat the leading word). Real single-word surnames (Molaison) still pass.
        for frag in ["Note", "Today", "Look", "First", "Finally", "Warning", "Remember"]:
            self.assertFalse(_looks_like_speaker(frag), frag)
        self.assertTrue(_looks_like_speaker("Molaison"))


class TestSpeakerCaseCanonicalization(unittest.TestCase):
    """A speaker spelled two ways in one video ("YANJAA" label vs "Yanjaa" <v> tag) must
    collapse to ONE speaker, not split into two ids with duplicated turns."""

    def test_case_variants_collapse_to_one_speaker(self):
        segs = [
            Segment(start=0.0, end=2.0, text="YANJAA: five is an S three is an A."),
            Segment(start=2.0, end=4.0, text="Yanjaa: I have three world records here."),
        ]
        out = [s for s in clean_segments(segs) if s.kind == "speech"]
        speakers = {s.speaker for s in out}
        self.assertEqual(len(speakers), 1, f"case-variant speakers did not collapse: {speakers}")


class TestBracketAnnotationLeak(unittest.TestCase):
    """A residual [bracket] annotation the EVENT_RE whitelist did not name must not land
    in claim text (sound effects, foreign-language markers, bracketed names, censored
    profanity), while code/math brackets ([i], [b,t]) are preserved."""

    def test_nonwhitelisted_annotations_stripped(self):
        cases = {
            "[clock ticking] This is an absolute": "This is an absolute",
            "[laughs] that was funny": "that was funny",
            "[speaking in Thai] the food is great": "the food is great",
            "[Mark Wiens] tries the soup": "tries the soup",
        }
        for raw, expected in cases.items():
            self.assertEqual(clean_text(raw)[0], expected, raw)

    def test_censored_profanity_dropped(self):
        text, _ = clean_text("[ __ ] up, but I can't even lie.")
        self.assertNotIn("[", text)
        self.assertNotIn("_", text)
        self.assertTrue(text.startswith("up,"))

    def test_code_math_brackets_preserved(self):
        # a 3+ letter word inside brackets is an annotation; short code indices are not.
        text, _ = clean_text("logits[b,t] and x[i] index into the tensor")
        self.assertIn("logits[b,t]", text)
        self.assertIn("x[i]", text)

    def test_markdown_link_label_kept_url_dropped(self):
        text, ev = clean_text("see [this short essay](https://example.com/p/x) for details")
        self.assertEqual(text, "see this short essay for details")
        self.assertNotIn("http", text)
        self.assertEqual(ev, [])  # the label is NOT recorded as a bogus audio event

    def test_orphaned_url_after_sentence_split_stripped(self):
        # the markdown link split across a sentence boundary leaves "label](url)"
        text, _ = clean_text("short](https://helentoner.substack.com/p/long).” She pointed out")
        self.assertNotIn("http", text)
        self.assertNotIn("](", text)
        self.assertIn("She pointed out", text)

    def test_fragment_prefix_not_hoisted_to_speaker(self):
        # "But caveat: ..." is content, not a speaker turn -> speaker stays unset
        # and the text is preserved (not stripped as a label).
        segs = [Segment(start=0.0, end=2.0, text="But caveat: the index can mislead.")]
        out = [s for s in clean_segments(segs) if s.kind == "speech"]
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0].speaker)
        self.assertIn("caveat", out[0].text)

    def test_real_label_still_extracted(self):
        segs = [Segment(start=0.0, end=2.0, text="BRADY HARAN: so what is it?")]
        out = [s for s in clean_segments(segs) if s.kind == "speech"]
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].speaker, "BRADY HARAN")
        self.assertNotIn("BRADY HARAN", out[0].text)


class TestHtmlEntityDecoding(unittest.TestCase):
    """WebVTT escapes ``&`` ``<`` ``>`` and emits ``&nbsp;``; raw, these leak into
    claim text ("By 1920,&nbsp;&nbsp;"). clean_text must decode + collapse them."""

    def test_nbsp_decoded_and_collapsed(self):
        text, _ = clean_text("By 1920,&nbsp;&nbsp; the banks failed")
        self.assertNotIn("&nbsp;", text)
        self.assertNotIn("\xa0", text)
        self.assertEqual(text, "By 1920, the banks failed")

    def test_named_and_numeric_entities(self):
        text, _ = clean_text("Tom &amp; Jerry said it&#39;s fine")
        self.assertEqual(text, "Tom & Jerry said it's fine")

    def test_decoded_lt_not_reparsed_as_tag(self):
        # A genuine "<i>" tag is stripped; an escaped "&lt;3" decodes to literal text.
        text, _ = clean_text("<i>love</i> is &lt;3 always")
        self.assertEqual(text, "love is <3 always")


class TestTopicContradictionCapOrdering(unittest.TestCase):
    """find_contradictions(topic=...) must filter to the topic BEFORE truncating to
    max_claims; otherwise a relevant pair that lands past the cap in ingest order is
    silently never compared (looks like an NLI miss but is a cap artifact)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        for v in ("yt:a", "yt:b"):
            self.store.upsert_video(Video(v, f"https://youtu.be/{v[3:]}", "talk"))
        # 40 filler claims (off-topic), THEN the on-topic contradiction pair last —
        # so the relevant claims sit at rowids past a small cap.
        def add(cid, mid, vid, text, t):
            self.store.add_moment(Moment(mid, vid, float(t), float(t) + 1.0, text, index=0))
            self.store.add_claim(Claim(cid, mid, vid, text,
                                       t_start_s=float(t), t_end_s=float(t) + 1.0))

        for i in range(40):
            add(f"yt:a#m{i:04d}.c0", f"yt:a#m{i:04d}", "yt:a",
                f"a generic remark number {i} about assorted weather and travel", i)
        add("yt:a#mz.c0", "yt:a#mz", "yt:a",
            "saturated fat causes heart disease", 100)
        add("yt:b#mz.c0", "yt:b#mz", "yt:b",
            "saturated fat does not cause heart disease", 5)
        self.nli = LexicalNLI()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_late_topic_pair_is_compared_under_small_cap(self):
        # cap below the rowid of the on-topic pair: cap-then-filter would yield zero
        # candidates; filter-then-cap keeps the two saturated-fat claims.
        pairs = find_contradictions(self.store, nli=self.nli, topic="saturated fat",
                                    max_claims=10, write_edges=False)
        vids = {frozenset((p.claim_a.video_id, p.claim_b.video_id)) for p in pairs}
        self.assertIn(frozenset(("yt:a", "yt:b")), vids,
                      "late-arriving on-topic contradiction pair was not compared")

    def test_no_topic_path_unchanged_by_cap(self):
        # Without a topic the cap still bounds the first max_claims (cost control).
        pairs = find_contradictions(self.store, nli=self.nli, topic=None,
                                    max_claims=10, write_edges=False)
        self.assertIsInstance(pairs, list)


class TestContradictionPrecisionGate(unittest.TestCase):
    """A full-corpus scan must not emit garbage cross-video edges: lexical NLI falsely
    flags unrelated SHORT claims that share a couple of generic tokens plus a negation.
    Only substantive NEAR-MIRRORS (>= min_shared shared tokens AND jaccard >= floor)
    may be NLI-compared, so real opposing-polarity duplicates survive and coincidences
    do not."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.nli = LexicalNLI()
        for v in ("yt:a", "yt:b"):
            self.store.upsert_video(Video(v, f"https://youtu.be/{v[3:]}", "talk"))

        def add(cid, mid, vid, text, t):
            self.store.add_moment(Moment(mid, vid, float(t), float(t) + 1.0, text, index=0))
            self.store.add_claim(Claim(cid, mid, vid, text, t_start_s=float(t), t_end_s=float(t) + 1.0))

        # near-mirror genuine contradiction (high overlap + opposing polarity)
        add("yt:a#m0.c0", "yt:a#m0", "yt:a", "remote work improves team productivity significantly", 0)
        add("yt:b#m0.c0", "yt:b#m0", "yt:b", "remote work does not improve team productivity significantly", 0)
        # unrelated short fragments that share 2 generic tokens + a negation cue
        add("yt:a#m1.c0", "yt:a#m1", "yt:a", "this work here is not done", 10)
        add("yt:b#m1.c0", "yt:b#m1", "yt:b", "the work here is good", 10)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_near_mirror_found_garbage_rejected(self):
        pairs = find_contradictions(self.store, nli=self.nli, write_edges=False)
        flagged = {frozenset((p.claim_a.claim_id, p.claim_b.claim_id)) for p in pairs}
        self.assertIn(frozenset(("yt:a#m0.c0", "yt:b#m0.c0")), flagged,
                      "genuine near-mirror contradiction was missed")
        self.assertNotIn(frozenset(("yt:a#m1.c0", "yt:b#m1.c0")), flagged,
                         "unrelated short fragments were flagged as a contradiction")


class TestRelevanceGenericVerbFilter(unittest.TestCase):
    """Generic advice/transaction verbs (recommend/suggest/purchase/buy) must not act as
    distinctive topicality tokens — the leak that let 'what is the best way to recommend a
    first home purchase?' be answered confidently against an unrelated corpus."""

    def test_advice_verbs_not_distinctive(self):
        # The verbs must NOT be distinctive TOPICALITY tokens (that's what closes the
        # OOC leak). They DO remain COVERAGE tokens (see test_advice_verbs_not_in_
        # coverage_filler) so a legit "buy a <topic>" query is not over-refused.
        from memovox.augur.answer import _rel_tokens
        for v in ["recommend", "recommended", "suggest", "purchase", "buy", "bought"]:
            self.assertNotIn(v, _rel_tokens(f"please {v} something"),
                             f"{v} still a distinctive topicality token")

    def test_real_topic_word_still_distinctive(self):
        from memovox.augur.answer import _rel_tokens
        self.assertIn("submariner", _rel_tokens("recommend the rolex submariner"))

    def test_advice_verbs_not_in_coverage_filler(self):
        # Regression: the verbs must live ONLY in topicality (_COMMON_WORDS), NOT in
        # _COVERAGE_FILLER — else a legit in-corpus "which Rolex should I buy?" loses its
        # coverage token and over-refuses. They must still be COVERAGE content tokens.
        from memovox.augur.answer import _coverage_tokens
        cov = _coverage_tokens("which rolex should I buy")
        self.assertIn("buy", cov)
        self.assertIn("rolex", cov)


class TestSynthesizeSalientFallback(unittest.TestCase):
    """When a topic IS covered (citations exist) but the free path extracts no
    consensus/contradiction structure, synthesize must emit a salient extractive summary
    (low_evidence=False) — NOT 'ingest more sources' alongside real citations."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.nli = LexicalNLI()
        for v in ("yt:a", "yt:b"):
            self.store.upsert_video(Video(v, f"https://youtu.be/{v[3:]}", "talk"))
        # Two videos on 'saturated fat' whose claims share few tokens -> no token-Jaccard
        # consensus and no lexical-NLI contradiction (different framings, same topic).
        rows = [
            ("yt:a#m0", "yt:a", "Saturated fat raises LDL cholesterol substantially.", 0.0),
            ("yt:a#m1", "yt:a", "Dietary saturated fat correlates with cardiovascular risk.", 30.0),
            ("yt:b#m0", "yt:b", "Saturated fat has minimal independent effect on mortality.", 0.0),
            ("yt:b#m1", "yt:b", "Replacing saturated fat with sugar provides no benefit.", 30.0),
        ]
        for mid, vid, text, t in rows:
            self.store.add_moment(Moment(mid, vid, float(t), float(t) + 5.0, text, index=0))
            self.store.add_claim(Claim(f"{mid}.c0", mid, vid, text,
                                       t_start_s=float(t), t_end_s=float(t) + 5.0, salience=1.0))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_citations_without_structure_yield_summary_not_low_evidence(self):
        from memovox.augur.synthesize import synthesize
        s = synthesize(self.store, "saturated fat", nli=self.nli).to_dict()
        self.assertGreater(len(s["citations"]), 0)
        self.assertFalse(s["low_evidence"], "rich topic wrongly reported as low evidence")
        self.assertTrue(s["text"].strip(), "synthesis text is empty")
        self.assertIn("[", s["text"], "summary sentences must carry [n] citations")
        # honest: no consensus claimed when none was algorithmically detected
        self.assertEqual(len(s["consensus_points"]), 0)

    def test_genuinely_empty_topic_still_low_evidence(self):
        from memovox.augur.synthesize import synthesize
        s = synthesize(self.store, "quantum chromodynamics", nli=self.nli).to_dict()
        self.assertTrue(s["low_evidence"])
        self.assertEqual(len(s["citations"]), 0)


class TestSynthesizeOutOfCorpusGate(unittest.TestCase):
    """The salient fallback must NOT confabulate a synthesis for an OUT-OF-CORPUS topic
    whose claims merely share a polysemous token (the panel's 'capital of Mongolia'
    regression). synthesize must apply the same topicality/coverage gate ask() uses."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.nli = LexicalNLI()
        self.store.upsert_video(Video("yt:a", "https://youtu.be/a", "talk"))
        # 60 off-topic claims that share the polysemous token "capital" (city / money
        # senses) but nothing about Mongolia.
        senses = ["the capital city is lovely this spring",
                  "raising capital is hard for a new startup",
                  "capital letters begin every sentence here"]
        for i in range(60):
            mid = f"yt:a#m{i:04d}"
            text = senses[i % 3] + f" number {i}"
            self.store.add_moment(Moment(mid, "yt:a", float(i), float(i) + 1.0, text, index=0))
            self.store.add_claim(Claim(f"{mid}.c0", mid, "yt:a", text,
                                       t_start_s=float(i), t_end_s=float(i) + 1.0, salience=1.0))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_polysemous_ooc_topic_refused(self):
        from memovox.augur.synthesize import synthesize
        # "capital" recurs, but "mongolia" is absent from every moment -> a single cited
        # moment never covers the topic -> coverage below floor -> refuse, no citations.
        s = synthesize(self.store, "capital of Mongolia", nli=self.nli).to_dict()
        self.assertTrue(s["low_evidence"])
        self.assertEqual(len(s["citations"]), 0)

    def test_ooc_topic_that_builds_structure_still_refused(self):
        # The OOC gate must run on the STRUCTURED path too, not only the salient
        # fallback: a generic question token ("form") that matches identical claims
        # builds a consensus cluster, making `parts` non-empty — a fallback-only gate
        # would be bypassed and confabulate a confident synthesis.
        from memovox.augur.synthesize import synthesize
        for i in range(6):  # identical claims -> a consensus cluster sharing "form"
            mid = f"yt:a#f{i:04d}"
            text = "the panel will form a strong consensus on this"
            self.store.add_moment(Moment(mid, "yt:a", float(i), float(i) + 1.0, text, index=0))
            self.store.add_claim(Claim(f"{mid}.c0", mid, "yt:a", text,
                                       t_start_s=float(i), t_end_s=float(i) + 1.0, salience=1.0))
        s = synthesize(self.store, "how do volcanoes form?", nli=self.nli).to_dict()
        self.assertTrue(s["low_evidence"], "OOC topic that built structure was not refused")
        self.assertEqual(len(s["citations"]), 0)


if __name__ == "__main__":
    unittest.main()
