"""Upgraded-path panel fixes (BGE-M3 + DeBERTa review):
  1. consensus must be NLI-ENTAILMENT-confirmed, not pure cosine co-location (a debunking
     source was reported as endorsing the claim it debunks);
  2. value-judgment framing words ('investment'/'good'/'worth') must not over-refuse an
     in-corpus value query ('are watches a good investment?');
  3. a MID-cue speaker label ('ADDIS :...') must be stripped from claim text.
"""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.base import Embedder, NLIBackend, NLIResult
from memovox.config import Config
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.models import Claim


# ---- Fix 2: value-judgment framing words -----------------------------------
class TestValueFramingNotTopical(unittest.TestCase):
    def test_value_words_dropped_from_both_signals(self):
        from memovox.augur.answer import _coverage_tokens, _rel_tokens
        q = "are watches a good investment"
        for w in ("good", "investment", "worth", "value", "best", "great"):
            self.assertNotIn(w, _rel_tokens(f"is it a {w} thing"), f"{w} still distinctive")
            self.assertNotIn(w, _coverage_tokens(f"is it a {w} thing"), f"{w} still a coverage token")
        self.assertEqual(_rel_tokens(q), {"watches"})        # only the real subject survives
        self.assertEqual(_coverage_tokens(q), {"watches"})


# ---- Fix 3: mid-cue speaker label -------------------------------------------
class TestMidCueSpeakerLabel(unittest.TestCase):
    def _speech(self, segs):
        from memovox.stentor.transcript import clean_segments
        out = [s for s in clean_segments(segs) if s.kind == "speech"]
        return " ".join(s.text for s in out)

    def test_mid_cue_label_stripped(self):
        from memovox.backends.base import Segment
        # one cue whose content has a mid-text speaker change "ADDIS :"
        segs = [Segment(start=0.0, end=4.0,
                        text="that was the question. ADDIS :When we go to retrieve that memory")]
        joined = self._speech(segs)
        self.assertNotIn("ADDIS :", joined)
        self.assertNotIn("ADDIS:", joined)
        self.assertIn("retrieve that memory", joined)

    def test_bare_no_colon_confirmed_speaker_stripped(self):
        # Round-2 panel UNDER-LEAK: a BARE (no-colon) occurrence of a speaker that is
        # confirmed elsewhere in the SAME document must also be stripped (it was leaking
        # into claim text + live answer snippets: "ADDIS So, the fact that we reconstruct...").
        from memovox.backends.base import Segment
        segs = [
            Segment(start=0.0, end=4.0, text="ADDIS: Memory is reconstructed in stages."),
            Segment(start=4.0, end=8.0,
                    text="ADDIS So, the fact that we reconstruct episodic memory matters."),
        ]
        joined = self._speech(segs)
        self.assertNotIn("ADDIS", joined)
        self.assertIn("the fact that we reconstruct episodic memory", joined)

    def test_bare_label_after_sentence_boundary_stripped(self):
        from memovox.backends.base import Segment
        segs = [
            Segment(start=0.0, end=4.0, text="ADDIS: Memory is reconstructed in stages."),
            Segment(start=4.0, end=8.0,
                    text="that was the question. ADDIS When we go to retrieve that memory."),
        ]
        joined = self._speech(segs)
        self.assertNotIn("ADDIS", joined)
        self.assertIn("that was the question", joined)
        self.assertIn("When we go to retrieve that memory", joined)

    def test_no_space_colon_label_still_stripped(self):
        # Guards against the REJECTED naive "require whitespace before the colon" fix,
        # which would re-leak the no-space "ADDIS:" form (4 corpus occurrences).
        from memovox.backends.base import Segment
        segs = [
            Segment(start=0.0, end=4.0, text="ADDIS :Memory is reconstructed in stages."),
            Segment(start=4.0, end=8.0, text="The story continues. ADDIS:So this poses a question."),
        ]
        joined = self._speech(segs)
        self.assertNotIn("ADDIS", joined)
        self.assertIn("So this poses a question", joined)

    def test_prose_appositive_subject_preserved(self):
        # Round-2 panel OVER-STRIP: a "Subject: predicate" prose appositive whose subject
        # is NOT a confirmed speaker must keep its subject noun (the topic word retrieval
        # needs). Previously "Steve Jobs: a visionary" -> " a visionary" (subject deleted).
        from memovox.backends.base import Segment
        segs = [
            Segment(start=0.0, end=5.0,
                    text="My third story is about Steve Jobs: a visionary who changed computing."),
            Segment(start=5.0, end=9.0,
                    text="Let us discuss the Submariner: a classic dive watch."),
        ]
        joined = self._speech(segs)
        self.assertIn("Steve Jobs", joined)
        self.assertIn("Submariner", joined)


# ---- Fix 1: consensus must be NLI-entailment-confirmed ----------------------
class _FakeEmbedder(Embedder):
    is_semantic = True
    name = "fake"

    def __init__(self, vecs):
        self._v = vecs

    @classmethod
    def is_available(cls):
        return True

    def embed(self, texts):
        return [self._v[t] for t in texts]


class _FakeNLI(NLIBackend):
    """label/scores keyed by frozenset(pair); default neutral."""
    name = "fake"

    def __init__(self, entailing=(), contradicting=()):
        self._ent = {frozenset(p) for p in entailing}
        self._con = {frozenset(p) for p in contradicting}

    @classmethod
    def is_available(cls):
        return True

    def classify(self, premise, hypothesis):
        key = frozenset((premise, hypothesis))
        if key in self._ent:
            return NLIResult("entailment", 0.95, 0.05, 0.0)
        if key in self._con:
            return NLIResult("contradiction", 0.0, 0.05, 0.95)
        return NLIResult("neutral", 0.08, 0.85, 0.07)


class TestConsensusEntailmentConfirmed(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        for v in ("yt:a", "yt:b"):
            self.store.upsert_video(Video(v, f"https://youtu.be/{v[3:]}", "talk"))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _add(self, cid, vid, text):
        mid = f"{vid}#m0"
        if self.store.get_moment(mid) is None:
            self.store.add_moment(Moment(mid, vid, 0.0, 5.0, text, index=0))
        self.store.add_claim(Claim(cid, mid, vid, text, salience=0.5, t_start_s=0.0, t_end_s=5.0))

    def test_cosine_grouped_but_neutral_cluster_is_not_consensus(self):
        from memovox.augur.synthesize import synthesize
        from memovox.config import Settings
        # Two breakfast claims: cosine-similar (forced) but NLI says they do NOT entail
        # (one endorses, one debunks). They must NOT be reported as consensus.
        pro = "breakfast jumpstarts your morning metabolism considerably each day"
        debunk = "the anti-breakfast position cites no solid metabolic evidence anywhere"
        self._add("yt:a#m0.c0", "yt:a", pro)
        self._add("yt:b#m0.c0", "yt:b", debunk)
        emb = _FakeEmbedder({pro: [1.0, 0.0, 0.05], debunk: [0.98, 0.0, 0.10]})  # cosine ~1
        nli = _FakeNLI()  # everything neutral
        s = synthesize(self.store, "breakfast", nli=nli, embedder=emb,
                       settings=Settings(consensus_cosine=0.9)).to_dict()
        self.assertEqual(len(s["consensus_points"]), 0,
                         "cosine-grouped but NON-entailing cluster wrongly reported as consensus")

    def test_genuinely_entailing_cluster_is_consensus(self):
        from memovox.augur.synthesize import synthesize
        from memovox.config import Settings
        a = "remote work improves measured team productivity substantially"
        b = "working remotely raises overall team output a lot"
        self._add("yt:a#m0.c0", "yt:a", a)
        self._add("yt:b#m0.c0", "yt:b", b)
        emb = _FakeEmbedder({a: [1.0, 0.0, 0.05], b: [0.98, 0.0, 0.10]})
        nli = _FakeNLI(entailing=[(a, b), (b, a)])
        s = synthesize(self.store, "team", nli=nli, embedder=emb,
                       settings=Settings(consensus_cosine=0.9)).to_dict()
        self.assertGreaterEqual(len(s["consensus_points"]), 1,
                                "genuine cross-video entailment dropped from consensus")


if __name__ == "__main__":
    unittest.main()
