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


# ---- Fix: corpus topic words wrongly in _COMMON_WORDS -----------------------
class TestCorpusTopicWordsAreDistinctive(unittest.TestCase):
    """WATCH_CAR_TOPIC_OVERREFUSAL (round-4 panel): a word the corpus discusses AS a
    subject ('watch' -> luxury-watch reviews, 'car' -> car reviews) must stay a
    DISTINCTIVE topic token. Listed in _COMMON_WORDS it is dropped from the topicality
    signal, so a question whose only other tokens are framing words ('what watch is best
    to buy?') loses its sole topic token and is wrongly refused. The df-topicality gate
    still holds the OOC line (an incidental verb use 'where can I watch the game?' refuses
    because the absent subject is below min_df)."""

    def test_watch_and_car_stay_distinctive(self):
        from memovox.augur.answer import _rel_tokens
        self.assertEqual(_rel_tokens("what watch is best for a first purchase"), {"watch"})
        self.assertEqual(_rel_tokens("what car should I buy"), {"car"})

    def test_political_role_words_dropped_but_chess_pieces_kept(self):
        # PRESIDENT_ROLE_WORD_OOC_LEAK (round-6 panel): a generic political/leadership ROLE
        # word must NOT clear topicality on its own (else "who is the president of Brazil?"
        # leaks — 'president' df=10 passes while the real subject 'brazil' df=3 is below
        # floor). Only the real subject survives, and it must be below the df floor to refuse.
        from memovox.augur.answer import _rel_tokens
        self.assertEqual(_rel_tokens("who is the president of brazil"), {"brazil"})
        self.assertEqual(_rel_tokens("what is the vice president"), set())
        for w in ("president", "minister", "senator", "governor", "mayor", "chancellor"):
            self.assertNotIn(w, _rel_tokens(f"who is the {w} now"), w)
        # chess pieces are genuine corpus SUBJECTS (king df=40, queen df=37) and must stay
        # distinctive — they were DELIBERATELY excluded from the role-word stoplist.
        self.assertIn("king", _rel_tokens("who has the king on g1"))
        self.assertIn("queen", _rel_tokens("where is the queen"))


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

    def test_titlecase_content_name_preserved_even_when_speaker(self):
        # Round-3 panel C_BARE_OVERSTRIP: the bare (no-colon) strip must fire ONLY on the
        # ALLCAPS broadcast-label form. A confirmed Title-case speaker whose name is also a
        # sentence-initial CONTENT word must keep that word in prose ("Mark my words...",
        # "Reagan was the president..."). Previously the IGNORECASE bare strip ate them.
        from memovox.backends.base import Segment
        segs = [
            Segment(start=0.0, end=4.0, text="Mark: Today we discuss the news."),
            Segment(start=4.0, end=8.0, text="Reagan: I want to address the nation."),
            Segment(start=8.0, end=12.0, text="Mark my words, this will change everything."),
            Segment(start=12.0, end=16.0, text="Reagan was the president during that era."),
        ]
        joined = self._speech(segs)
        self.assertIn("Mark my words", joined)
        self.assertIn("Reagan was the president", joined)

    def test_allcaps_bare_label_of_titlecase_speaker_stripped(self):
        # The ALLCAPS broadcast form of a confirmed (here Title-case) speaker IS a bare
        # label and must still be stripped — the strip is not weakened, only narrowed.
        from memovox.backends.base import Segment
        segs = [
            Segment(start=0.0, end=4.0, text="Reagan: I want to address the nation."),
            Segment(start=4.0, end=8.0, text="The crowd waited. REAGAN We begin with the economy."),
        ]
        joined = self._speech(segs)
        self.assertNotIn("REAGAN", joined)
        self.assertIn("We begin with the economy", joined)


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


# ---- Fix: garbage SUPPORTS/CONTRADICTS on filler-only phatic overlap ---------
class TestPhaticOverlapRejected(unittest.TestCase):
    """GARBAGE_SUPPORTS_PHATIC (round-3 panel): a cross-video near-mirror pair whose
    shared tokens are ALL generic discourse filler (e.g. "okay let's try one thing" /
    "okay let's try one thing") must NOT produce a SUPPORTS/CONTRADICTS edge — DeBERTa
    hallucinates high-confidence entailment on phatic fragments. A pair that shares a
    DISTINCTIVE topical token still forms an edge. (Salience can't separate these: real
    short-form contradictions score 0.20-0.27, identical to the phatic fragments.)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        for v in ("yt:a", "yt:b"):
            self.store.upsert_video(Video(v, f"https://youtu.be/{v[3:]}", "talk"))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _add(self, cid, vid, text, salience=0.5):
        mid = f"{vid}#m_{cid[-2:]}"
        if self.store.get_moment(mid) is None:
            self.store.add_moment(Moment(mid, vid, 0.0, 5.0, text, index=0))
        self.store.add_claim(Claim(cid, mid, vid, text, salience=salience, t_start_s=0.0, t_end_s=5.0))

    def test_filler_only_overlap_makes_no_edge(self):
        from memovox.loom.consolidate import find_contradictions
        # shared tokens are all discourse filler (okay/let/try/one/thing) -> no edge,
        # even with HIGH salience (proving it is the content gate, not a salience floor).
        a = "okay let us try one thing today"
        b = "okay let us try one thing now"
        self._add("yt:a#c0", "yt:a", a, salience=0.9)
        self._add("yt:b#c0", "yt:b", b, salience=0.9)
        nli = _FakeNLI(entailing=[(a, b), (b, a)])
        pairs = find_contradictions(self.store, nli=nli, include_supports=True)
        self.assertEqual(len(pairs), 0, "filler-only phatic pair wrongly produced an edge")

    def test_distinctive_overlap_still_makes_edge_even_low_salience(self):
        from memovox.loom.consolidate import find_contradictions
        # genuine short-form contradiction-style near-mirror with LOW salience (0.22):
        # it shares DISTINCTIVE topical tokens, so it must STILL form an edge — a salience
        # floor would have wrongly dropped it (this is the golden-gate failure mode).
        a = "scaling laws will keep holding beyond current compute budgets"
        b = "scaling laws will keep holding beyond current compute budgets entirely"
        self._add("yt:a#c0", "yt:a", a, salience=0.22)
        self._add("yt:b#c0", "yt:b", b, salience=0.24)
        nli = _FakeNLI(entailing=[(a, b), (b, a)])
        pairs = find_contradictions(self.store, nli=nli, include_supports=True)
        self.assertGreaterEqual(len(pairs), 1, "distinctive near-mirror wrongly dropped")

    def test_discourse_frame_overlap_makes_no_edge(self):
        # PHATIC_FRAME_GATE_INCOMPLETE (round-4 panel): a generic discourse FRAME whose
        # only shared tokens are intensifier/frame words ("the most important thing is
        # really X") must NOT form an edge even at high salience — DeBERTa hallucinates
        # contradiction ~1.0 on these. Closes the CLASS, not just the 3 corpus fragments.
        from memovox.loom.consolidate import find_contradictions
        a = "the most important thing here is really patience"
        b = "the most important thing now is really speed"
        self._add("yt:a#c0", "yt:a", a, salience=0.9)
        self._add("yt:b#c0", "yt:b", b, salience=0.9)
        nli = _FakeNLI(contradicting=[(a, b), (b, a)])
        pairs = find_contradictions(self.store, nli=nli, include_supports=True)
        self.assertEqual(len(pairs), 0, "discourse-frame pair wrongly produced an edge")

    def test_contraction_artifact_not_distinctive(self):
        # tokenize("don't") == ['don','t'] -> 'don' survived as a bogus len-3 'distinctive'
        # token (likewise won/didn/isn/...). The contraction STEM must be treated as filler
        # (a genuine content word in the same phrase, e.g. 'agree', stays distinctive).
        from memovox.loom.consolidate import _distinctive_tokens, _content_tokens
        for frag, stem in (("we don't really think so", "don"),
                           ("they won't ever change", "won"),
                           ("it isn't one thing", "isn"),
                           ("that doesn't matter", "doesn")):
            self.assertNotIn(stem, _distinctive_tokens(_content_tokens(frag)), frag)


# ---- Fix: consensus path lacked the distinctive-token gate -------------------
class TestConsensusPartitionDistinctiveGate(unittest.TestCase):
    """CONSENSUS_PARTITION_NO_DISTINCTIVE_GATE (round-5 panel): the consensus path
    (partition_claims/cluster_claims) must apply the SAME distinctive-token gate the
    contradiction path got in rounds 3-4, else filler-only phatic near-mirrors form bogus
    cross-video SUPPORTS edges and inflate the consensus_clusters metric (surfaced via
    MCP/API). The genuine cosine (cos_equiv) branch is untouched."""

    def _claims(self, *pairs):
        from memovox.loom.models import Claim
        return [Claim(f"{vid}#m{i}.c0", f"{vid}#m{i}", vid, t, salience=0.5,
                      t_start_s=0.0, t_end_s=5.0) for i, (vid, t) in enumerate(pairs)]

    def test_filler_only_pair_not_grouped(self):
        from memovox.loom.consensus import partition_claims
        claims = self._claims(("yt:a", "okay let us try one thing today"),
                              ("yt:b", "okay let us try one thing now"))
        _groups, cross = partition_claims(claims)
        self.assertEqual(cross, [], "filler-only phatic pair wrongly grouped as consensus")

    def test_but_only_overlap_not_grouped(self):
        from memovox.loom.consensus import partition_claims
        claims = self._claims(("yt:a", "yes but I don't really know"),
                              ("yt:b", "well yes but I don't know honestly"))
        _groups, cross = partition_claims(claims)
        self.assertEqual(cross, [], "'but'-only phatic pair wrongly grouped")

    def test_distinctive_pair_still_grouped(self):
        from memovox.loom.consensus import partition_claims
        claims = self._claims(
            ("yt:a", "scaling laws hold beyond current compute budgets"),
            ("yt:b", "scaling laws hold beyond current compute budgets entirely"))
        _groups, cross = partition_claims(claims)
        self.assertEqual(len(cross), 1, "distinctive cross-video near-mirror dropped")

    def test_cluster_claims_defaults_to_no_edge_write(self):
        # defense-in-depth: the default must be write_edges=False so a future caller using
        # the default never writes consensus edges (both live callers pass it explicitly).
        import inspect
        from memovox.loom.consensus import cluster_claims
        self.assertIs(
            inspect.signature(cluster_claims).parameters["write_edges"].default, False)


if __name__ == "__main__":
    unittest.main()
