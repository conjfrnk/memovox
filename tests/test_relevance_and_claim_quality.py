"""W5 stress-hardening: out-of-corpus relevance gate, low-value-claim demotion,
and lexical-NLI negation coverage. Regression tests for the stress-test findings."""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import augur
from memovox.assay import run as assay_run
from memovox.assay.claims import (
    is_low_value_claim,
    is_non_claim,
    is_sentence_fragment,
    transcript_is_punctuated,
)
from memovox.backends.embed import HashingEmbedder
from memovox.backends.nli import LexicalNLI
from memovox.config import Config, Settings
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.models import STATUS_COMMITTED, STATUS_UNSUPPORTED


class TestRelevanceGate(unittest.TestCase):
    """W5.1: an out-of-corpus question must be refused (low_evidence, no citations)
    rather than answered with the nearest-but-irrelevant moments — the 'no citation,
    no claim' promise. In-corpus questions stay answered."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.store.upsert_video(Video("yt:v", "https://youtu.be/v", "talk"))
        # Enough moments (> answer_relevance_min_moments) for IDF to be meaningful.
        for i in range(55):
            m = Moment(f"yt:v#m{i:04d}", "yt:v", float(i), float(i) + 1.0,
                       f"a generic remark number {i} on assorted everyday topics.",
                       speaker_id="spk_0", index=i)
            self.store.add_moment(m, self.emb.embed_one(m.text_for_embedding()))
        # A genuine TOPIC recurs across several moments (df above the topicality
        # floor) — unlike an incidental hapax. A real video mentions its subject many
        # times, so this mirrors reality.
        for i in range(6):
            d = Moment(f"yt:v#m90{i:02d}", "yt:v", 900.0 + i, 930.0 + i,
                       "Photosynthesis converts sunlight into chemical energy inside "
                       "chloroplasts during the light reactions of photosynthesis.",
                       speaker_id="spk_0", index=900 + i)
            self.store.add_moment(d, self.emb.embed_one(d.text_for_embedding()))
        # An INCIDENTAL hapax token ("capital", one moment) — must NOT let an
        # out-of-corpus question whose distinctive topic is absent pass the gate.
        e = Moment("yt:v#m9100", "yt:v", 980.0, 990.0,
                   "The capital expenditure that quarter was unusually high.",
                   speaker_id="spk_0", index=910)
        self.store.add_moment(e, self.emb.embed_one(e.text_for_embedding()))
        # A second video whose TITLE names a speaker NOT spoken in the transcript;
        # its topic (glucose metabolism) recurs across moments.
        self.store.upsert_video(Video("yt:w", "https://youtu.be/w",
                                      "Dr Jane Halvorsen explains glucose metabolism"))
        for i in range(5):
            g = Moment(f"yt:w#m00{i:02d}", "yt:w", float(i), float(i) + 30.0,
                       "Glucose metabolism produces ATP through cellular respiration and "
                       "glucose oxidation.", speaker_id="spk_0", index=i)
            self.store.add_moment(g, self.emb.embed_one(g.text_for_embedding()))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_in_corpus_question_is_answered(self):
        ans = augur.ask(self.store, "how does photosynthesis convert sunlight into energy?",
                        embedder=self.emb, settings=Settings())
        self.assertFalse(ans.low_evidence)
        self.assertTrue(ans.citations)

    def test_out_of_corpus_question_is_refused(self):
        ans = augur.ask(self.store, "what is the capital of Mongolia?",
                        embedder=self.emb, settings=Settings())
        self.assertTrue(ans.low_evidence)
        self.assertEqual(ans.citations, [])

    def test_floor_zero_disables_gate(self):
        ans = augur.ask(self.store, "what is the capital of Mongolia?",
                        embedder=self.emb, settings=Settings(answer_relevance_floor=0.0))
        self.assertFalse(ans.low_evidence)
        self.assertTrue(ans.citations)

    def test_single_incidental_token_is_refused_even_when_retrieved(self):
        # "capital" is an incidental hapax (df=1) that IS retrieved (proven below),
        # but it is not a genuine corpus topic, so the question must be refused — the
        # single-distinctive-token leak the iter2 panel caught ("what time does the
        # bank open?"). Without the gate it would (wrongly) be answered.
        ungated = augur.ask(self.store, "what is the capital of Mongolia?",
                            embedder=self.emb, settings=Settings(answer_relevance_floor=0.0))
        self.assertTrue(any("capital" in (c.snippet or "").lower() for c in ungated.citations),
                        "the incidental 'capital' moment must actually be retrieved")
        gated = augur.ask(self.store, "what is the capital of Mongolia?",
                          embedder=self.emb, settings=Settings())
        self.assertTrue(gated.low_evidence)
        self.assertEqual(gated.citations, [])

    def test_proper_name_in_title_is_answered(self):
        # The speaker name is only in the video TITLE, not the transcript; the topic
        # (glucose) is spoken. Must NOT be wrongly refused (W5.1 over-refusal fix).
        ans = augur.ask(self.store, "what does Jane Halvorsen say about glucose metabolism?",
                        embedder=self.emb, settings=Settings())
        self.assertFalse(ans.low_evidence)
        self.assertTrue(ans.citations)

    def test_polysemy_context_mismatch_refused(self):
        # "energy" recurs in the corpus (photosynthesis: "chemical energy"), but a
        # question about household energy shares only that one word — its context
        # words (save/home) are absent -> refuse. The polysemy defense: the count
        # signal uses CONTEXT words, not just the topic word.
        ans = augur.ask(self.store, "how do I save energy at home?",
                        embedder=self.emb, settings=Settings())
        self.assertTrue(ans.low_evidence)
        self.assertEqual(ans.citations, [])

    def test_title_only_subject_is_answered(self):
        # A distinctive subject that appears ONLY in the video title (df=0 in
        # transcripts) is still a genuine topic -> title-aware topicality keeps it.
        ans = augur.ask(self.store, "what does Halvorsen explain about glucose?",
                        embedder=self.emb, settings=Settings())
        self.assertFalse(ans.low_evidence)
        self.assertTrue(ans.citations)

    def test_small_corpus_is_not_gated(self):
        # Below answer_relevance_min_moments the IDF signal is unreliable; an
        # out-of-corpus query on a tiny store must NOT be spuriously refused.
        with tempfile.TemporaryDirectory() as t2:
            cfg = Config(store=pathlib.Path(t2) / "s").ensure()
            store = LoomStore(cfg)
            store.upsert_video(Video("yt:x", "https://youtu.be/x", "t"))
            m = Moment("yt:x#m0", "yt:x", 0.0, 5.0,
                       "The recommended chunk size is 512 tokens.", speaker_id="spk_0", index=0)
            store.add_moment(m, self.emb.embed_one(m.text_for_embedding()))
            ans = augur.ask(store, "what is the capital of Mongolia?",
                            embedder=self.emb, settings=Settings())
            self.assertTrue(ans.citations)  # not gated (corpus too small to assess)
            store.close()


class TestLowValueClaimFilter(unittest.TestCase):
    """W5.2: greetings, ad reads, navigational imperatives, and continuation
    fragments are demoted (kept as unsupported), never silently dropped."""

    def test_non_claim_predicates(self):
        for t in ("My name is Eric.", "Hello everyone.", "Hey, welcome back.",
                  "Support for NPR Music comes from Capital One.",
                  "Learn more at example.com.", "Subscribe to the channel."):
            self.assertTrue(is_non_claim(t), t)
        for t in ("The Transformer is the foundation of modern language models.",
                  "The study was supported by a grant from the NSF.",
                  "Studies are often sponsored by food companies with an agenda.",
                  "Support groups help patients recover faster.",
                  "Hello everyone, today saturated fat turns out to be more nuanced.",
                  "Look at the data showing that scaling laws hold."):
            self.assertFalse(is_non_claim(t), t)

    def test_fragment_detection(self):
        self.assertTrue(is_sentence_fragment("number in your head would collapse."))
        self.assertFalse(is_sentence_fragment("Numbers can be very large."))

    def test_punctuation_gate_protects_auto_captions(self):
        # All-lowercase unpunctuated auto-caption text is NOT a fragment-checkable
        # cased transcript, so leading-case must not demote it.
        self.assertFalse(transcript_is_punctuated("so today we want to talk about metabolism"))
        self.assertTrue(transcript_is_punctuated("A new study. It changed everything."))
        self.assertTrue(transcript_is_punctuated("The cats sleep."))
        self.assertFalse(transcript_is_punctuated("the cats like to sleep here often."))
        # Fragment demotion is suppressed when the transcript isn't cased/punctuated.
        self.assertFalse(is_low_value_claim("number in your head collapses.", punctuated=False))
        self.assertTrue(is_low_value_claim("number in your head collapses.", punctuated=True))

    def test_run_demotes_fragment_in_cased_transcript(self):
        # A continuation fragment arises when a sentence is split across a Moment
        # boundary, so the Moment's transcript STARTS with a lowercase tail.
        nli = LexicalNLI()
        m = Moment("yt:x#m0", "yt:x", 0.0, 10.0,
                   "number in your head would collapse. The next point is clear here.",
                   speaker_id="spk_0")
        claims = assay_run(m, nli=nli, settings=Settings())
        by_text = {c.text: c.status for c in claims}
        self.assertEqual(by_text.get("number in your head would collapse."), STATUS_UNSUPPORTED)
        self.assertEqual(by_text.get("The next point is clear here."), STATUS_COMMITTED)

    def test_run_keeps_claims_in_lowercase_autocaptions(self):
        # Regression: the fragment rule once demoted EVERY claim of an all-lowercase
        # auto-caption video (each "sentence" starts lowercase). Must not happen.
        nli = LexicalNLI()
        m = Moment("yt:y#m0", "yt:y", 0.0, 10.0,
                   "eating breakfast actually slows your metabolism it does not speed it up",
                   speaker_id="spk_0")
        claims = assay_run(m, nli=nli, settings=Settings())
        self.assertTrue(any(c.status == STATUS_COMMITTED for c in claims))


class TestDocFreq(unittest.TestCase):
    """W5.5: doc_freq must escape LIKE wildcards (else doc_freq('%') matches all)."""

    def test_doc_freq_escapes_wildcards(self):
        with tempfile.TemporaryDirectory() as t:
            store = LoomStore(Config(store=pathlib.Path(t) / "s").ensure())
            store.upsert_video(Video("yt:z", "https://youtu.be/z", "t"))
            store.add_moment(Moment("yt:z#m0", "yt:z", 0.0, 5.0,
                                    "Chinchilla scaling laws govern compute budgets.",
                                    speaker_id="spk_0", index=0))
            self.assertEqual(store.doc_freq("chinchilla"), 1)
            self.assertEqual(store.doc_freq("nonexistentword"), 0)
            self.assertEqual(store.doc_freq("%"), 0)   # wildcard, not "match all"
            self.assertEqual(store.doc_freq("_"), 0)
            store.close()


class TestConsensusCosineFallback(unittest.TestCase):
    """W5.6: an opt-in embedding-cosine fallback groups paraphrases that token-Jaccard
    misses. Off by default (byte-identical free path); fires with real embeddings."""

    def _claims(self):
        from memovox.loom.models import Claim
        # Two videos asserting the same thing in different words: they share only the
        # token 'agi', so token-Jaccard (>=0.5) cannot group them.
        return [
            Claim("yt:a#m0.c0", "yt:a#m0", "yt:a", "AGI is coming very soon", subject="agi"),
            Claim("yt:b#m0.c0", "yt:b#m0", "yt:b", "AGI will arrive imminently", subject="agi"),
        ]

    def test_token_jaccard_alone_does_not_group(self):
        from memovox.loom.consensus import partition_claims
        groups, xv = partition_claims(self._claims())
        self.assertEqual(len(groups), 2)   # not merged
        self.assertEqual(xv, [])

    def test_cosine_fallback_groups_paraphrases(self):
        from memovox.loom.consensus import partition_claims
        claims = self._claims()
        # Synthetic embeddings: near-identical vectors for the two paraphrases.
        vectors = {claims[0].claim_id: [1.0, 0.0, 0.05],
                   claims[1].claim_id: [0.98, 0.0, 0.10]}
        groups, xv = partition_claims(claims, cosine=0.9, vectors=vectors)
        self.assertEqual(len(groups), 1)   # merged into one consensus cluster
        self.assertEqual(len(xv), 1)       # one cross-video agreement pair

    def test_cosine_off_is_noop(self):
        from memovox.loom.consensus import partition_claims
        claims = self._claims()
        vectors = {claims[0].claim_id: [1.0, 0.0], claims[1].claim_id: [1.0, 0.0]}
        groups, _ = partition_claims(claims, cosine=0.0, vectors=vectors)
        self.assertEqual(len(groups), 2)   # disabled -> unchanged


class TestLexicalNliNegation(unittest.TestCase):
    """W5.3: negation-polarity words beyond a bare 'not' (e.g. 'nothing') flip
    polarity so a high-overlap pair is detected as a contradiction."""

    def test_nothing_is_a_negation(self):
        nli = LexicalNLI()
        res = nli.classify(
            "reducing saturated fat does nothing to protect your heart",
            "reducing saturated fat does protect your heart",
        )
        self.assertEqual(res.label, "contradiction")

    def test_aligned_claims_are_not_contradiction(self):
        nli = LexicalNLI()
        res = nli.classify(
            "reducing saturated fat protects your heart",
            "reducing saturated fat does protect your heart",
        )
        self.assertNotEqual(res.label, "contradiction")


if __name__ == "__main__":
    unittest.main()
