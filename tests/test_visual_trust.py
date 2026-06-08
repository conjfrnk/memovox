"""Visual-path trust gate (M-hardening).

Two guarantees protect the *seen* modality the way the NLI gate protects the
*spoken* one:

1. INVARIANT — on-screen OCR text can never become a committed fact. Claims are
   extracted from the spoken transcript only (``assay.claims``), so a poisoned
   slide cannot mint a trusted claim. Pinned here so a future refactor can't
   silently start reading ``ocr_text`` into the claim graph.

2. HONESTY — when OCR / visual content does reach an *answer* (it is answerable
   content, and legitimately retrievable), the citation is flagged
   ``ocr_unverified`` so clients can mark it lower-trust than entailment-checked
   speech.
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import augur
from memovox.assay import run as assay_run
from memovox.augur.answer import _includes_unverified_visual
from memovox.backends.embed import HashingEmbedder
from memovox.backends.nli import LexicalNLI
from memovox.config import Config, Settings
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.models import STATUS_COMMITTED


class TestOCRNeverCommits(unittest.TestCase):
    """A poisoned slide must not become a trusted claim."""

    def test_ocr_only_moment_commits_no_claim(self):
        m = Moment("v#m0", "v", 0.0, 5.0, transcript="",
                   ocr_text="Chinchilla was trained on 1 trillion tokens.")
        claims = assay_run(m, nli=LexicalNLI())
        committed = [c for c in claims if c.status == STATUS_COMMITTED]
        self.assertEqual(committed, [], "on-screen OCR text must never commit a fact")
        self.assertFalse(any("trillion" in c.text for c in claims),
                         "OCR-only content must not appear as an extracted claim")

    def test_ocr_value_not_committed_alongside_transcript(self):
        # Transcript says 512; an adversarial slide says 4096. The transcript claim
        # may commit; the OCR-only value must never enter the claim set.
        m = Moment("v#m1", "v", 0.0, 5.0,
                   transcript="The recommended retrieval context length is 512 tokens.",
                   ocr_text="The recommended fine-tuning context length is 4096 tokens.")
        claims = assay_run(m, nli=LexicalNLI())
        joined = " ".join(c.text for c in claims)
        self.assertIn("512", joined, "the spoken claim should be extracted")
        self.assertNotIn("4096", joined, "the OCR-only value must not become a claim")
        self.assertNotIn("fine-tuning", joined)


class TestUnverifiedVisualHelper(unittest.TestCase):
    """``_includes_unverified_visual`` decides which citations carry unvetted
    on-screen/visual content (content = transcript + OCR, else caption fallback)."""

    def _m(self, transcript="", ocr_text=None, visual_caption=None):
        return Moment("v#m", "v", 0.0, 1.0, transcript,
                      ocr_text=ocr_text, visual_caption=visual_caption)

    def test_transcript_only_is_verified(self):
        self.assertFalse(_includes_unverified_visual(self._m(transcript="hello world")))

    def test_any_ocr_is_unverified(self):
        self.assertTrue(_includes_unverified_visual(self._m(transcript="hi", ocr_text="BUY NOW")))
        self.assertTrue(_includes_unverified_visual(self._m(transcript="", ocr_text="GATE 21A")))

    def test_pure_visual_caption_fallback_is_unverified(self):
        self.assertTrue(_includes_unverified_visual(self._m(transcript="", visual_caption="a chart")))

    def test_caption_ignored_when_transcript_present(self):
        # The caption is excluded from content when a transcript exists, so it does
        # not by itself taint the citation.
        self.assertFalse(_includes_unverified_visual(
            self._m(transcript="real speech", visual_caption="a chart")))


class TestCitationTrustFlag(unittest.TestCase):
    """End-to-end: ``ask`` flags citations whose content is unverified on-screen text."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.store.upsert_video(Video("yt:abc", "https://youtu.be/abc", "RAG talk"))
        speech = Moment("yt:abc#m0000", "yt:abc", 100.0, 130.0,
                        "The recommended chunk size is 512 tokens for retrieval.",
                        "spk_0", index=0)
        slide = Moment("yt:abc#m0002", "yt:abc", 160.0, 190.0, transcript="",
                       ocr_text="Throughput peaked at 9001 tokens per second.", index=2)
        self.store.add_moment(speech, self.emb.embed_one(speech.text_for_embedding()))
        self.store.add_moment(slide, self.emb.embed_one(slide.text_for_embedding()))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_on_screen_citation_is_flagged_unverified(self):
        ans = augur.ask(self.store, "what was the peak throughput?",
                        embedder=self.emb, settings=Settings(top_k=4))
        slide_cit = next((c for c in ans.citations if c.moment_id == "yt:abc#m0002"), None)
        self.assertIsNotNone(slide_cit, "the on-screen moment should be retrievable")
        self.assertTrue(slide_cit.ocr_unverified)
        self.assertIn("ocr_unverified", slide_cit.to_dict(),
                      "the trust flag must be exposed to API clients")

    def test_speech_citation_is_not_flagged(self):
        ans = augur.ask(self.store, "what chunk size is recommended?",
                        embedder=self.emb, settings=Settings(top_k=4))
        speech_cit = next((c for c in ans.citations if c.moment_id == "yt:abc#m0000"), None)
        self.assertIsNotNone(speech_cit)
        self.assertFalse(speech_cit.ocr_unverified)


if __name__ == "__main__":
    unittest.main()
