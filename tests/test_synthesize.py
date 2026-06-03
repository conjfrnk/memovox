"""W4 — corpus-level synthesis / "literature review" (Phase 3, spec §5).

synthesize() composes a grounded, every-sentence-cited synthesis of what the
corpus says about a topic: it surfaces cross-source CONSENSUS (claims that agree
across videos) and DISAGREEMENTS (NLI contradictions), keeping the two apart even
when a contradiction is lexically near-identical to its negation (the lexical NLI
flags the polarity flip, so it never masquerades as consensus).
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.augur.synthesize import Synthesis, synthesize
from memovox.backends.embed import HashingEmbedder
from memovox.backends.nli import LexicalNLI
from memovox.config import Config
from memovox.loom import Claim, LoomStore, Moment, Video


class SynthesizeTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.nli = LexicalNLI()
        for vid in ("vid:a", "vid:b"):
            self.store.upsert_video(Video(video_id=vid, source_url=f"https://x/{vid}",
                                          title=vid, content_hash=vid))

    def _add(self, cid, vid, text, *, salience=0.6, idx=0):
        mid = f"{vid}#m{idx:04d}"
        if self.store.get_moment(mid) is None:
            self.store.add_moment(Moment(mid, vid, float(idx * 10), float(idx * 10 + 10),
                                         text, "spk_0", index=idx),
                                  self.emb.embed_one(text))
        self.store.add_claim(Claim(claim_id=cid, moment_id=mid, video_id=vid, text=text,
                                   subject=text, salience=salience, t_start_s=float(idx * 10),
                                   t_end_s=float(idx * 10 + 10), speaker_id="spk_0",
                                   status="committed"))

    def _corpus(self):
        # Genuine cross-video consensus (same claim, both talks).
        self._add("a.con", "vid:a", "Chinchilla scaling needs more training tokens.", idx=0)
        self._add("b.con", "vid:b", "Chinchilla scaling needs more training tokens.", idx=0)
        # A real disagreement (lexically near-identical, opposite polarity).
        self._add("a.dis", "vid:a", "Scaling laws will hold beyond current compute budgets.", idx=1)
        self._add("b.dis", "vid:b", "Scaling laws will not hold beyond current compute budgets.", idx=1)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class TestSynthesize(SynthesizeTestBase):
    def test_surfaces_cross_video_consensus(self):
        self._corpus()
        syn = synthesize(self.store, "scaling chinchilla", nli=self.nli)
        self.assertIsInstance(syn, Synthesis)
        chinchilla = [cp for cp in syn.consensus_points if "chinchilla" in cp["text"].lower()]
        self.assertEqual(len(chinchilla), 1)
        self.assertEqual(chinchilla[0]["support_count"], 2)

    def test_surfaces_contradiction(self):
        self._corpus()
        syn = synthesize(self.store, "scaling chinchilla", nli=self.nli)
        self.assertTrue(syn.contradictions)
        self.assertTrue(any(c["relation"] == "CONTRADICTS" for c in syn.contradictions))

    def test_contradiction_not_reported_as_consensus(self):
        self._corpus()
        syn = synthesize(self.store, "scaling chinchilla", nli=self.nli)
        # The "scaling laws hold/not hold" pair is token-equivalent but is a
        # contradiction — it must NOT appear as a consensus point.
        for cp in syn.consensus_points:
            self.assertNotIn("will hold", cp["text"].lower())
            self.assertNotIn("will not hold", cp["text"].lower())

    def test_every_text_sentence_is_cited(self):
        self._corpus()
        syn = synthesize(self.store, "scaling chinchilla", nli=self.nli)
        self.assertTrue(syn.citations)
        import re
        from memovox.util import split_sentences
        for sentence in split_sentences(syn.text):
            self.assertTrue(re.search(r"\[\d+\]", sentence),
                            f"uncited synthesis sentence: {sentence!r}")

    def test_low_evidence_when_topic_absent(self):
        self._corpus()
        syn = synthesize(self.store, "quantum chromodynamics", nli=self.nli)
        self.assertTrue(syn.low_evidence)

    def test_serializable(self):
        self._corpus()
        syn = synthesize(self.store, "scaling chinchilla", nli=self.nli)
        d = syn.to_dict()
        self.assertEqual(d["topic"], "scaling chinchilla")
        self.assertIn("consensus_points", d)
        self.assertIn("contradictions", d)
        self.assertIn("citations", d)


if __name__ == "__main__":
    unittest.main()
