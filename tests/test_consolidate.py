"""W5 — consolidation as a background job + dedup/decay (Phase 3, spec §4.7).

``consolidate`` is the corpus-wide background pass that runs topic induction,
NLI-verified contradiction/agreement detection, consensus clustering, and dedup —
moved OFF the per-video ingest path (spec §4 stage 7: "runs as a background job
as the library grows").

``dedup_claims`` is the first LIVE caller of the W1.4 ``supersede_claim``
lifecycle. It is conservative: it supersedes only (a) within-video EXACT
duplicates and (b) the corrected side of a ``CORRECTS`` edge — it NEVER supersedes
a cross-video equivalent (that is consensus evidence, not a duplicate).
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.embed import HashingEmbedder
from memovox.backends.nli import LexicalNLI
from memovox.config import Config
from memovox.loom import Claim, LoomStore, Moment, Video
from memovox.loom.consolidate import ConsolidationReport, consolidate, dedup_claims
from memovox.loom.models import STATUS_COMMITTED, STATUS_SUPERSEDED


class ConsolidateTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.nli = LexicalNLI()
        for vid in ("vid:a", "vid:b"):
            self.store.upsert_video(Video(video_id=vid, source_url=f"https://x/{vid}",
                                          title=vid, content_hash=vid))

    def _add(self, cid, vid, text, *, idx=0, claim_type="FACT", status=STATUS_COMMITTED,
             t0=None):
        mid = f"{vid}#m{idx:04d}"
        if self.store.get_moment(mid) is None:
            self.store.add_moment(Moment(mid, vid, float(idx * 10), float(idx * 10 + 10),
                                         text, "spk_0", index=idx),
                                  self.emb.embed_one(text))
        t0 = float(idx * 10) if t0 is None else t0
        self.store.add_claim(Claim(claim_id=cid, moment_id=mid, video_id=vid, text=text,
                                   subject=text, claim_type=claim_type, status=status,
                                   t_start_s=t0, t_end_s=t0 + 5, speaker_id="spk_0"))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class TestDedupClaims(ConsolidateTestBase):
    def test_within_video_exact_duplicate_superseded(self):
        self._add("a.0", "vid:a", "The model has 100 layers.", idx=0, t0=0.0)
        self._add("a.1", "vid:a", "the model has 100 layers", idx=1, t0=10.0)  # same tokens
        n = dedup_claims(self.store)
        self.assertEqual(n, 1)
        # Earliest kept committed; later duplicate superseded (never deleted).
        self.assertEqual(self.store.get_claim("a.0").status, STATUS_COMMITTED)
        self.assertEqual(self.store.get_claim("a.1").status, STATUS_SUPERSEDED)
        self.assertEqual(self.store.get_claim("a.1").superseded_by, "a.0")

    def test_corrects_edge_supersedes_corrected(self):
        self._add("a.0", "vid:a", "the model has 100 layers", idx=0)
        self._add("a.1", "vid:a", "actually the model has 96 layers", idx=1, claim_type="CORRECTION")
        self.store.add_edge("a.1", "CORRECTS", "a.0", src_type="Claim", dst_type="Claim",
                            video_id="vid:a")
        n = dedup_claims(self.store)
        self.assertEqual(n, 1)
        self.assertEqual(self.store.get_claim("a.0").status, STATUS_SUPERSEDED)
        self.assertEqual(self.store.get_claim("a.0").superseded_by, "a.1")
        self.assertEqual(self.store.get_claim("a.1").status, STATUS_COMMITTED)

    def test_cross_video_equivalent_not_superseded(self):
        # The SAME claim in two videos is consensus evidence — keep BOTH committed.
        self._add("a.0", "vid:a", "Chinchilla needs more training tokens.", idx=0)
        self._add("b.0", "vid:b", "Chinchilla needs more training tokens.", idx=0)
        n = dedup_claims(self.store)
        self.assertEqual(n, 0)
        self.assertEqual(self.store.get_claim("a.0").status, STATUS_COMMITTED)
        self.assertEqual(self.store.get_claim("b.0").status, STATUS_COMMITTED)

    def test_dedup_idempotent(self):
        self._add("a.0", "vid:a", "The model has 100 layers.", idx=0, t0=0.0)
        self._add("a.1", "vid:a", "the model has 100 layers", idx=1, t0=10.0)
        self.assertEqual(dedup_claims(self.store), 1)
        self.assertEqual(dedup_claims(self.store), 0)  # nothing left to supersede


class TestConsolidate(ConsolidateTestBase):
    def _corpus(self):
        self._add("a.con", "vid:a", "Chinchilla scaling needs more training tokens.", idx=0)
        self._add("b.con", "vid:b", "Chinchilla scaling needs more training tokens.", idx=0)
        self._add("a.dis", "vid:a", "Scaling laws will hold beyond current compute budgets.", idx=1)
        self._add("b.dis", "vid:b", "Scaling laws will not hold beyond current compute budgets.", idx=1)

    def test_runs_all_legs(self):
        self._corpus()
        report = consolidate(self.store, nli=self.nli)
        self.assertIsInstance(report, ConsolidationReport)
        self.assertGreaterEqual(report.topics, 1)          # topic induction ran
        self.assertGreaterEqual(report.contradictions, 1)  # scaling hold/not
        self.assertGreaterEqual(report.supports, 1)        # chinchilla agreement
        # Edges actually persisted.
        self.assertTrue(self.store.edges(rel="CONTRADICTS"))
        self.assertTrue(self.store.edges(rel="ABOUT"))

    def test_idempotent(self):
        self._corpus()
        consolidate(self.store, nli=self.nli)
        contradicts1 = len(self.store.edges(rel="CONTRADICTS"))
        about1 = len(self.store.edges(rel="ABOUT"))
        consolidate(self.store, nli=self.nli)
        self.assertEqual(len(self.store.edges(rel="CONTRADICTS")), contradicts1)
        self.assertEqual(len(self.store.edges(rel="ABOUT")), about1)


if __name__ == "__main__":
    unittest.main()
