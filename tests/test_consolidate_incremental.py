"""M0.2 W5 — incremental + observable consolidation.

Incremental consolidation (watermark over claim rowid; scan NEW claims vs ALL
committed) must produce the IDENTICAL graph as a single full pass, must report
scanned/skipped/capped instead of silently truncating, and must do zero new work
when there are no new claims.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import pipeline
from memovox.backends import get_nli
from memovox.backends.embed import HashingEmbedder
from memovox.config import Config, Settings
from memovox.loom import Claim, Moment, Video
from memovox.loom.consolidate import consolidate
from memovox.loom.store import LoomStore

_GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "eval" / "golden"
_FREE = dict(embed_backend="hashing", nli_backend="lexical", asr_backend="captions",
             llm_backend="none", vlm_backend="none", ocr_backend="none", entity_backend="none")


def _edge_set(store):
    out = set()
    for rel in ("CONTRADICTS", "SUPPORTS"):
        for e in store.edges(rel=rel):
            out.add((e["src"], e["rel"], e["dst"], e["video_id"]))
    return out


class IncrementalConsolidateTest(unittest.TestCase):
    def _config(self, name):
        cfg = Config(store=self.dir / name, settings=Settings(**_FREE)).ensure()
        return cfg

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.talk_a = str(_GOLDEN / "talk_a.en.vtt")
        self.talk_b = str(_GOLDEN / "talk_b.en.vtt")

    def tearDown(self):
        self._tmp.cleanup()

    def test_incremental_equals_full(self):
        # FULL: ingest both, one cold consolidate.
        full_cfg = self._config("full")
        pipeline.ingest(full_cfg, self.talk_a, source_url="https://x/a")
        pipeline.ingest(full_cfg, self.talk_b, source_url="https://x/b")
        with LoomStore(full_cfg) as store:
            consolidate(store, nli=get_nli("lexical", config=full_cfg))
            full_edges = _edge_set(store)

        # INCREMENTAL: ingest a, consolidate; ingest b, consolidate.
        inc_cfg = self._config("inc")
        pipeline.ingest(inc_cfg, self.talk_a, source_url="https://x/a")
        with LoomStore(inc_cfg) as store:
            consolidate(store, nli=get_nli("lexical", config=inc_cfg))
        pipeline.ingest(inc_cfg, self.talk_b, source_url="https://x/b")
        with LoomStore(inc_cfg) as store:
            consolidate(store, nli=get_nli("lexical", config=inc_cfg))
            inc_edges = _edge_set(store)

        self.assertEqual(inc_edges, full_edges)
        self.assertTrue(full_edges, "expected at least one cross-video edge in the golden corpus")

    def test_report_counts_scanned_skipped_capped(self):
        cfg = self._config("cap")
        pipeline.ingest(cfg, self.talk_a, source_url="https://x/a")
        pipeline.ingest(cfg, self.talk_b, source_url="https://x/b")
        with LoomStore(cfg) as store:
            rep = consolidate(store, nli=get_nli("lexical", config=cfg),
                              since_watermark=0, max_claims=1)
        self.assertTrue(rep.capped)
        self.assertEqual(rep.claims_scanned, 1)
        self.assertGreater(rep.claims_skipped, 0)

    def test_watermark_advances_and_is_idempotent(self):
        cfg = self._config("wm")
        pipeline.ingest(cfg, self.talk_a, source_url="https://x/a")
        pipeline.ingest(cfg, self.talk_b, source_url="https://x/b")
        with LoomStore(cfg) as store:
            rep1 = consolidate(store, nli=get_nli("lexical", config=cfg))
            wm1 = store.get_meta("consolidation_watermark")
            edges1 = _edge_set(store)
            # second pass, no new claims -> zero new edges/NLI, watermark unchanged
            rep2 = consolidate(store, nli=get_nli("lexical", config=cfg))
            self.assertEqual(store.get_meta("consolidation_watermark"), wm1)
            self.assertEqual(_edge_set(store), edges1)
            # report totals are stable across the idempotent re-run...
            self.assertEqual(rep2.contradictions, rep1.contradictions)
            self.assertEqual(rep2.supports, rep1.supports)
            # ...and the second pass did zero NEW NLI work (sparse-scope no-op)
            contra_span = next(s for s in rep2.metrics["spans"] if s["stage"] == "contradictions")
            self.assertEqual(contra_span["counters"].get("new_contradictions", 0), 0)
            self.assertEqual(contra_span["counters"].get("nli_calls", 0), 0)


class ThreeVideoSyntheticTest(unittest.TestCase):
    """Insurance against the '2-video corpus masks a bug' concern: 3 cross-video
    claims exercise new-vs-MULTIPLE-old pairing, which 2 videos cannot."""

    _TEXT = {
        "a": "the model has exactly one hundred layers",
        "b": "the model does not have exactly one hundred layers",  # contradicts a & c
        "c": "the model has exactly one hundred layers",            # agrees a, contradicts b
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.emb = HashingEmbedder(dim=256)

    def tearDown(self):
        self._tmp.cleanup()

    def _store(self, name):
        cfg = Config(store=self.dir / name, settings=Settings(**_FREE)).ensure()
        return LoomStore(cfg), cfg

    def _add(self, store, key):
        vid = f"vid_{key}"
        store.upsert_video(Video(video_id=vid, source_url=f"https://x/{key}",
                                 title=vid, content_hash=vid))
        mid = f"{vid}#m0000"
        store.add_moment(Moment(mid, vid, 0.0, 5.0, self._TEXT[key], "spk", index=0),
                         self.emb.embed_one(self._TEXT[key]))
        store.add_claim(Claim(claim_id=f"{vid}.c0", moment_id=mid, video_id=vid,
                              text=self._TEXT[key], subject="model", claim_type="FACT",
                              status="committed", t_start_s=0.0, t_end_s=5.0, speaker_id="spk"))

    def test_incremental_equals_full_three_videos(self):
        nli = get_nli("lexical", config=Config(store=self.dir / "n").ensure())

        full, _ = self._store("full")
        for key in ("a", "b", "c"):
            self._add(full, key)
        consolidate(full, nli=nli, since_watermark=0)
        full_edges = _edge_set(full)
        full.close()

        inc, _ = self._store("inc")
        for key in ("a", "b", "c"):
            self._add(inc, key)
            consolidate(inc, nli=nli)  # incremental after each new video
        inc_edges = _edge_set(inc)
        inc.close()

        self.assertEqual(inc_edges, full_edges)
        self.assertTrue(full_edges, "synthetic corpus should yield cross-video edges")


if __name__ == "__main__":
    unittest.main()
