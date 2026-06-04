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
from memovox.config import Config, Settings
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
            consolidate(store, nli=get_nli("lexical", config=cfg))
            wm1 = store.get_meta("consolidation_watermark")
            edges1 = _edge_set(store)
            # second pass, no new claims -> zero new edges, watermark unchanged
            rep2 = consolidate(store, nli=get_nli("lexical", config=cfg))
            self.assertEqual(store.get_meta("consolidation_watermark"), wm1)
            self.assertEqual(_edge_set(store), edges1)
            self.assertEqual(rep2.contradictions, 0)  # no new claims -> no new NLI work


if __name__ == "__main__":
    unittest.main()
