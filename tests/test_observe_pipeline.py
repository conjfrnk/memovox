"""W4 (M0.1) — the tracer is wired into ingest / ask / consolidate stages.

A caller may pass its own Tracer to inspect the per-stage spans; otherwise each
operation makes one internally. No control flow changes — purely wrapping — so
every existing test/gate stays green.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import augur, pipeline
from memovox.backends import get_embedder, get_nli
from memovox.config import Config, Settings
from memovox.loom.consolidate import consolidate
from memovox.loom.store import LoomStore
from memovox.observe import Tracer

VTT = """WEBVTT

00:00:10.000 --> 00:00:20.000
The recommended chunk size is 512 tokens for retrieval.

00:00:20.000 --> 00:00:30.000
Hybrid retrieval combines dense and sparse search for the best recall.
"""


class IngestTracerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.config = Config(store=self.dir / "store", settings=Settings(
            embed_backend="hashing", nli_backend="lexical", llm_backend="none")).ensure()
        vtt = self.dir / "rag.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        self.vtt = str(vtt)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ingest_emits_reconciling_stage_spans(self):
        tracer = Tracer("ingest")
        report = pipeline.ingest(self.config, self.vtt,
                                 source_url="https://youtu.be/abc123", tracer=tracer)
        self.assertEqual(report.status, "ingested")
        stages = {s.stage for s in tracer.spans}
        for expected in ("asr", "visual", "moments", "embed", "claims", "resolve", "digest"):
            self.assertIn(expected, stages, f"missing span {expected}")
        for s in tracer.spans:
            self.assertEqual(s.status, "ok")
            self.assertGreaterEqual(s.wall_ms, 0.0)
        # counter reconciliation: committed + unsupported == claims
        claims = tracer.find("claims")
        self.assertIsNotNone(claims)
        self.assertEqual(
            claims.counters.get("committed", 0) + claims.counters.get("unsupported", 0),
            claims.counters.get("claims", 0),
        )
        self.assertEqual(claims.counters.get("claims", 0),
                         report.n_claims_committed + report.n_claims_unsupported)


class AskConsolidateTracerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.config = Config(store=self.dir / "store", settings=Settings(
            embed_backend="hashing", nli_backend="lexical", llm_backend="none")).ensure()
        vtt = self.dir / "rag.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        pipeline.ingest(self.config, str(vtt), source_url="https://youtu.be/abc123")

    def tearDown(self):
        self._tmp.cleanup()

    def test_ask_emits_retrieve_and_synthesize_spans(self):
        tracer = Tracer("ask")
        with LoomStore(self.config) as store:
            embedder = get_embedder("hashing", config=self.config)
            ans = augur.ask(store, "what chunk size is recommended?",
                            embedder=embedder, settings=self.config.settings, tracer=tracer)
        self.assertTrue(ans.citations)
        stages = {s.stage for s in tracer.spans}
        self.assertIn("retrieve", stages)
        self.assertIn("synthesize", stages)

    def test_consolidate_emits_span_with_max_claims_cap(self):
        # This is a *wiring* test: it proves consolidate() threads its span into
        # find_contradictions so the max_claims cap is recorded on the trace.
        # The cap's dropped-count semantics (byte-identity + dropped>0 when the
        # corpus exceeds the limit) are exercised in test_caps.FindContradictionsCapTest.
        tracer = Tracer("consolidate")
        with LoomStore(self.config) as store:
            nli = get_nli("lexical", config=self.config)
            consolidate(store, nli=nli, settings=self.config.settings, tracer=tracer)
        contra = tracer.find("contradictions")
        self.assertIsNotNone(contra)
        cap = next((c for c in contra.caps if c["name"] == "max_claims"), None)
        self.assertIsNotNone(cap)
        self.assertEqual(cap["limit"], 600)  # the live default threaded through


class MetricsPersistenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.config = Config(store=self.dir / "store", settings=Settings(
            embed_backend="hashing", nli_backend="lexical", llm_backend="none")).ensure()
        vtt = self.dir / "rag.en.vtt"
        vtt.write_text(VTT, encoding="utf-8")
        self.report = pipeline.ingest(self.config, str(vtt),
                                      source_url="https://youtu.be/abc123")

    def tearDown(self):
        self._tmp.cleanup()

    def test_ingest_report_carries_metrics(self):
        d = self.report.to_dict()
        self.assertIn("metrics", d)
        self.assertIn("spans", d["metrics"])
        stages = {s["stage"] for s in d["metrics"]["spans"]}
        self.assertIn("claims", stages)

    def test_stage_metrics_and_ledger_persisted(self):
        with LoomStore(self.config) as store:
            rows = store.stage_metrics(self.report.video_id)
            self.assertTrue(rows)
            row = rows[0]
            self.assertEqual(set(row), {"stage", "wall_ms", "counters", "caps", "recorded_at"})
            ledger = store.metrics_ledger()
            self.assertGreaterEqual(ledger.get("moments", 0), 1)
            self.assertGreaterEqual(ledger.get("videos", 0), 1)
            self.assertIn("claims_committed", ledger)

    def test_ledger_accumulates_across_ingests(self):
        vtt2 = self.dir / "two.en.vtt"
        vtt2.write_text(VTT.replace("512 tokens", "256 tokens"), encoding="utf-8")
        pipeline.ingest(self.config, str(vtt2), source_url="https://youtu.be/def456")
        with LoomStore(self.config) as store:
            self.assertGreaterEqual(store.metrics_ledger().get("videos", 0), 2)

    def test_answer_metrics_additive_byte_identity(self):
        with LoomStore(self.config) as store:
            embedder = get_embedder("hashing", config=self.config)
            ans = augur.ask(store, "what chunk size is recommended?",
                            embedder=embedder, settings=self.config.settings)
        d = ans.to_dict()
        self.assertIn("metrics", d)
        # the new key is purely additive — removing it leaves today's exact shape
        d.pop("metrics")
        self.assertEqual(set(d), {"text", "strategy", "low_evidence", "citations"})

    def test_consolidation_report_carries_metrics(self):
        from memovox.loom.consolidate import consolidate
        with LoomStore(self.config) as store:
            nli = get_nli("lexical", config=self.config)
            rep = consolidate(store, nli=nli, settings=self.config.settings)
        self.assertIn("metrics", rep.to_dict())


if __name__ == "__main__":
    unittest.main()
