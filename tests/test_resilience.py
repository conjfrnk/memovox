"""Bulletproofing: graceful degradation when an OPTIONAL backend is present but its
model fails to load (offline / uncached / OOM / corrupt).

The discipline is "every model+storage slot has a deterministic fallback". A machine
that has, say, sentence-transformers installed but no cached model and no network
must NOT crash — ``auto`` selection degrades to the free backend (with a stderr
warning). Explicit selection of a broken optional backend still fails loud (the
user opted in). These tests mock the MODEL-LOAD point (the real offline failure
site) and assert the CONTRACT (behavior), not the implementation.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends import embed as embed_mod
from memovox.backends import get_embedder, get_nli, get_reranker
from memovox.backends import nli as nli_mod
from memovox.backends import rerank as rerank_mod


def _force_available(cls):
    return mock.patch.object(cls, "is_available", classmethod(lambda c: True))


def _reset_degraded():
    # The auto-fallback memoizes validated/degraded backends process-globally; clear
    # it so a test that forces (de)gradation can't leak that decision into another.
    from memovox.backends import _reset_auto_memo
    _reset_auto_memo()


class AutoFallsBackWhenModelUnloadableTest(unittest.TestCase):
    """auto + optional-present-but-broken -> a WORKING free backend, no crash."""

    def setUp(self):
        _reset_degraded()

    def tearDown(self):
        _reset_degraded()

    def test_embedder_auto_degrades_when_model_load_fails(self):
        with _force_available(embed_mod.SentenceTransformerEmbedder), \
                mock.patch.object(embed_mod.SentenceTransformerEmbedder, "_load",
                                  side_effect=OSError("offline: cannot fetch BAAI/bge-m3")):
            emb = get_embedder("auto")
            vec = emb.embed_one("the recommended chunk size is 512 tokens")
        self.assertTrue(vec and any(x != 0.0 for x in vec))  # produced a real vector

    def test_nli_auto_degrades_when_model_load_fails(self):
        with _force_available(nli_mod.TransformersNLI), \
                mock.patch.object(nli_mod.TransformersNLI, "_pipe",
                                  side_effect=OSError("offline: cannot fetch deberta")):
            nli = get_nli("auto")
            res = nli.classify("scaling laws hold", "scaling laws hold")
        self.assertTrue(0.0 <= res.entail <= 1.0)  # produced a real result

    def test_reranker_auto_degrades_when_model_load_fails(self):
        with _force_available(rerank_mod.CrossEncoderReranker), \
                mock.patch.object(rerank_mod.CrossEncoderReranker, "_model",
                                  side_effect=OSError("offline: cannot fetch ms-marco")):
            r = get_reranker("auto")
            cands = [("m2", 0.9), ("m1", 0.8)]
            out = r.rerank("q", cands, texts={"m2": "a", "m1": "b"})
        self.assertEqual(set(m for m, _ in out), {"m1", "m2"})  # same set, no crash


class TestHarnessHermeticityTest(unittest.TestCase):
    """Every *_backend Settings slot MUST be pinned by tests/__init__.py, so the suite
    is hermetic on ANY machine (incl. one with the optional ML deps installed)."""

    def test_every_backend_slot_is_pinned_in_env(self):
        import dataclasses
        import os

        from memovox.config import Settings
        slots = [f.name for f in dataclasses.fields(Settings) if f.name.endswith("_backend")]
        self.assertTrue(slots)
        missing = [s for s in slots if f"MEMOVOX_{s.upper()}" not in os.environ]
        self.assertEqual(missing, [], f"unpinned backend slots in tests/__init__.py: {missing}")

    def test_pinned_to_free_values(self):
        from memovox.config import Settings
        s = Settings.from_env()
        self.assertEqual(s.embed_backend, "hashing")
        self.assertEqual(s.nli_backend, "lexical")
        self.assertEqual(s.rerank_backend, "none")
        self.assertEqual(s.visual_embed_backend, "signature")


class ExplicitSelectionStillFailsLoudTest(unittest.TestCase):
    """Explicit selection of a broken optional backend must NOT silently degrade."""

    def test_explicit_embedder_propagates_model_error(self):
        with _force_available(embed_mod.SentenceTransformerEmbedder), \
                mock.patch.object(embed_mod.SentenceTransformerEmbedder, "_load",
                                  side_effect=OSError("offline")):
            emb = get_embedder("sentence-transformers")  # opted in
            with self.assertRaises(OSError):
                emb.embed_one("x")  # fails loud, no hidden fallback


class UnimplementedSkeletonsFailCleanTest(unittest.TestCase):
    """Explicit selection of an unimplemented named-backend skeleton must fail with a
    clean BackendUnavailable at the factory, NEVER a NotImplementedError mid-pipeline."""

    def test_skeletons_report_unavailable(self):
        from memovox.backends.asr_align import WhisperXAlign
        from memovox.backends.diarize_turns import PyannoteTurns
        from memovox.backends.ocr import SuryaOCR
        from memovox.backends.vlm import Qwen25VL
        from memovox.backends.visual_embed import ColPaliVisualEmbedder
        for cls in (Qwen25VL, SuryaOCR, ColPaliVisualEmbedder, WhisperXAlign, PyannoteTurns):
            self.assertFalse(cls.is_available(), cls.__name__)

    def test_explicit_selection_raises_backend_unavailable(self):
        from memovox.backends import get_ocr, get_vlm, get_visual_embedder
        from memovox.errors import BackendUnavailable
        for getter, name in [(get_vlm, "qwen2.5-vl"), (get_ocr, "surya"),
                             (get_visual_embedder, "colpali")]:
            with self.assertRaises(BackendUnavailable):
                getter(name)


class EndToEndHostileEnvTest(unittest.TestCase):
    """A full ingest+ask succeeds even when an optional backend is present-but-broken."""

    def setUp(self):
        _reset_degraded()

    def tearDown(self):
        _reset_degraded()

    def test_memovox_ingest_ask_survives_broken_optional_backends(self):
        import contextlib

        from memovox import Memovox
        broken = [
            _force_available(embed_mod.SentenceTransformerEmbedder),
            mock.patch.object(embed_mod.SentenceTransformerEmbedder, "_load", side_effect=OSError("offline")),
            _force_available(nli_mod.TransformersNLI),
            mock.patch.object(nli_mod.TransformersNLI, "_pipe", side_effect=OSError("offline")),
        ]
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            for cm in broken:
                stack.enter_context(cm)
            vtt = pathlib.Path(tmp) / "t.en.vtt"
            vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:09.000\nChunk size is 512 tokens.\n",
                           encoding="utf-8")
            # Force auto explicitly (the harness env pins free backends) so the
            # broken optional backends are actually exercised.
            mv = Memovox(store=pathlib.Path(tmp) / "store", llm_backend="none",
                         embed_backend="auto", nli_backend="auto")
            rep = mv.ingest(str(vtt), source_url="https://x/a")
            self.assertEqual(rep.n_moments, 1)
            ans = mv.ask("chunk size?")
            self.assertTrue(ans.citations)  # answered despite broken optional backends


class AsrWhisperFallsBackToCaptionsTest(unittest.TestCase):
    """A whisper model-load/transcription failure degrades to captions (when present)
    instead of aborting ingest with an opaque faster-whisper error."""

    def test_whisper_failure_uses_captions(self):
        import pathlib
        from memovox.backends.asr_whisper import WhisperASR
        from memovox.config import Config
        from memovox.stentor import asr as asr_mod
        from memovox.stentor.acquire import SourceMeta
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(store=pathlib.Path(tmp) / "s").ensure()
            vtt = pathlib.Path(tmp) / "c.vtt"
            vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nHello from captions.\n",
                           encoding="utf-8")
            meta = SourceMeta(source_url=None, title="t", media_path=pathlib.Path(tmp) / "m.mp4",
                              captions_path=vtt)
            with mock.patch.object(WhisperASR, "is_available", classmethod(lambda c: True)), \
                    mock.patch.object(asr_mod, "_prepare_audio",
                                      return_value=pathlib.Path(tmp) / "x.wav"), \
                    mock.patch.object(WhisperASR, "transcribe",
                                      side_effect=OSError("offline: whisper weights")):
                result = asr_mod.run_asr(config, meta, backend="whisper")  # must not raise
            self.assertTrue(result.segments)  # captions used
            self.assertIn("captions", result.segments[0].text.lower())

    def test_device_placement_error_is_not_swallowed(self):
        # the §9 fail-loud guard must surface, NOT be downgraded to captions
        import pathlib
        from memovox.backends.asr_whisper import WhisperASR
        from memovox.config import Config
        from memovox.errors import DevicePlacementError
        from memovox.stentor import asr as asr_mod
        from memovox.stentor.acquire import SourceMeta
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(store=pathlib.Path(tmp) / "s").ensure()
            vtt = pathlib.Path(tmp) / "c.vtt"
            vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nHi.\n", encoding="utf-8")
            meta = SourceMeta(source_url=None, title="t", media_path=pathlib.Path(tmp) / "m.mp4",
                              captions_path=vtt)
            with mock.patch.object(WhisperASR, "is_available", classmethod(lambda c: True)), \
                    mock.patch.object(asr_mod, "_prepare_audio",
                                      return_value=pathlib.Path(tmp) / "x.wav"), \
                    mock.patch.object(WhisperASR, "transcribe",
                                      side_effect=DevicePlacementError("large model on CPU")):
                with self.assertRaises(DevicePlacementError):  # propagates despite captions
                    asr_mod.run_asr(config, meta, backend="whisper")


class BadInputFailsCleanTest(unittest.TestCase):
    """Bad inputs on the public API must fail with a CLEAR error, never a cryptic
    crash deep in the pipeline."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        from memovox import Memovox
        self.mv = Memovox(store=self.dir / "store", llm_backend="none")

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_and_directory_paths_raise_acquisition_error(self):
        from memovox.errors import AcquisitionError
        for bad in ("", "   ", str(self.dir)):  # empty, whitespace, a directory
            with self.assertRaises(AcquisitionError):
                self.mv.ingest(bad)

    def test_non_dict_subscriptions_does_not_crash(self):
        (self.dir / "store").mkdir(parents=True, exist_ok=True)
        self.mv.config.subscriptions_path.write_text("[1, 2, 3]", encoding="utf-8")  # a list
        self.assertEqual(self.mv.list_subscriptions(), [])  # ignored, no AttributeError

    def test_malformed_config_json_falls_back_to_defaults(self):
        from memovox import Memovox
        store = self.dir / "store2"
        store.mkdir(parents=True, exist_ok=True)
        (store / "config.json").write_text("{not json", encoding="utf-8")
        mv = Memovox(store=store)  # must not crash
        self.assertFalse(mv.settings.local_only)  # defaults applied


if __name__ == "__main__":
    unittest.main()
