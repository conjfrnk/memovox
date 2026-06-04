"""W3 (M0.1) — every silent cap surfaces as a structured event, byte-identically.

Each test pins the *new* behavior (a cap event is recorded on the passed span)
AND the *unchanged* behavior (the kept results are byte-identical to a baseline
captured without a span). Provenance is sacred: surfacing a cap must never change
*what* is kept.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.embed import HashingEmbedder
from memovox.backends.nli import LexicalNLI
from memovox.config import Config, Settings
from memovox.loom import Claim, LoomStore, Moment, Video
from memovox.loom.consolidate import find_contradictions
from memovox.observe import Span
from memovox.tessera.frames import FrameSig, sample_frame_signatures
from memovox.tessera.keyframes import select_keyframes
from memovox.tessera.scenes import Scene


def _cap(span, name):
    return next((c for c in span.caps if c["name"] == name), None)


class FindContradictionsCapTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.nli = LexicalNLI()
        for vid in ("vid:a", "vid:b"):
            self.store.upsert_video(Video(video_id=vid, source_url=f"https://x/{vid}",
                                          title=vid, content_hash=vid))
        # 6 committed claims, alternating videos, all sharing tokens.
        for i in range(6):
            vid = "vid:a" if i % 2 == 0 else "vid:b"
            cid = f"{vid}.{i}"
            mid = f"{vid}#m{i:04d}"
            self.store.add_moment(Moment(mid, vid, float(i), float(i) + 5,
                                         "the model has many layers", "spk_0", index=i),
                                  self.emb.embed_one("the model has many layers"))
            self.store.add_claim(Claim(claim_id=cid, moment_id=mid, video_id=vid,
                                       text="the model has many layers", subject="model",
                                       claim_type="FACT", status="committed",
                                       t_start_s=float(i), t_end_s=float(i) + 5,
                                       speaker_id="spk_0"))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_max_claims_cap_event_with_identical_pairs(self):
        baseline = find_contradictions(self.store, nli=self.nli, max_claims=3,
                                       write_edges=False)
        span = Span(stage="consolidate")
        with_span = find_contradictions(self.store, nli=self.nli, max_claims=3,
                                        write_edges=False, span=span)
        # behavior byte-identical for the same max_claims
        self.assertEqual([p.to_dict() for p in baseline],
                         [p.to_dict() for p in with_span])
        # new: a structured cap event surfaced
        cap = _cap(span, "max_claims")
        self.assertIsNotNone(cap)
        self.assertEqual(cap["limit"], 3)
        self.assertGreater(cap["dropped"], 0)  # 6 committed - 3 kept = 3


class SelectKeyframesCapTest(unittest.TestCase):
    def test_per_scene_cap_event_with_identical_kept_indices(self):
        # 21 frames whose every neighbor differs above min_gain => unbounded keep
        # but for the per_scene_cap, which truncates.
        sigs = [FrameSig(t=float(i), vec=[float(i % 2)] * 4) for i in range(21)]
        scenes = [Scene(index=0, start_idx=0, end_idx=20, t_start=0.0, t_end=20.0)]
        baseline = select_keyframes(sigs, scenes, min_gain=0.1, per_scene_cap=3)
        span = Span(stage="visual")
        kept = select_keyframes(sigs, scenes, min_gain=0.1, per_scene_cap=3, span=span)
        self.assertEqual(baseline, kept)  # identical kept-index list
        cap = _cap(span, "per_scene_cap")
        self.assertIsNotNone(cap)
        self.assertEqual(cap["limit"], 3)
        self.assertGreater(cap["dropped"], 0)


class SampleFrameSignaturesCapTest(unittest.TestCase):
    def test_frame_max_cap_event_when_truncating(self):
        side = 2
        cell = side * side  # 4 bytes per frame
        raw = bytes(range(256)) * 1  # 256 bytes => 64 frames worth at cell=4
        with mock.patch("memovox.tessera.frames.audio.which_ffmpeg", return_value="ffmpeg"), \
             mock.patch("memovox.tessera.frames.Path.exists", return_value=True), \
             mock.patch("memovox.tessera.frames.subprocess.run") as run:
            run.return_value = mock.Mock(stdout=raw)
            base = sample_frame_signatures("x.mp4", side=side, max_frames=5)
            span = Span(stage="visual")
            sigs = sample_frame_signatures("x.mp4", side=side, max_frames=5, span=span)
        self.assertEqual(len(base), 5)
        self.assertEqual([s.vec for s in base], [s.vec for s in sigs])  # identical
        cap = _cap(span, "frame_max")
        self.assertIsNotNone(cap)
        self.assertEqual(cap["limit"], 5)
        self.assertGreater(cap["dropped"], 0)  # 64 available - 5 kept


class RetrieveCapTest(unittest.TestCase):
    def setUp(self):
        from memovox.augur.retrieve import retrieve
        self.retrieve = retrieve
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.store.upsert_video(Video(video_id="vid:a", source_url="https://x/a",
                                       title="a", content_hash="a"))
        for i in range(40):
            mid = f"vid:a#m{i:04d}"
            text = f"machine learning topic number {i} about models and training"
            self.store.add_moment(Moment(mid, "vid:a", float(i), float(i) + 5, text,
                                         "spk_0", index=i), self.emb.embed_one(text))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_pool_and_top_k_counters_with_identical_order(self):
        settings = Settings()
        base = self.retrieve(self.store, "models and training", embedder=self.emb,
                             settings=settings)
        span = Span(stage="retrieve")
        got = self.retrieve(self.store, "models and training", embedder=self.emb,
                            settings=settings, span=span)
        self.assertEqual([mid for mid, _ in base], [mid for mid, _ in got])  # order intact
        self.assertIn("pool", span.counters)
        cap = _cap(span, "top_k")
        self.assertIsNotNone(cap)
        self.assertEqual(cap["limit"], settings.top_k)
        # 40 moments => fused candidates exceed top_k, so the cap genuinely fires.
        # Asserting dropped>0 proves the truncation actually happened (a broken
        # rrf_fuse that returned everything would record dropped==0 and fail here).
        self.assertGreater(cap["dropped"], 0)


if __name__ == "__main__":
    unittest.main()
