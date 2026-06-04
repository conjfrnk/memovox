import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends import get_ocr, get_vlm
from memovox.backends.base import OCRBackend, VLMBackend
from memovox.config import Settings
from memovox.tessera.classify import classify_frame


class TestFrameTypeClassifier(unittest.TestCase):
    def test_dense_ocr_text_is_slide_or_document(self):
        self.assertIn(classify_frame([0.5] * 256, "word " * 10), ("slide", "document"))
        self.assertEqual(classify_frame([0.5] * 256, "word " * 40), "document")

    def test_high_variance_no_text_is_diagram(self):
        diagram = [0.0, 1.0] * 128  # alternating -> high variance, structure
        self.assertEqual(classify_frame(diagram, ""), "diagram")

    def test_low_variance_no_text_is_talking_head(self):
        self.assertEqual(classify_frame([0.5] * 256, ""), "talking_head")
from memovox.tessera import VisualEvent, VisualResult, run
from memovox.tessera.frames import FrameSig, bytes_to_signature
from memovox.tessera.keyframes import select_keyframes
from memovox.tessera.scenes import frame_distance, segment_scenes


def const_sig(t, level):
    """A FrameSig for a flat 16x16 frame at brightness ``level`` (0-255)."""
    return FrameSig(t=t, vec=bytes_to_signature(bytes([level] * 256)))


class TestSignature(unittest.TestCase):
    def test_bytes_to_signature_scales_to_unit_interval(self):
        sig = bytes_to_signature(bytes([0, 255, 51]))
        self.assertEqual(len(sig), 3)
        self.assertAlmostEqual(sig[0], 0.0)
        self.assertAlmostEqual(sig[1], 1.0)
        self.assertAlmostEqual(sig[2], 0.2, places=2)

    def test_identical_frames_have_zero_distance(self):
        a = bytes_to_signature(bytes([100] * 16))
        self.assertEqual(frame_distance(a, a), 0.0)

    def test_distance_captures_brightness_change(self):
        # A flat dark frame vs a flat bright frame must register a large
        # difference (cosine would call them identical; content-diff must not).
        dark = bytes_to_signature(bytes([20] * 16))
        light = bytes_to_signature(bytes([220] * 16))
        self.assertGreater(frame_distance(dark, light), 0.5)


class TestScenes(unittest.TestCase):
    def test_uniform_frames_are_one_scene(self):
        sigs = [const_sig(float(i), 30) for i in range(5)]
        scenes = segment_scenes(sigs, threshold=0.3)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].start_idx, 0)
        self.assertEqual(scenes[0].end_idx, 4)
        self.assertAlmostEqual(scenes[0].t_start, 0.0)
        self.assertAlmostEqual(scenes[0].t_end, 4.0)

    def test_hard_cut_splits_into_two_scenes(self):
        sigs = [const_sig(0, 20), const_sig(1, 20), const_sig(2, 220), const_sig(3, 220)]
        scenes = segment_scenes(sigs, threshold=0.3)
        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[0].end_idx, 1)
        self.assertEqual(scenes[1].start_idx, 2)
        self.assertEqual([s.index for s in scenes], [0, 1])

    def test_empty_input_yields_no_scenes(self):
        self.assertEqual(segment_scenes([], threshold=0.3), [])


class TestKeyframes(unittest.TestCase):
    def test_static_scene_collapses_to_one_keyframe(self):
        sigs = [const_sig(float(i), 30) for i in range(6)]
        scenes = segment_scenes(sigs, threshold=0.3)
        kept = select_keyframes(sigs, scenes, min_gain=0.12, per_scene_cap=8)
        self.assertEqual(kept, [0])

    def test_high_information_scene_keeps_multiple_keyframes(self):
        # Brightness steps of ~0.27 each (>min_gain) => each frame is novel.
        sigs = [const_sig(float(i), 20 + i * 70) for i in range(4)]
        scenes = segment_scenes(sigs, threshold=2.0)  # threshold high => single scene
        self.assertEqual(len(scenes), 1)
        kept = select_keyframes(sigs, scenes, min_gain=0.12, per_scene_cap=8)
        self.assertEqual(kept, [0, 1, 2, 3])

    def test_per_scene_cap_bounds_keyframes(self):
        sigs = [const_sig(float(i), (i * 90) % 256) for i in range(10)]
        scenes = segment_scenes(sigs, threshold=2.0)  # single scene
        kept = select_keyframes(sigs, scenes, min_gain=0.0, per_scene_cap=3)
        self.assertLessEqual(len(kept), 3)
        self.assertEqual(kept[0], 0)

    def test_first_frame_of_every_scene_is_kept(self):
        sigs = [const_sig(0, 20), const_sig(1, 20), const_sig(2, 220), const_sig(3, 220)]
        scenes = segment_scenes(sigs, threshold=0.3)
        kept = select_keyframes(sigs, scenes, min_gain=0.12, per_scene_cap=8)
        self.assertIn(0, kept)
        self.assertIn(2, kept)


class _FixedVLM(VLMBackend):
    name = "fixed"

    def caption(self, image_path, *, ocr_text=None, prompt=None):
        return "a slide titled Backpropagation"


class _FixedOCR(OCRBackend):
    name = "fixed"

    def extract(self, image_path):
        return "Backpropagation chain rule"


class TestVisualBackends(unittest.TestCase):
    def test_null_vlm_and_ocr_are_always_available_and_empty(self):
        vlm = get_vlm("none")
        ocr = get_ocr("none")
        self.assertEqual(vlm.caption(None), "")
        self.assertEqual(ocr.extract(None), "")

    def test_auto_resolves_to_a_usable_backend(self):
        # Whatever is installed, auto must yield a working backend (free fallback).
        self.assertTrue(hasattr(get_vlm("auto"), "caption"))
        self.assertTrue(hasattr(get_ocr("auto"), "extract"))


class TestRun(unittest.TestCase):
    def _frames(self):
        # 5 dark frames, then a hard cut to 3 bright frames.
        dark = [const_sig(float(i), 30) for i in range(5)]
        bright = [const_sig(float(i), 220) for i in range(5, 8)]
        return dark + bright

    def test_no_video_is_unavailable(self):
        class Meta:
            media_path = None
            is_video = False

        result = run(None, Meta(), settings=Settings())
        self.assertIsInstance(result, VisualResult)
        self.assertFalse(result.available)
        self.assertEqual(result.events, [])

    def test_injected_frames_produce_one_event_per_keyframe(self):
        result = run(None, None, settings=Settings(), frames=self._frames(),
                     vlm=get_vlm("none"), ocr=get_ocr("none"))
        self.assertTrue(result.available)
        # Two scenes (dark | bright), each static => one keyframe each.
        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.n_scenes, 2)
        self.assertAlmostEqual(result.events[0].t_start_s, 0.0)
        self.assertAlmostEqual(result.events[1].t_start_s, 5.0)
        self.assertEqual([e.scene_index for e in result.events], [0, 1])
        for e in result.events:
            self.assertIsInstance(e, VisualEvent)
            self.assertEqual(len(e.embedding), 256)

    def test_scene_start_keyframe_has_zero_info_gain(self):
        # info_gain is intra-scene novelty; the first keyframe of every scene
        # (incl. scenes after a cut) has no prior in-scene frame => gain 0.0,
        # not the large cross-scene cut distance.
        result = run(None, None, settings=Settings(), frames=self._frames(),
                     vlm=get_vlm("none"), ocr=get_ocr("none"))
        self.assertEqual([e.scene_index for e in result.events], [0, 1])
        for e in result.events:
            self.assertEqual(e.info_gain, 0.0)

    def test_events_carry_caption_and_ocr_from_backends(self):
        result = run(None, None, settings=Settings(), frames=self._frames(),
                     vlm=_FixedVLM(), ocr=_FixedOCR())
        self.assertTrue(result.events)
        self.assertEqual(result.events[0].ocr_text, "Backpropagation chain rule")
        self.assertIn("Backpropagation", result.events[0].caption)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class TestRealVideoSampling(unittest.TestCase):
    """End-to-end coverage of the ffmpeg frame-sampling adapter on a real video."""

    def _make_video(self, path):
        # 3s black then 3s white => one hard cut => two scenes.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3:r=5",
            "-f", "lavfi", "-i", "color=c=white:s=64x64:d=3:r=5",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1",
            "-pix_fmt", "yuv420p", str(path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def test_sampling_detects_scene_cut_and_keyframes(self):
        from memovox.config import Config
        from memovox.tessera.frames import sample_frame_signatures

        with tempfile.TemporaryDirectory() as tmp:
            video = pathlib.Path(tmp) / "clip.mp4"
            self._make_video(video)

            sigs = sample_frame_signatures(video, fps=1.0, side=16, max_frames=600)
            self.assertGreaterEqual(len(sigs), 4)
            self.assertEqual(len(sigs[0].vec), 256)

            class Meta:
                media_path = video
                is_video = True
                title = "clip"

            config = Config(store=pathlib.Path(tmp) / "store").ensure()
            result = run(config, Meta(), settings=Settings(),
                         vlm=get_vlm("none"), ocr=get_ocr("none"))
            self.assertTrue(result.available)
            self.assertGreaterEqual(result.n_scenes, 2)
            self.assertGreaterEqual(len(result.events), 2)
            # keyframe images were extracted to the frames dir
            self.assertTrue(any(e.frame_ref for e in result.events))


if __name__ == "__main__":
    unittest.main()
