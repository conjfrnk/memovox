import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.base import Segment
from memovox.escapement import build_moments, moment_visual_embedding
from memovox.tessera import VisualEvent

VID = "yt:test"


def seg(start, end, text, speaker=None, kind="speech"):
    return Segment(start=start, end=end, text=text, speaker=speaker, kind=kind)


def vevent(t_start, t_end, ocr=None, caption=None, embedding=None):
    return VisualEvent(
        t_start_s=t_start, t_end_s=t_end, ocr_text=ocr, caption=caption,
        embedding=embedding or [0.0],
    )


class TestFusion(unittest.TestCase):
    def test_speaker_change_splits(self):
        segs = [
            seg(0, 6, "alpha one two three", "spk_a"),
            seg(6, 12, "alpha four five six", "spk_a"),
            seg(12, 20, "beta takes over now here", "spk_b"),
        ]
        moments = build_moments(VID, segs)
        self.assertEqual(len(moments), 2)
        self.assertEqual(moments[0].speaker_id, "spk_a")
        self.assertEqual(moments[1].speaker_id, "spk_b")
        self.assertEqual(moments[0].index, 0)
        self.assertEqual(moments[1].index, 1)

    def test_gap_splits(self):
        segs = [
            seg(0, 9, "talking about topic one here", "spk_a"),
            seg(20, 29, "after a long pause a new topic", "spk_a"),
        ]
        moments = build_moments(VID, segs)
        self.assertEqual(len(moments), 2)

    def test_continuous_single_moment(self):
        segs = [
            seg(0, 5, "one continuous", "spk_a"),
            seg(5, 10, "stream of speech", "spk_a"),
            seg(10, 15, "without any breaks", "spk_a"),
        ]
        moments = build_moments(VID, segs)
        self.assertEqual(len(moments), 1)
        self.assertIn("continuous", moments[0].transcript)
        self.assertAlmostEqual(moments[0].t_start_s, 0.0)
        self.assertAlmostEqual(moments[0].t_end_s, 15.0)

    def test_max_duration_splits(self):
        segs = [seg(i * 30, (i + 1) * 30, f"block {i}", "spk_a") for i in range(4)]
        moments = build_moments(VID, segs)
        self.assertEqual(len(moments), 2)

    def test_small_tail_merges_same_speaker(self):
        segs = [
            seg(0, 9, "first topic block here", "spk_a"),
            seg(20, 23, "oops", "spk_a"),  # gap boundary, but tiny -> merged back
        ]
        moments = build_moments(VID, segs)
        self.assertEqual(len(moments), 1)
        self.assertIn("oops", moments[0].transcript)

    def test_moment_ids_unique_and_ordered(self):
        segs = [
            seg(0, 9, "aaa", "spk_a"),
            seg(9, 18, "bbb", "spk_b"),
            seg(18, 27, "ccc", "spk_a"),
        ]
        moments = build_moments(VID, segs)
        ids = [m.moment_id for m in moments]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))


class TestVisualFusion(unittest.TestCase):
    def _one_moment_segs(self):
        # A single continuous spoken Moment spanning 0–15s.
        return [
            seg(0, 5, "one continuous", "spk_a"),
            seg(5, 10, "stream of speech", "spk_a"),
            seg(10, 15, "without any breaks", "spk_a"),
        ]

    def test_without_visual_events_modality_is_speech_only(self):
        moments = build_moments(VID, self._one_moment_segs())
        self.assertEqual(len(moments), 1)
        self.assertIsNone(moments[0].ocr_text)
        self.assertIsNone(moments[0].visual_caption)
        self.assertEqual(moments[0].modality, "speech")

    def test_overlapping_visual_events_fuse_into_moment(self):
        events = [
            vevent(2, 4, ocr="Slide one: Backpropagation", caption="title slide"),
            vevent(10, 12, ocr="Slide two: chain rule", caption="diagram"),
        ]
        moments = build_moments(VID, self._one_moment_segs(), visual_events=events)
        self.assertEqual(len(moments), 1)
        m = moments[0]
        self.assertIn("Backpropagation", m.ocr_text)
        self.assertIn("chain rule", m.ocr_text)
        self.assertIn("diagram", m.visual_caption)
        self.assertEqual(m.modality, "speech+slide")
        # On-screen text must flow into the embedded/searchable text.
        self.assertIn("Backpropagation", m.text_for_embedding())

    def test_nonoverlapping_event_does_not_bind(self):
        events = [vevent(100, 102, ocr="unrelated later slide")]
        moments = build_moments(VID, self._one_moment_segs(), visual_events=events)
        self.assertIsNone(moments[0].ocr_text)
        self.assertEqual(moments[0].modality, "speech")

    def test_moment_visual_embedding_is_mean_of_overlapping(self):
        events = [
            vevent(2, 4, embedding=[0.0, 1.0]),
            vevent(10, 12, embedding=[1.0, 0.0]),
            vevent(100, 102, embedding=[9.0, 9.0]),  # outside the moment
        ]
        moments = build_moments(VID, self._one_moment_segs(), visual_events=events)
        emb = moment_visual_embedding(moments[0], events)
        self.assertEqual(emb, [0.5, 0.5])

    def test_moment_visual_embedding_none_when_no_overlap(self):
        events = [vevent(100, 102, embedding=[1.0, 1.0])]
        moments = build_moments(VID, self._one_moment_segs(), visual_events=events)
        self.assertIsNone(moment_visual_embedding(moments[0], events))

    def test_moment_visual_embedding_tolerates_a_mismatched_leading_vector(self):
        # A single anomalous-dimension embedding must not discard all the others.
        events = [
            vevent(2, 4, embedding=[9.0]),        # wrong dim, leading
            vevent(6, 8, embedding=[0.0, 1.0]),
            vevent(10, 12, embedding=[1.0, 0.0]),
        ]
        moments = build_moments(VID, self._one_moment_segs(), visual_events=events)
        self.assertEqual(moment_visual_embedding(moments[0], events), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
