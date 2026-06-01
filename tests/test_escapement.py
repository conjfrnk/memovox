import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.base import Segment
from memovox.config import Settings
from memovox.escapement import build_moments

VID = "yt:test"


def seg(start, end, text, speaker=None, kind="speech"):
    return Segment(start=start, end=end, text=text, speaker=speaker, kind=kind)


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


if __name__ == "__main__":
    unittest.main()
