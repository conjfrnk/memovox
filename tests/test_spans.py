import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.assay.spans import locate_span


class TestLocateSpan(unittest.TestCase):
    def test_locate_span_returns_segment_window_for_a_sentence(self):
        segments = [
            (0.0, 5.0, "Neural nets learn by backprop."),
            (5.0, 12.0, "The chain rule is central."),
        ]
        self.assertEqual(
            locate_span("The chain rule is central.", segments), (5.0, 12.0)
        )

    def test_locate_span_falls_back_to_full_when_unmatched(self):
        self.assertEqual(
            locate_span("unrelated", [(0.0, 9.0, "alpha beta")], default=(0.0, 9.0)),
            (0.0, 9.0),
        )

    def test_locate_span_empty_segments_returns_default(self):
        self.assertEqual(
            locate_span("anything at all", [], default=(1.0, 2.0)), (1.0, 2.0)
        )

    def test_locate_span_empty_sentence_returns_default(self):
        self.assertEqual(
            locate_span("", [(0.0, 5.0, "alpha beta")], default=(0.0, 5.0)), (0.0, 5.0)
        )

    def test_locate_span_below_floor_returns_default(self):
        # One of three claim tokens overlaps (1/3 < 0.5) -> below the floor.
        segments = [(0.0, 5.0, "alpha gamma delta")]
        self.assertEqual(
            locate_span("alpha beta epsilon", segments, default=(0.0, 5.0)), (0.0, 5.0)
        )

    def test_locate_span_default_none_when_no_default(self):
        self.assertIsNone(locate_span("unrelated", [(0.0, 9.0, "alpha beta")]))

    def test_locate_span_caps_overlap_against_repeating_segment(self):
        # A longer segment that REPEATS a SUBSET of the sentence's tokens must
        # not beat the segment that actually contains the whole sentence. Under
        # the old uncapped metric the repeating segment scores 6/4 = 1.5 and
        # wins; under the capped set-intersection it covers only 2/4 = 0.5 and
        # loses to the exact segment's 4/4 = 1.0. Fails on old, passes on new.
        segments = [
            (0.0, 5.0, "the chain the chain the chain"),
            (5.0, 12.0, "the chain rule here"),
        ]
        self.assertEqual(locate_span("the chain rule here", segments), (5.0, 12.0))

    def test_locate_span_boundary_crossing_sentence_falls_back(self):
        # A 5-token sentence split across two segments: each covers only 2/5 =
        # 0.4 < 0.5 on its own, so neither clears the floor -> conservative
        # fallback to default (the whole-Moment span).
        segments = [
            (0.0, 5.0, "alpha beta filler one"),
            (5.0, 10.0, "gamma delta filler two"),
        ]
        self.assertEqual(
            locate_span("alpha beta gamma delta epsilon", segments, default=(0.0, 10.0)),
            (0.0, 10.0),
        )


if __name__ == "__main__":
    unittest.main()
