"""M0.3 W3 — citation window ⊆ verified-premise window (provenance is sacred).

Word-window tightening (W2) narrows the DISPLAYED citation span, but the NLI
premise (span_text) must stay segment-granular — so the shown span can never drift
narrower than the text the verification gate actually checked.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.assay.spans import locate_span, span_text
from memovox.loom.models import SegmentRef


class PremiseInvariantTest(unittest.TestCase):
    def _seg(self):
        return SegmentRef(0.0, 10.0, "the chain rule is central here", words=(
            (0.0, 0.2, "the"), (0.2, 0.5, "chain"), (0.5, 0.9, "rule"),
            (0.9, 1.4, "is"), (1.4, 2.0, "central"), (2.0, 2.4, "here")))

    def test_citation_window_narrows_but_premise_stays_segment_granular(self):
        seg = self._seg()
        # the displayed citation span is the matched word window (narrow)
        t0, t1 = locate_span("chain rule", [seg])
        self.assertLess(t1 - t0, 10.0)               # genuinely narrower than the cue
        self.assertEqual((t0, t1), (0.2, 0.9))
        # the NLI premise for that narrowed window is the WHOLE cue text
        premise = span_text([seg], t0, t1)
        self.assertEqual(premise, "the chain rule is central here")
        # invariant: the citation window is contained within the verified premise.
        # The premise (whole cue, 0.0..10.0) ⊇ the citation window (0.2..0.9).
        self.assertLessEqual(0.0, t0)
        self.assertGreaterEqual(10.0, t1)

    def test_no_words_premise_equals_cue(self):
        seg = SegmentRef(0.0, 10.0, "the chain rule is central here")
        t0, t1 = locate_span("chain rule", [seg])
        self.assertEqual((t0, t1), (0.0, 10.0))      # identity (no words)
        self.assertEqual(span_text([seg], t0, t1), "the chain rule is central here")


if __name__ == "__main__":
    unittest.main()
