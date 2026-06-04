"""M0.3 W1 — per-word timings survive Segment -> SegmentRef fusion (spec §4.2)."""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import escapement
from memovox.backends.base import Segment, Word
from memovox.backends.embed import HashingEmbedder
from memovox.config import Settings


class FusionWordsTest(unittest.TestCase):
    def test_words_preserved_on_segment_ref(self):
        words = [Word("the", 0.0, 0.2), Word("chain", 0.2, 0.5),
                 Word("rule", 0.5, 0.9), Word("here", 0.9, 1.2)]
        segs = [Segment(start=0.0, end=1.2, text="the chain rule here", speaker="spk_0",
                        words=words)]
        moments = escapement.build_moments("vid", segs, embedder=HashingEmbedder(dim=64),
                                           settings=Settings())
        self.assertTrue(moments)
        ref = moments[0].segments[0]
        # positional 3-tuple unpack still works (the hot-path invariant)
        t0, t1, text = ref[0], ref[1], ref[2]
        self.assertEqual((t0, t1), (0.0, 1.2))
        # words round-trip as (start, end, word) tuples
        self.assertEqual(len(ref.words), 4)
        self.assertEqual(ref.words[1], (0.2, 0.5, "chain"))

    def test_free_path_segment_has_empty_words(self):
        segs = [Segment(start=0.0, end=5.0, text="no word timings here", speaker="spk_0")]
        moments = escapement.build_moments("vid", segs, embedder=HashingEmbedder(dim=64),
                                           settings=Settings())
        self.assertEqual(moments[0].segments[0].words, ())


if __name__ == "__main__":
    unittest.main()
