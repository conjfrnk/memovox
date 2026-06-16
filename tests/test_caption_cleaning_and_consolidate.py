"""Richer-corpus stress-hardening regressions (iter11):

  * caption cleaning — sentence fragments must not become bogus speaker labels,
    and HTML entities (``&nbsp;``/``&#39;``) must be decoded out of claim text;
  * topic-scoped contradiction search must filter by topic BEFORE the max_claims
    cap, so a relevant pair that arrives late in ingest order still gets compared.
"""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.base import Segment
from memovox.backends.nli import LexicalNLI
from memovox.config import Config
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.consolidate import find_contradictions
from memovox.loom.models import Claim
from memovox.stentor.transcript import (
    _looks_like_speaker,
    clean_segments,
    clean_text,
)


class TestSpeakerLabelHeuristic(unittest.TestCase):
    """A leading ``Name:`` becomes a speaker only when it looks like a name, not a
    sentence fragment that happens to carry a colon (the richer-corpus false
    positives: 'But caveat', 'For example', 'And that worked great', ...)."""

    def test_real_labels_accepted(self):
        for name in ["BRADY HARAN", "MATT PARKER", "Rob Wiblin", "Molaison",
                     "NEWSCASTER", "Yanjaa", "YANJAA", "Dr. Lee", "McDonald",
                     "Tony Padilla", "REAGAN"]:
            self.assertTrue(_looks_like_speaker(name), name)

    def test_sentence_fragments_rejected(self):
        for frag in ["But caveat", "For example", "And that worked great",
                     "I have three world records", "So", "Well", "Now",
                     "What I mean is", "this is the part"]:
            self.assertFalse(_looks_like_speaker(frag), frag)

    def test_fragment_prefix_not_hoisted_to_speaker(self):
        # "But caveat: ..." is content, not a speaker turn -> speaker stays unset
        # and the text is preserved (not stripped as a label).
        segs = [Segment(start=0.0, end=2.0, text="But caveat: the index can mislead.")]
        out = [s for s in clean_segments(segs) if s.kind == "speech"]
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0].speaker)
        self.assertIn("caveat", out[0].text)

    def test_real_label_still_extracted(self):
        segs = [Segment(start=0.0, end=2.0, text="BRADY HARAN: so what is it?")]
        out = [s for s in clean_segments(segs) if s.kind == "speech"]
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].speaker, "BRADY HARAN")
        self.assertNotIn("BRADY HARAN", out[0].text)


class TestHtmlEntityDecoding(unittest.TestCase):
    """WebVTT escapes ``&`` ``<`` ``>`` and emits ``&nbsp;``; raw, these leak into
    claim text ("By 1920,&nbsp;&nbsp;"). clean_text must decode + collapse them."""

    def test_nbsp_decoded_and_collapsed(self):
        text, _ = clean_text("By 1920,&nbsp;&nbsp; the banks failed")
        self.assertNotIn("&nbsp;", text)
        self.assertNotIn("\xa0", text)
        self.assertEqual(text, "By 1920, the banks failed")

    def test_named_and_numeric_entities(self):
        text, _ = clean_text("Tom &amp; Jerry said it&#39;s fine")
        self.assertEqual(text, "Tom & Jerry said it's fine")

    def test_decoded_lt_not_reparsed_as_tag(self):
        # A genuine "<i>" tag is stripped; an escaped "&lt;3" decodes to literal text.
        text, _ = clean_text("<i>love</i> is &lt;3 always")
        self.assertEqual(text, "love is <3 always")


class TestTopicContradictionCapOrdering(unittest.TestCase):
    """find_contradictions(topic=...) must filter to the topic BEFORE truncating to
    max_claims; otherwise a relevant pair that lands past the cap in ingest order is
    silently never compared (looks like an NLI miss but is a cap artifact)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.store = LoomStore(self.config)
        for v in ("yt:a", "yt:b"):
            self.store.upsert_video(Video(v, f"https://youtu.be/{v[3:]}", "talk"))
        # 40 filler claims (off-topic), THEN the on-topic contradiction pair last —
        # so the relevant claims sit at rowids past a small cap.
        def add(cid, mid, vid, text, t):
            self.store.add_moment(Moment(mid, vid, float(t), float(t) + 1.0, text, index=0))
            self.store.add_claim(Claim(cid, mid, vid, text,
                                       t_start_s=float(t), t_end_s=float(t) + 1.0))

        for i in range(40):
            add(f"yt:a#m{i:04d}.c0", f"yt:a#m{i:04d}", "yt:a",
                f"a generic remark number {i} about assorted weather and travel", i)
        add("yt:a#mz.c0", "yt:a#mz", "yt:a",
            "saturated fat causes heart disease", 100)
        add("yt:b#mz.c0", "yt:b#mz", "yt:b",
            "saturated fat does not cause heart disease", 5)
        self.nli = LexicalNLI()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_late_topic_pair_is_compared_under_small_cap(self):
        # cap below the rowid of the on-topic pair: cap-then-filter would yield zero
        # candidates; filter-then-cap keeps the two saturated-fat claims.
        pairs = find_contradictions(self.store, nli=self.nli, topic="saturated fat",
                                    max_claims=10, write_edges=False)
        vids = {frozenset((p.claim_a.video_id, p.claim_b.video_id)) for p in pairs}
        self.assertIn(frozenset(("yt:a", "yt:b")), vids,
                      "late-arriving on-topic contradiction pair was not compared")

    def test_no_topic_path_unchanged_by_cap(self):
        # Without a topic the cap still bounds the first max_claims (cost control).
        pairs = find_contradictions(self.store, nli=self.nli, topic=None,
                                    max_claims=10, write_edges=False)
        self.assertIsInstance(pairs, list)


if __name__ == "__main__":
    unittest.main()
