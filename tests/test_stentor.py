import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import stentor
from memovox.config import Config
from memovox.stentor import transcript

VTT = """WEBVTT

00:00:00.000 --> 00:00:04.000
<v Alice>Um, welcome to the talk on scaling laws.

00:00:04.000 --> 00:00:06.000
[Music]

00:00:06.000 --> 00:00:10.000
Bob: The recommended chunk size is 512 tokens.
"""


class TestTranscriptParsing(unittest.TestCase):
    def test_parse_vtt_times_and_speaker(self):
        segs = transcript.parse_vtt(VTT)
        self.assertEqual(len(segs), 3)
        self.assertAlmostEqual(segs[0].start, 0.0)
        self.assertAlmostEqual(segs[0].end, 4.0)
        self.assertEqual(segs[0].speaker, "Alice")

    def test_clean_strips_fillers_and_extracts_events(self):
        segs = transcript.clean_segments(transcript.parse_vtt(VTT))
        speech = [s for s in segs if s.kind == "speech"]
        events = [s for s in segs if s.kind == "event"]
        self.assertTrue(any("welcome to the talk" in s.text for s in speech))
        self.assertFalse(any(s.text.lower().startswith("um") for s in speech))
        self.assertEqual(len(events), 1)
        self.assertIn("music", events[0].text.lower())

    def test_speaker_prefix_detection(self):
        segs = transcript.clean_segments(transcript.parse_vtt(VTT))
        bob = [s for s in segs if "chunk size" in s.text]
        self.assertEqual(bob[0].speaker, "Bob")
        self.assertNotIn("Bob:", bob[0].text)

    def test_parse_json(self):
        segs = transcript.parse_json(
            {"segments": [{"start": 1, "end": 2, "text": "hello", "speaker": "x"}]}
        )
        self.assertEqual(segs[0].text, "hello")
        self.assertEqual(segs[0].speaker, "x")

    def test_parse_plain_orders_sentences(self):
        segs = transcript.parse_plain("First idea. Second idea. Third.", duration=30)
        self.assertEqual(len(segs), 3)
        self.assertLess(segs[0].start, segs[1].start)


# YouTube auto-generated "rolling" captions: each visible window repeats the
# previous line as a carried-over scroll-up line plus a new inline-<c>-timed
# line, and tiny 10ms cues hold a pure carry-over. Naively joining every line of
# every cue duplicates ~half the transcript. See parse_cues' rolling-format path.
# Faithful to the real format: a whitespace-only line (" ") is an empty caption
# row *inside* a cue; cues are separated by a truly-blank line.
ROLLING_VTT = "\n".join(
    [
        "WEBVTT",
        "Kind: captions",
        "Language: en",
        "",
        "00:00:01.040 --> 00:00:02.310 align:start position:0%",
        " ",
        "Want<00:00:01.199><c> to</c><00:00:01.360><c> see</c><00:00:01.600><c> the</c>",
        "",
        "00:00:02.310 --> 00:00:02.320 align:start position:0%",
        "Want to see the",
        " ",
        "",
        "00:00:02.320 --> 00:00:05.000 align:start position:0%",
        "Want to see the",
        "coolest<00:00:02.399><c> thing</c><00:00:03.120><c> here.</c>",
        "",
        "00:00:05.000 --> 00:00:05.010 align:start position:0%",
        "coolest thing here.",
        " ",
        "",
        "00:00:05.010 --> 00:00:08.000 align:start position:0%",
        "coolest thing here.",
        "This<00:00:05.200><c> is</c><00:00:05.600><c> caviar.</c>",
        "",
    ]
)


class TestRollingCaptions(unittest.TestCase):
    def test_rolling_captions_are_deduped(self):
        # parse_cues returns raw cue text (inline tags stripped later by
        # clean_segments); cleaning here lets us assert on the dedup result.
        segs = transcript.clean_segments(transcript.parse_cues(ROLLING_VTT))
        texts = [s.text for s in segs if s.kind == "speech"]
        # Only the inline-timed "new" lines survive — pure carry-over cues drop.
        self.assertEqual(
            texts,
            [
                "Want to see the",
                "coolest thing here.",
                "This is caviar.",
            ],
        )
        # The carried-over phrase must appear exactly once across the whole transcript.
        joined = " ".join(texts)
        self.assertEqual(joined.count("Want to see the"), 1)
        self.assertEqual(joined.count("coolest thing here."), 1)

    def test_rolling_inline_timestamps_are_stripped_on_clean(self):
        clean = transcript.clean_segments(transcript.parse_cues(ROLLING_VTT))
        speech = [s for s in clean if s.kind == "speech"]
        self.assertEqual(len(speech), 3)
        self.assertNotIn("<", " ".join(s.text for s in speech))

    def test_turn_markers_yield_multiple_speakers(self):
        # The CEA-608/broadcast convention: ">>" marks a speaker change. Captions
        # that use it must surface as multiple (anonymous) speakers on the free path
        # even without names, instead of collapsing everyone onto spk_0.
        vtt = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:03.000\n>> Want to see the seat?\n\n"
            "00:00:03.000 --> 00:00:06.000\n>> Yes, it is incredible here.\n\n"
            "00:00:06.000 --> 00:00:09.000\nIt reclines fully flat now.\n"
        )
        from memovox.stentor.diarize import assign_speakers
        segs = assign_speakers(transcript.clean_segments(transcript.parse_cues(vtt)))
        speakers = [s.speaker for s in segs if s.kind == "speech"]
        # two turn markers -> at least two distinct speakers; ">>" stripped from text
        self.assertGreaterEqual(len(set(speakers)), 2)
        self.assertFalse(any(">>" in s.text for s in segs))
        # the second turn persists onto the following unmarked line (same speaker)
        self.assertEqual(speakers[1], speakers[2])

    def test_musical_note_lines_are_events_not_speech(self):
        # Captions mark sung lyrics with musical notes; that is music, not spoken
        # video content, so it must become a timeline event, never a claim.
        vtt = (
            "WEBVTT\nKind: captions\n\n"
            "00:00:00.000 --> 00:00:05.000\n"
            "♪ This is the heavy heavy monster sound ♪\n\n"
            "00:00:05.000 --> 00:00:09.000\nThe TV here is enormous.\n"
        )
        clean = transcript.clean_segments(transcript.parse_cues(vtt))
        speech = [s for s in clean if s.kind == "speech"]
        events = [s for s in clean if s.kind == "event"]
        self.assertFalse(any("monster sound" in s.text for s in speech))
        self.assertTrue(any("music" in s.text.lower() for s in events))
        # genuine speech is untouched
        self.assertTrue(any("TV here is enormous" in s.text for s in speech))

    def test_non_rolling_vtt_keeps_all_lines(self):
        # A normal (non-rolling) two-line cue is NOT touched by the rolling path:
        # both lines are joined, since neither carries inline <c> timing.
        plain = (
            "WEBVTT\n\n00:00:00.000 --> 00:00:04.000\n"
            "first line\nsecond line\n"
        )
        segs = transcript.parse_cues(plain)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].text, "first line second line")


class TestStentorRun(unittest.TestCase):
    def test_run_on_transcript_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            vtt = tmp / "talk.en.vtt"
            vtt.write_text(VTT, encoding="utf-8")
            config = Config(store=tmp / "store").ensure()

            result = stentor.run(
                config, str(vtt), source_url="https://youtu.be/abc123",
            )

            self.assertEqual(result.asr_backend, "captions")
            self.assertIsNotNone(result.meta.content_hash)
            speech = [s for s in result.segments if s.kind == "speech"]
            self.assertEqual(len(speech), 2)
            self.assertEqual({s.speaker for s in speech}, {"Alice", "Bob"})
            self.assertIn("Alice", result.speaker_names)


if __name__ == "__main__":
    unittest.main()
