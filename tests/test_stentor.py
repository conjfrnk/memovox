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
