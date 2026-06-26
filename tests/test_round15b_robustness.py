"""Round-15b robustness: fixes from the second (fresh-dimensions) stress panel.

Covers cross-video speaker identity (non-Latin scripts), transcript rolling-caption
false-positives, generative-extract type clamping, timestamp bounds, and extract
ordering — surfaced by the panel over transcript/resolution/CLI/claims dimensions.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.assay.extract_output import _claim_sort_key, extract_document
from memovox.config import Config
from memovox.loom.models import Moment, Speaker
from memovox.loom.resolve import resolve_speakers
from memovox.loom.store import LoomStore
from memovox.stentor.transcript import _is_rolling, _parse_ts, parse_cues


class _FakeGenLLM:
    name = "fake-gen"
    is_generative = True

    def __init__(self, payload):
        self._payload = payload

    def complete(self, *a, **k):
        return self._payload


class NonLatinSpeakerResolutionTest(unittest.TestCase):
    """[0] non-Latin-script named speakers must NOT collapse into one canonical identity."""

    def test_distinct_non_latin_names_stay_distinct_same_unifies(self):
        tmp = tempfile.TemporaryDirectory()
        store = LoomStore(Config(store=pathlib.Path(tmp.name) / "s").ensure())
        people = [("vid:aaa", "习近平"), ("vid:bbb", "Владимир"),
                  ("vid:ccc", "محمد"), ("vid:ddd", "习近平")]  # ddd repeats the first name
        for vid, nm in people:
            store.upsert_speaker(Speaker(speaker_id=f"{vid}:{nm}", label=nm, resolved_name=nm))
        resolve_speakers(store)
        canon = [store.canonical_speaker(f"{v}:{n}") for v, n in people]
        store.close()
        tmp.cleanup()
        self.assertEqual(len(set(canon)), 3, "three distinct names must yield three identities")
        self.assertEqual(canon[0], canon[3], "the same non-Latin name must unify across videos")
        self.assertNotIn("spk:speaker", canon, "no collapse onto the degenerate slug")

    def test_latin_speaker_resolution_unchanged(self):
        tmp = tempfile.TemporaryDirectory()
        store = LoomStore(Config(store=pathlib.Path(tmp.name) / "s").ensure())
        for vid in ("vid:aaa", "vid:bbb"):
            store.upsert_speaker(Speaker(speaker_id=f"{vid}:Rob Wiblin", label="Rob Wiblin",
                                         resolved_name="Rob Wiblin"))
        resolve_speakers(store)
        c1 = store.canonical_speaker("vid:aaa:Rob Wiblin")
        c2 = store.canonical_speaker("vid:bbb:Rob Wiblin")
        store.close()
        tmp.cleanup()
        self.assertEqual(c1, c2, "same Latin name still unifies")
        self.assertEqual(c1, "spk:rob-wiblin", "Latin slug canonical id is byte-identical")


class RollingCaptionFalsePositiveTest(unittest.TestCase):
    """[2] a stray inline-timestamp token must not flip a plain VTT into lossy 'rolling' mode."""

    def test_note_block_inline_token_does_not_drop_cues(self):
        vtt = ("WEBVTT\n\nNOTE\nedited around <00:00:01.000> to fix sync\n\n"
               "00:00:01.000 --> 00:00:03.000\nFirst sentence here\n\n"
               "00:00:03.000 --> 00:00:06.000\nSecond sentence here\n")
        segs = parse_cues(vtt)
        self.assertEqual(len(segs), 2, "plain cues must survive a NOTE-block inline token")
        self.assertFalse(_is_rolling(vtt))

    def test_single_karaoke_cue_does_not_drop_plain_cues(self):
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nplain one\n\n"
               "00:00:03.000 --> 00:00:05.000\nka<00:00:03.500>raoke line\n\n"
               "00:00:05.000 --> 00:00:07.000\nplain two\n")
        self.assertEqual(len(parse_cues(vtt)), 3, "one karaoke cue must not delete plain cues")

    def test_genuine_rolling_still_deduplicated(self):
        roll = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhello\nhello <00:00:01.500>world\n\n"
                "00:00:02.000 --> 00:00:03.000\nhello world\nhello world <00:00:02.500>again\n")
        self.assertTrue(_is_rolling(roll), "real rolling captions are still detected")


class TimestampBoundTest(unittest.TestCase):
    """[4] an absurd-but-finite timestamp must clamp, not poison ordering / deep links."""

    def test_huge_finite_timestamp_clamped(self):
        self.assertEqual(_parse_ts("99999999999999999999999:00:00.000"), 0.0)
        self.assertEqual(_parse_ts("00:00:05.000"), 5.0)  # normal value unaffected


class GenerativeClaimTypeClampTest(unittest.TestCase):
    """[3] an off-enum LLM-supplied claim type must clamp, not crash the whole extraction."""

    def test_off_enum_type_does_not_crash_extract(self):
        m = Moment(moment_id="vid:aaa#m0000", video_id="vid:aaa", t_start_s=0.0, t_end_s=5.0,
                   transcript="The cat is a mammal here.", index=0)
        for bad in ("ASSERTION", "", "fact."):
            payload = ('[{"text":"The cat is a mammal here.","subject":"cat","predicate":"is",'
                       f'"object":"mammal","type":"{bad}"}}]')
            doc = extract_document(m, llm=_FakeGenLLM(payload))  # must not raise
            self.assertTrue(all(c["type"] in
                                {"FACT", "DEFINITION", "OPINION", "PROCEDURE", "EXAMPLE",
                                 "PREDICTION", "CORRECTION"} for c in doc["claims"]))


class ClaimOrderingTest(unittest.TestCase):
    """[5] extract ordering must follow extraction sequence, not lexicographic claim_id."""

    def test_hundred_plus_claims_order_numerically(self):
        class C:
            def __init__(self, i):
                self.claim_id = f"yt:x#m0000.c{i:02d}"

        ordered = sorted((C(i) for i in range(103)), key=_claim_sort_key)
        idxs = [c.claim_id.split(".c")[1] for c in ordered]
        self.assertEqual(idxs[:13], [f"{i:02d}" for i in range(13)],
                         "c100 must NOT sort between c10 and c11")
        self.assertEqual(idxs[-1], "102", "the last claim is c102, in sequence")


if __name__ == "__main__":
    unittest.main()
