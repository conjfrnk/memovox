"""Scoring logic for the real-corpus benchmark (eval/benchmark.py).

The orchestration (ingest real video + ask) needs a connected machine + ffmpeg /
tesseract, so it is exercised via an injected fake engine here; the math that turns
responses into the two headline numbers (shown-only lift, refusal vs confabulation)
is pure and fully tested.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval.benchmark import (
    ABSENT,
    AUDIO_ONLY,
    CONFABULATED,
    CORRECT,
    PRESENT,
    REFUSED,
    WITH_VIDEO,
    WRONG,
    classify,
    run_benchmark,
    summarize,
)


class TestClassify(unittest.TestCase):
    def test_present_hit_is_correct(self):
        self.assertEqual(classify(False, "It was 9001 tps.", PRESENT, ["9001"]), CORRECT)

    def test_present_miss_is_wrong(self):
        self.assertEqual(classify(False, "Something unrelated.", PRESENT, ["9001"]), WRONG)

    def test_present_refused_is_refused(self):
        self.assertEqual(classify(True, "I don't have enough evidence.", PRESENT, ["9001"]), REFUSED)

    def test_absent_refused_is_correct_refusal(self):
        self.assertEqual(classify(True, "I can't find that.", ABSENT, []), REFUSED)

    def test_absent_answered_is_confabulation(self):
        self.assertEqual(classify(False, "The fine-tuning length is 4096.", ABSENT, []), CONFABULATED)


def _fake_engine():
    """Simulates a corpus where one answer is ONLY on screen (shown-only): the
    audio-only condition can't surface it, the with-video condition can; a speech
    fact is answerable in both; and an out-of-corpus question is refused in both."""
    def ingest(condition, video):
        return None

    def ask(condition, query):
        if "absent" in query:
            return {"low_evidence": True, "text": "I don't have enough indexed evidence.",
                    "citations": []}
        if "chart" in query:  # shown-only: only the visual track sees the slide
            if condition == WITH_VIDEO:
                return {"low_evidence": False, "text": "Throughput peaked at 9001 tps.",
                        "citations": [{"modality": "speech+slide", "ocr_unverified": True}]}
            return {"low_evidence": True, "text": "I don't have enough indexed evidence.",
                    "citations": []}
        # speech-only: answerable in both conditions
        return {"low_evidence": False, "text": "The chunk size is 512 tokens.",
                "citations": [{"modality": "speech", "ocr_unverified": False}]}

    return ingest, ask


class TestRunBenchmark(unittest.TestCase):
    QA = [
        {"q": "what chart value peaked?", "expects": PRESENT, "modality": "shown-only",
         "answer_substrings": ["9001"]},
        {"q": "what spoken chunk size?", "expects": PRESENT, "modality": "speech-only",
         "answer_substrings": ["512"]},
        {"q": "what is the absent fine-tuning length?", "expects": ABSENT, "modality": "none",
         "answer_substrings": []},
    ]

    def test_shown_only_lift_and_refusal(self):
        report = run_benchmark([], self.QA, work_dir=None, engine=_fake_engine())
        s = report["summary"]
        # Visual track recovers the shown-only answer; audio-only cannot.
        self.assertEqual(s["present_accuracy"][WITH_VIDEO]["shown-only"], 1.0)
        self.assertEqual(s["present_accuracy"][AUDIO_ONLY]["shown-only"], 0.0)
        self.assertEqual(s["shown_only_lift"], 1.0)
        # Speech facts are answerable regardless of the visual track.
        self.assertEqual(s["present_accuracy"][AUDIO_ONLY]["speech-only"], 1.0)
        self.assertEqual(s["present_accuracy"][WITH_VIDEO]["speech-only"], 1.0)
        # Out-of-corpus question is refused, not confabulated.
        self.assertEqual(s["refusal"][WITH_VIDEO]["correct_refusal_rate"], 1.0)
        self.assertEqual(s["refusal"][WITH_VIDEO]["confabulation_rate"], 0.0)
        # The recovered shown-only answer is honestly flagged unverified.
        self.assertEqual(s["shown_only_ocr_flag_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
