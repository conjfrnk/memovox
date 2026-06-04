"""M0.3 W6 — ffprobe ASR-readiness pre-check (§4.1) + captions-as-prior lever (§9)."""

from __future__ import annotations

import pathlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.errors import DemuxError
from memovox.stentor import asr as asr_mod
from memovox.stentor.acquire import SourceMeta
from memovox.stentor.asr import resolve_asr_backend


def _meta(*, media=None, captions=None):
    return SourceMeta(source_url="https://x/v", title="v",
                      media_path=Path(media) if media else None,
                      captions_path=Path(captions) if captions else None)


class AsrReadinessTest(unittest.TestCase):
    def test_no_audio_stream_raises_demuxerror_before_load(self):
        meta = _meta(media="clip.mp4")  # media present, no captions
        with mock.patch.object(asr_mod.WhisperASR, "is_available", return_value=True), \
             mock.patch.object(asr_mod.audio, "has_audio_stream", return_value=False):
            with self.assertRaises(DemuxError):
                resolve_asr_backend(meta, "auto")

    def test_audio_present_resolves_whisper(self):
        meta = _meta(media="clip.mp4")
        with mock.patch.object(asr_mod.WhisperASR, "is_available", return_value=True), \
             mock.patch.object(asr_mod.audio, "has_audio_stream", return_value=True):
            self.assertEqual(resolve_asr_backend(meta, "auto"), "whisper")


class CaptionsAsPriorTest(unittest.TestCase):
    def test_default_prefers_captions_even_with_media(self):
        meta = _meta(media="clip.mp4", captions="clip.vtt")
        with mock.patch.object(asr_mod.WhisperASR, "is_available", return_value=True):
            self.assertEqual(resolve_asr_backend(meta, "auto"), "captions")

    def test_prior_disabled_prefers_whisper_when_media(self):
        meta = _meta(media="clip.mp4", captions="clip.vtt")
        with mock.patch.object(asr_mod.WhisperASR, "is_available", return_value=True), \
             mock.patch.object(asr_mod.audio, "has_audio_stream", return_value=True):
            self.assertEqual(
                resolve_asr_backend(meta, "auto", captions_as_prior=False), "whisper")

    def test_prior_disabled_falls_back_to_captions_without_whisper(self):
        meta = _meta(media="clip.mp4", captions="clip.vtt")
        with mock.patch.object(asr_mod.WhisperASR, "is_available", return_value=False):
            self.assertEqual(
                resolve_asr_backend(meta, "auto", captions_as_prior=False), "captions")


if __name__ == "__main__":
    unittest.main()
