"""Stentor stage 2 — ASR orchestration.

Selects an ASR backend based on what's available: an existing transcript
(``captions``, free), faster-whisper (``whisper``, optional), or a deterministic
``fake`` backend for tests. Demuxes media to 16 kHz mono WAV before Whisper and
runs the ffprobe validation pre-check (spec §4 stages 1–2).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .. import audio
from ..backends.asr_whisper import WhisperASR
from ..backends.base import ASRBackend, ASRResult, Segment
from ..config import Config
from ..errors import BackendUnavailable, DemuxError
from .acquire import SourceMeta
from .transcript import clean_segments, load_transcript


class FakeASR(ASRBackend):
    """Deterministic ASR for tests: turns provided text/segments into output."""

    name = "fake"

    def transcribe(self, audio_path=None, *, captions_path=None, language=None) -> ASRResult:
        segs = self.options.get("segments")
        if segs:
            segments = [
                Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"],
                        speaker=s.get("speaker"))
                for s in segs
            ]
        else:
            from ..util import split_sentences

            segments = []
            t = 0.0
            for sentence in split_sentences(self.options.get("text", "")):
                segments.append(Segment(start=t, end=t + 3.0, text=sentence))
                t += 3.0
        return ASRResult(segments=segments, language=language, backend=self.name)


class CaptionsASR(ASRBackend):
    """Use an existing transcript file as the ASR result (free, exact timing)."""

    name = "captions"

    def transcribe(self, audio_path=None, *, captions_path=None, language=None) -> ASRResult:
        if not captions_path:
            raise ValueError("CaptionsASR requires a captions_path.")
        segments = load_transcript(captions_path)
        return ASRResult(segments=segments, language=language, backend=self.name)


_REGISTRY = {"whisper": WhisperASR, "captions": CaptionsASR, "fake": FakeASR}


def get_asr(name: str, *, config: Optional[Config] = None, **options) -> ASRBackend:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown ASR backend {name!r}. Options: {list(_REGISTRY)} or 'auto'.")
    if not cls.is_available():
        raise BackendUnavailable(f"ASR backend {name!r} is not installed (try: pip install 'memovox[asr]').")
    return cls(config=config, **options)


def resolve_asr_backend(meta: SourceMeta, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if meta.captions_path is not None:
        return "captions"
    if WhisperASR.is_available() and meta.media_path is not None:
        return "whisper"
    raise BackendUnavailable(
        "No ASR backend available for this source. Either supply a transcript "
        "(.vtt/.srt/.json), or install offline ASR: pip install 'memovox[asr]'."
    )


def run_asr(
    config: Config,
    meta: SourceMeta,
    *,
    backend: str = "auto",
    language: Optional[str] = None,
    glossary: Optional[List[str]] = None,
    device: str = "auto",
    compute_type: str = "default",
    allow_cpu: bool = False,
) -> ASRResult:
    """Transcribe ``meta`` into cleaned, ordered segments."""
    name = resolve_asr_backend(meta, backend)
    options = {}
    if name == "whisper":
        if glossary:
            options["glossary_prompt"] = ", ".join(glossary)
        options["device"] = device
        options["compute_type"] = compute_type
        options["allow_cpu"] = allow_cpu
    asr = get_asr(name, config=config, **options)

    audio_path = None
    if name == "whisper":
        audio_path = _prepare_audio(config, meta)

    result = asr.transcribe(
        audio_path=str(audio_path) if audio_path else None,
        captions_path=str(meta.captions_path) if meta.captions_path else None,
        language=language or meta.lang,
    )
    result.segments = clean_segments(result.segments)
    if result.duration is None:
        result.duration = meta.duration
    return result


def _prepare_audio(config: Config, meta: SourceMeta) -> Path:
    if meta.media_path is None:
        raise BackendUnavailable("Whisper ASR needs media but none was acquired.")
    if not audio.has_audio_stream(meta.media_path):
        raise DemuxError(f"No decodable audio stream in {meta.media_path} (ffprobe pre-check).")
    wav = config.media_dir / f"{Path(meta.media_path).stem}.16k.wav"
    if not wav.exists():
        audio.demux_to_wav(meta.media_path, wav)
    return wav
