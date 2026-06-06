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
from ..backends.asr_align import WhisperXAlign
from ..backends.asr_whisper import WhisperASR
from ..backends.base import ASRBackend, ASRResult, Segment
from ..backends.diarize_turns import PyannoteTurns
from ..config import Config
from ..errors import BackendUnavailable, BudgetExceeded, DemuxError, DevicePlacementError
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

# Opt-in trailing upgrades (M0.3 W7): is_available-gated, never on the free path.
_ALIGNERS = {"whisperx": WhisperXAlign}
_TURN_DIARIZERS = {"pyannote-turns": PyannoteTurns}


def get_aligner(name: str = "whisperx", *, config: Optional[Config] = None, **options):
    """Resolve an opt-in forced-alignment backend, or raise BackendUnavailable."""
    cls = _ALIGNERS.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown aligner {name!r}. Options: {list(_ALIGNERS)}.")
    if not cls.is_available():
        raise BackendUnavailable(
            f"Aligner {name!r} is not installed (try: pip install 'memovox[align]')."
        )
    return cls(config=config, **options)


def get_diarizer_turns(name: str = "pyannote-turns", *, config: Optional[Config] = None, **options):
    """Resolve an opt-in diarization-turns backend, or raise BackendUnavailable."""
    cls = _TURN_DIARIZERS.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown diarizer {name!r}. Options: {list(_TURN_DIARIZERS)}.")
    if not cls.is_available():
        raise BackendUnavailable(
            f"Diarizer {name!r} is not installed (try: pip install 'memovox[diarize]')."
        )
    return cls(config=config, **options)


def get_asr(name: str, *, config: Optional[Config] = None, **options) -> ASRBackend:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown ASR backend {name!r}. Options: {list(_REGISTRY)} or 'auto'.")
    if not cls.is_available():
        raise BackendUnavailable(f"ASR backend {name!r} is not installed (try: pip install 'memovox[asr]').")
    return cls(config=config, **options)


def asr_readiness(meta: SourceMeta) -> tuple:
    """Structured §4.1 ASR-readiness verdict ``(ok, reason)`` for the whisper path."""
    if meta.media_path is None:
        return False, "no media acquired"
    if not audio.has_audio_stream(meta.media_path):
        return False, f"no decodable audio stream in {meta.media_path} (ffprobe pre-check)"
    return True, "ok"


def resolve_asr_backend(meta: SourceMeta, requested: str = "auto", *,
                        captions_as_prior: bool = True) -> str:
    """Resolve the ASR backend for ``auto`` (spec §4.1/§9).

    ``captions_as_prior`` (the §9 cost lever, default True) keeps the current
    behavior: free, exact-timing captions win over paying for Whisper whenever a
    transcript is present. Set it False to force re-transcription from media even
    when captions exist. Before committing to ``whisper`` an ffprobe readiness
    pre-check runs and raises ``DemuxError`` early if the media has no audio.
    """
    if requested != "auto":
        return requested
    has_captions = meta.captions_path is not None
    can_whisper = WhisperASR.is_available() and meta.media_path is not None
    if has_captions and (captions_as_prior or not can_whisper):
        return "captions"
    if can_whisper:
        ok, reason = asr_readiness(meta)
        if not ok:
            # Captions are a PRIORITY lever, not whisper-only: if whisper can't run
            # (no audio) but captions exist, use them rather than hard-failing.
            if has_captions:
                return "captions"
            raise DemuxError(f"ASR not ready: {reason}")
        return "whisper"
    if has_captions:
        return "captions"
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
    captions_as_prior: bool = True,
) -> ASRResult:
    """Transcribe ``meta`` into cleaned, ordered segments."""
    name = resolve_asr_backend(meta, backend, captions_as_prior=captions_as_prior)
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

    try:
        result = asr.transcribe(
            audio_path=str(audio_path) if audio_path else None,
            captions_path=str(meta.captions_path) if meta.captions_path else None,
            language=language or meta.lang,
        )
    except (DevicePlacementError, DemuxError, BudgetExceeded):
        # DELIBERATE fail-loud signals (spec §9): a heavy model on CPU, no audio
        # stream, a blown budget — these must surface their actionable message, NOT
        # be silently downgraded to captions. Propagate.
        raise
    except Exception as exc:  # noqa: BLE001
        # A genuine whisper model-load / transcription failure (offline/uncached
        # weights, OOM, an unsupported compute_type). If captions exist, degrade to
        # them rather than aborting with an opaque faster-whisper error — captions are
        # the §9 priority lever. Otherwise the failure is genuine, so re-raise.
        if name == "whisper" and meta.captions_path is not None:
            import sys
            print(f"memovox: whisper ASR failed ({type(exc).__name__}: {exc}); "
                  "falling back to captions.", file=sys.stderr)
            result = get_asr("captions", config=config).transcribe(
                audio_path=None, captions_path=str(meta.captions_path),
                language=language or meta.lang,
            )
        else:
            raise
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
