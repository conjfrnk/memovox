"""Stentor — acquisition + ASR + diarization (spec §3, stages 0–2).

``run()`` ties the stage together: acquire a source, transcribe it, and label
speakers, returning cleaned segments on the master timeline plus source metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..backends.base import Segment
from ..config import Config
from .acquire import EnumeratedEntry, SourceMeta, acquire, enumerate_source
from .asr import run_asr
from .diarize import assign_speakers, speaker_names


@dataclass
class StentorResult:
    meta: SourceMeta
    segments: List[Segment]
    language: Optional[str] = None
    asr_backend: str = ""
    duration: Optional[float] = None
    speaker_names: dict = field(default_factory=dict)


def run(
    config: Config,
    source: str,
    *,
    source_url: Optional[str] = None,
    title: Optional[str] = None,
    captions: Optional[str] = None,
    cookies: Optional[str] = None,
    asr_backend: str = "auto",
    language: Optional[str] = None,
    glossary: Optional[List[str]] = None,
    asr_device: str = "auto",
    asr_compute_type: str = "default",
    asr_allow_cpu: bool = False,
    captions_as_prior: bool = True,
) -> StentorResult:
    meta = acquire(
        config, source, source_url=source_url, title=title, captions=captions, cookies=cookies
    )
    asr = run_asr(config, meta, backend=asr_backend, language=language, glossary=glossary,
                  device=asr_device, compute_type=asr_compute_type, allow_cpu=asr_allow_cpu,
                  captions_as_prior=captions_as_prior)
    segments = assign_speakers(asr.segments)
    return StentorResult(
        meta=meta,
        segments=segments,
        language=asr.language or meta.lang,
        asr_backend=asr.backend,
        duration=asr.duration or meta.duration,
        speaker_names=speaker_names(segments),
    )


__all__ = ["StentorResult", "run", "acquire", "run_asr", "SourceMeta",
           "EnumeratedEntry", "enumerate_source"]
