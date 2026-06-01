"""Stentor — speaker diarization.

Default fallback: keep any speaker labels already present (e.g. from VTT ``<v>``
tags); otherwise assign a single ``spk_0``. Optional upgrade: pyannote.audio for
real speaker turns + voiceprints (``pip install "memovox[diarize]"``). Per spec
§12, cross-video diarization is hard, so we start within-video and conservative.
"""

from __future__ import annotations

import importlib.util
from typing import List

from ..backends.base import Segment


def pyannote_available() -> bool:
    # find_spec raises (not returns None) when a *parent* package is missing.
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except (ImportError, ValueError):
        return False


def assign_speakers(segments: List[Segment], *, default: str = "spk_0") -> List[Segment]:
    """Ensure every speech segment has a speaker label (free fallback)."""
    seen = {}
    next_idx = 0
    for seg in segments:
        if seg.kind != "speech":
            continue
        if seg.speaker:
            # Normalize named speakers to stable spk_* ids while remembering names.
            if seg.speaker not in seen:
                seen[seg.speaker] = seg.speaker
        else:
            seg.speaker = default
    return segments


def speaker_names(segments: List[Segment]) -> dict:
    """Map of speaker_id -> display name gathered from the transcript."""
    names = {}
    for seg in segments:
        if seg.kind == "speech" and seg.speaker and not seg.speaker.startswith("spk_"):
            names.setdefault(seg.speaker, seg.speaker)
    return names
