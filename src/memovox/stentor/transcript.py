"""Transcript parsing and cleaning (Stentor).

Parses VTT / SRT / JSON / plain-text transcripts into :class:`Segment` objects,
strips filler tokens and bracketed audio events from the knowledge text while
retaining the events as timeline markers (spec §4 stage 2).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..backends.base import Segment, Word
from ..util import split_sentences

FILLERS = {"um", "uh", "erm", "uhh", "umm", "mm", "hmm", "mhm", "uhm", "ah", "er"}
EVENT_RE = re.compile(
    r"\[\s*(music|applause|laughter|silence|inaudible|noise|crosstalk|cheering|chuckles?)[^\]]*\]",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_SPEAKER_VTT_RE = re.compile(r"<v\s+([^>]+)>")
_SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 ._-]{0,30}):\s+")


def _parse_ts(value: str) -> float:
    s = value.strip().replace(",", ".")
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        h, m, sec = 0.0, nums[0], nums[1]
    else:
        h, m, sec = 0.0, 0.0, nums[0]
    return h * 3600 + m * 60 + sec


def parse_cues(text: str) -> List[Segment]:
    """Parse a VTT or SRT document into raw (uncleaned) segments."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text)
    segments: List[Segment] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        time_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if time_idx is None:
            continue
        try:
            left, right = lines[time_idx].split("-->")
            start = _parse_ts(left)
            end = _parse_ts(right.strip().split(" ")[0])
        except ValueError:
            continue
        raw = " ".join(lines[time_idx + 1:]).strip()
        if not raw:
            continue
        speaker = None
        m = _SPEAKER_VTT_RE.search(raw)
        if m:
            speaker = m.group(1).strip()
        segments.append(Segment(start=start, end=end, text=raw, speaker=speaker))
    return segments


# VTT and SRT share the same block parser.
parse_vtt = parse_cues
parse_srt = parse_cues


def parse_json(data) -> List[Segment]:
    """Parse a JSON transcript: a list of cues or ``{"segments": [...]}``."""
    if isinstance(data, dict):
        data = data.get("segments", [])
    segments: List[Segment] = []
    for item in data or []:
        start = float(item.get("start", item.get("t_start", 0.0)) or 0.0)
        end = float(item.get("end", item.get("t_end", start)) or start)
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        # Optional per-word timings (M0.3): a free-path fixture can carry word
        # precision via a "words": [{"word","start","end"}, ...] array per cue.
        words = [
            Word(word=str(w.get("word", "")), start=float(w.get("start") or 0.0),
                 end=float(w.get("end") or 0.0))  # `or 0.0` tolerates null/absent timings
            for w in (item.get("words") or [])
            if w.get("word")
        ]
        segments.append(Segment(start=start, end=end, text=text,
                                speaker=item.get("speaker"), words=words))
    return segments


def parse_plain(text: str, *, duration: Optional[float] = None) -> List[Segment]:
    """Parse untimed text into sentence segments with synthetic timestamps.

    Without real timing, citations can only preserve *order*; this is a
    last-resort path (real use should supply VTT/SRT or run ASR).
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    per = (duration / len(sentences)) if duration else 3.0
    segments = []
    t = 0.0
    for s in sentences:
        segments.append(Segment(start=round(t, 3), end=round(t + per, 3), text=s))
        t += per
    return segments


def load_transcript(path: "str | Path", *, duration: Optional[float] = None) -> List[Segment]:
    """Load + parse a transcript file, dispatching on extension/content."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()
    if ext == ".vtt" or raw.lstrip().upper().startswith("WEBVTT"):
        return parse_cues(raw)
    if ext == ".srt":
        return parse_cues(raw)
    if ext == ".json":
        try:
            return parse_json(json.loads(raw))
        except ValueError:
            return parse_plain(raw, duration=duration)
    if "-->" in raw:  # unlabeled but looks like cues
        return parse_cues(raw)
    return parse_plain(raw, duration=duration)


def clean_text(raw: str) -> Tuple[str, List[str]]:
    """Return (cleaned speech text, list of audio-event keywords)."""
    events = [e.lower() for e in EVENT_RE.findall(raw)]
    stripped = EVENT_RE.sub(" ", raw)
    stripped = _TAG_RE.sub(" ", stripped)
    stripped = _SPEAKER_PREFIX_RE.sub("", stripped)
    words = [w for w in stripped.split() if _normalize_word(w) not in FILLERS]
    text = re.sub(r"\s+", " ", " ".join(words)).strip()
    return text, events


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def clean_segments(segments: List[Segment]) -> List[Segment]:
    """Strip fillers/events; emit cleaned speech segments + event markers."""
    out: List[Segment] = []
    for seg in segments:
        text, events = clean_text(seg.text)
        speaker = seg.speaker
        if not speaker:
            m = _SPEAKER_PREFIX_RE.match(seg.text)
            if m:
                speaker = m.group(1).strip()
        if text:
            out.append(
                Segment(
                    start=seg.start, end=max(seg.end, seg.start), text=text,
                    speaker=speaker, words=seg.words, kind="speech",
                )
            )
        for ev in events:
            out.append(Segment(start=seg.start, end=seg.start, text=f"[{ev}]", kind="event"))
    out.sort(key=lambda s: (s.start, 0 if s.kind == "speech" else 1))
    return out
