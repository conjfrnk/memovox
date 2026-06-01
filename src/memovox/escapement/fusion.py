"""Escapement — temporal fusion into Moments (spec §3 stage 4).

Merges speech segments (and, later, visual events) into coherent, time-bounded
**Moments**. Boundaries are placed at natural seams — speaker changes, silences,
audio events, the max-duration cap, and (when an embedder is supplied) topic
shifts detected by embedding similarity — rather than at arbitrary token counts.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

from ..backends.base import Embedder, Segment
from ..config import Settings
from ..loom.models import Moment
from ..util import make_moment_id
from ..vectormath import cosine


def _event_between(event_times: List[float], lo: float, hi: float) -> bool:
    return any(lo <= t <= hi for t in event_times)


def _similarity(embedder: Embedder, text_a: str, text_b: str) -> float:
    va, vb = embedder.embed([text_a, text_b])
    return cosine(va, vb)


def _dominant_speaker(segs: List[Segment]) -> Optional[str]:
    speakers = [s.speaker for s in segs if s.speaker]
    if not speakers:
        return None
    return Counter(speakers).most_common(1)[0][0]


def _make_moment(video_id: str, index: int, segs: List[Segment]) -> Moment:
    transcript = " ".join(s.text.strip() for s in segs if s.text.strip()).strip()
    return Moment(
        moment_id=make_moment_id(video_id, index),
        video_id=video_id,
        t_start_s=round(segs[0].start, 3),
        t_end_s=round(max(s.end for s in segs), 3),
        transcript=transcript,
        speaker_id=_dominant_speaker(segs),
        index=index,
    )


def _overlaps(event, t_start: float, t_end: float) -> bool:
    return event.t_start_s <= t_end and event.t_end_s >= t_start


def _overlapping(events, t_start: float, t_end: float) -> list:
    return [e for e in events if _overlaps(e, t_start, t_end)]


def _join_distinct(values) -> str:
    seen: List[str] = []
    for value in values:
        text = (value or "").strip()
        if text and text not in seen:
            seen.append(text)
    return " ".join(seen)


def _attach_visuals(moments: List[Moment], visual_events) -> None:
    """Bind co-occurring visual events to each Moment (spec §4 stage 4).

    On-screen text and captions for events overlapping a Moment's span become
    that Moment's ``ocr_text`` / ``visual_caption`` — so a Moment binds speech +
    slide + speaker, and the on-screen knowledge flows into retrieval.
    """
    for moment in moments:
        overlap = _overlapping(visual_events, moment.t_start_s, moment.t_end_s)
        if not overlap:
            continue
        moment.ocr_text = _join_distinct(e.ocr_text for e in overlap) or None
        moment.visual_caption = _join_distinct(e.caption for e in overlap) or None


def moment_visual_embedding(moment: Moment, visual_events) -> Optional[List[float]]:
    """Mean visual embedding of the events overlapping ``moment`` (or ``None``)."""
    vecs = [e.embedding for e in _overlapping(visual_events, moment.t_start_s, moment.t_end_s) if e.embedding]
    if not vecs:
        return None
    # Anchor to the most common dimension so one anomalous vector can't drop the rest.
    dims = [len(v) for v in vecs]
    dim = max(set(dims), key=dims.count)
    acc = [0.0] * dim
    n = 0
    for vec in vecs:
        if len(vec) != dim:
            continue
        for i, x in enumerate(vec):
            acc[i] += x
        n += 1
    if n == 0:
        return None
    return [x / n for x in acc]


def _merge_small(groups: List[List[Segment]], settings: Settings) -> List[List[Segment]]:
    """Fold sub-minimum-duration groups into a neighbor to avoid fragmentation."""
    if len(groups) <= 1:
        return groups
    merged: List[List[Segment]] = []
    for group in groups:
        duration = group[-1].end - group[0].start
        same_speaker = merged and _dominant_speaker(merged[-1]) == _dominant_speaker(group)
        if merged and duration < settings.moment_min_sec and same_speaker:
            merged[-1].extend(group)
        else:
            merged.append(list(group))
    return merged


def build_moments(
    video_id: str,
    segments: List[Segment],
    *,
    embedder: Optional[Embedder] = None,
    settings: Optional[Settings] = None,
    visual_events=None,
) -> List[Moment]:
    settings = settings or Settings()
    speech = [s for s in segments if s.kind == "speech" and s.text.strip()]
    event_times = sorted(s.start for s in segments if s.kind == "event")
    if not speech:
        return []
    speech.sort(key=lambda s: s.start)

    groups: List[List[Segment]] = []
    current: List[Segment] = [speech[0]]
    for prev, seg in zip(speech, speech[1:]):
        gap = seg.start - prev.end
        cur_dur = current[-1].end - current[0].start
        cur_speaker = _dominant_speaker(current)

        boundary = False
        if seg.speaker and cur_speaker and seg.speaker != cur_speaker:
            boundary = True
        elif gap > settings.moment_gap_sec:
            boundary = True
        elif cur_dur >= settings.moment_max_sec:
            boundary = True
        elif _event_between(event_times, prev.end, seg.start):
            boundary = True
        elif embedder is not None and cur_dur >= settings.moment_min_sec:
            sim = _similarity(
                embedder, " ".join(s.text for s in current[-3:]), seg.text
            )
            if sim < settings.boundary_similarity:
                boundary = True

        if boundary:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)
    groups.append(current)

    groups = _merge_small(groups, settings)
    moments = [_make_moment(video_id, i, g) for i, g in enumerate(groups)]
    if visual_events:
        _attach_visuals(moments, visual_events)
    return moments
