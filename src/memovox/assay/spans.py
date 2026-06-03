"""Bind a claim to its EXACT source span (spec §4.5 provenance).

A Moment may fuse several source segments, each carrying its own time window
(``SegmentRef``, W1.1). A claim, however, typically comes from a single
sentence — so its provenance should point at the segment that sentence lives
in, not the whole-Moment span. ``locate_span`` resolves that narrower window by
token-overlap, falling back to the supplied ``default`` (the Moment span) when
no segment clears the overlap floor or when segments are unavailable (e.g. a
Moment reloaded from the store carries ``segments == []``).
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..util import tokenize


def locate_span(sentence, segments, *, default=None) -> Optional[Tuple[float, float]]:
    """Return the ``(t_start_s, t_end_s)`` of the segment best containing
    ``sentence``, or ``default`` if none clears the 0.5 overlap floor.

    Overlap is the fraction of the sentence's distinct tokens a segment covers
    (set-intersection over sentence tokens), so it stays a true ``[0, 1]``
    fraction — a long segment that repeats the sentence's words cannot beat the
    segment that actually contains it.

    ``segments`` items are unpacked as ``(t0, t1, text)`` — works for both the
    production ``SegmentRef`` NamedTuple and plain 3-tuples. The result is
    ``Optional`` when ``default`` is omitted: a caller unpacking the return
    value must pass a ``default`` 2-tuple (as ``claims.py`` does) or guard
    against ``None``.
    """
    s = set(tokenize(sentence))
    if not s or not segments:
        return default
    best, best_ov = None, 0.0
    for (t0, t1, text) in segments:
        ov = len(set(tokenize(text)) & s) / len(s)
        if ov > best_ov:
            best, best_ov = (t0, t1), ov
    return best if best_ov >= 0.5 else default


def span_text(segments, t_start_s, t_end_s) -> str:
    """Source text of the segments overlapping ``[t_start_s, t_end_s]``.

    Used by Assay's verification gate to build a claim's premise from *only* its
    own located span (W1.2) rather than the whole Moment — so a hallucinated
    claim, whose tokens appear nowhere in its span, is rejected.

    Strict overlap (``s0 < t_end and s1 > t_start``) avoids pulling in a
    boundary-touching neighbour. Items are unpacked as ``(t0, t1, text)`` so it
    works for both ``SegmentRef`` and plain 3-tuples. Returns ``""`` when there
    are no segments (e.g. a store-reloaded Moment), letting callers fall back to
    the whole-Moment text and preserve legacy behaviour.
    """
    parts = [text for (s0, s1, text) in segments if s0 < t_end_s and s1 > t_start_s]
    return " ".join(p.strip() for p in parts if p and p.strip()).strip()
