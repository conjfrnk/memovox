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


def locate_span(sentence, segments, *, default=None, tighten=True) -> Optional[Tuple[float, float]]:
    """Return the ``(t_start_s, t_end_s)`` of the segment best containing
    ``sentence``, or ``default`` if none clears the 0.5 overlap floor.

    Overlap is the fraction of the sentence's distinct tokens a segment covers
    (set-intersection over sentence tokens), so it stays a true ``[0, 1]``
    fraction — a long segment that repeats the sentence's words cannot beat the
    segment that actually contains it.

    **Word-window tightening (M0.3, spec §4.5):** when the best segment carries
    per-word timings (``SegmentRef.words``), the returned window is narrowed to the
    span of the words that actually match the sentence — "the 0.7 s she says it",
    not the whole 10 s cue. With no words (the free captions path) the segment
    window is returned **unchanged** (identity). ``tighten=False`` forces the
    cue-granular window. This narrows only the citation/display span; the NLI
    premise (``span_text``) stays segment-granular (see ``assay/__init__``).

    ``segments`` items are unpacked as ``(t0, t1, text, *words)`` — works for both
    the production ``SegmentRef`` NamedTuple and plain 3-tuples. The result is
    ``Optional`` when ``default`` is omitted.
    """
    s = set(tokenize(sentence))
    if not s or not segments:
        return default
    best, best_ov, best_words = None, 0.0, ()
    for (t0, t1, text, *rest) in segments:  # *rest tolerates SegmentRef.words (M0.3)
        ov = len(set(tokenize(text)) & s) / len(s)
        if ov > best_ov:
            best, best_ov, best_words = (t0, t1), ov, (rest[0] if rest else ())
    if best is None or best_ov < 0.5:
        return default
    if tighten and best_words:
        matched = [(w0, w1) for (w0, w1, word) in best_words
                   if (tok := tokenize(word)) and tok[0] in s]
        if matched:
            lo, hi = best
            return (max(lo, min(m[0] for m in matched)), min(hi, max(m[1] for m in matched)))
    return best


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
    parts = [text for (s0, s1, text, *_rest) in segments if s0 < t_end_s and s1 > t_start_s]
    return " ".join(p.strip() for p in parts if p and p.strip()).strip()
