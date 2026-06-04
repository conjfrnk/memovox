"""Deterministic frame-type classifier (Tessera, spec §4.3) — pure stdlib.

Labels a keyframe ``slide`` / ``document`` / ``diagram`` vs ``talking_head`` from
OCR text density + the grayscale signature's spatial variance. No model. It gates
the opt-in ColPali path (never run on talking-head frames, per spec §12 storage
cost) and is the modality tag the visual leg can use.
"""

from __future__ import annotations

from typing import Sequence

# Heuristic thresholds (synthetic-tunable; refine against real keyframes in M3.4).
_SLIDE_MIN_TOKENS = 8       # on-screen text this dense reads as a slide
_DOCUMENT_MIN_TOKENS = 30   # very dense text reads as a document/page
_DIAGRAM_MIN_VARIANCE = 0.05  # high spatial variance w/ little text => diagram/chart


def _variance(sig: Sequence[float]) -> float:
    n = len(sig)
    if n == 0:
        return 0.0
    mean = sum(sig) / n
    return sum((x - mean) ** 2 for x in sig) / n


def classify_frame(signature: Sequence[float], ocr_text: str = "") -> str:
    """Return ``"slide" | "document" | "diagram" | "talking_head"``."""
    tokens = len((ocr_text or "").split())
    if tokens >= _DOCUMENT_MIN_TOKENS:
        return "document"
    if tokens >= _SLIDE_MIN_TOKENS:
        return "slide"
    if _variance(signature) >= _DIAGRAM_MIN_VARIANCE:
        return "diagram"
    return "talking_head"
