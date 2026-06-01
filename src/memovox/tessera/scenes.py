"""Content-aware scene segmentation (Tessera, spec §4 stage 3).

A scene boundary is a *content* discontinuity — a hard cut or a slide change —
detected when the frame-to-frame signature distance exceeds a threshold. This is
the dependency-free analogue of PySceneDetect's content detector (and of
perceptual-hash slide-change detection): a 50-minute single-shot lecture still
splits into one scene per slide. ``segment_scenes`` operates purely on
signatures, so it is fully testable without ffmpeg or a real video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .frames import FrameSig


@dataclass
class Scene:
    index: int
    start_idx: int
    end_idx: int
    t_start: float
    t_end: float


def frame_distance(a: List[float], b: List[float]) -> float:
    """Mean absolute difference between two intensity vectors, in ``[0, 1]``.

    Unlike cosine similarity this registers pure brightness changes (a fade to
    black, a slide swap), matching how content-aware detectors behave.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(abs(a[i] - b[i]) for i in range(n)) / n


def segment_scenes(sigs: List[FrameSig], *, threshold: float = 0.3) -> List[Scene]:
    """Split a frame-signature sequence into scenes at content discontinuities."""
    if not sigs:
        return []
    scenes: List[Scene] = []
    start = 0
    for i in range(1, len(sigs)):
        if frame_distance(sigs[i - 1].vec, sigs[i].vec) > threshold:
            scenes.append(
                Scene(
                    index=len(scenes),
                    start_idx=start,
                    end_idx=i - 1,
                    t_start=sigs[start].t,
                    t_end=sigs[i - 1].t,
                )
            )
            start = i
    scenes.append(
        Scene(
            index=len(scenes),
            start_idx=start,
            end_idx=len(sigs) - 1,
            t_start=sigs[start].t,
            t_end=sigs[-1].t,
        )
    )
    return scenes
