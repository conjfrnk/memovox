"""Adaptive keyframe selection by information gain (Tessera, spec §4 stage 3).

This is differentiator #2: sample frames by *information gain*, not a fixed
interval. Within each scene the first frame is always kept; a later frame is
kept only when its signature distance from the last *kept* frame exceeds
``min_gain``. Static talking-head segments collapse to a single keyframe;
slide- and demo-dense segments are sampled densely. A per-scene cap and the
``min_gain`` floor (which also suppresses near-duplicates) bound cost.
"""

from __future__ import annotations

from typing import List

from .frames import FrameSig
from .scenes import Scene, frame_distance


def select_keyframes(
    sigs: List[FrameSig],
    scenes: List[Scene],
    *,
    min_gain: float = 0.12,
    per_scene_cap: int = 8,
) -> List[int]:
    """Return the indices of frames to keep, ordered, deduplicated by gain."""
    kept: List[int] = []
    for scene in scenes:
        last = scene.start_idx
        kept.append(last)
        in_scene = 1
        for i in range(scene.start_idx + 1, scene.end_idx + 1):
            if in_scene >= per_scene_cap:
                break
            if frame_distance(sigs[last].vec, sigs[i].vec) >= min_gain:
                kept.append(i)
                last = i
                in_scene += 1
    return kept
