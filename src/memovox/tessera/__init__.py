"""Tessera — the visual track (spec §3, §4 stage 3).

Turns a video stream into timestamped **visual events** — keyframe captions,
on-screen text (OCR), and a visual embedding — by content-aware scene
segmentation and information-gain keyframe selection. This is the tri-modal
"differentiating core": knowledge that exists nowhere in the audio.

``run()`` ties the stage together and degrades gracefully (spec §9): with no
video stream, no ffmpeg, or no frames it returns an empty, ``available=False``
result so transcript-derived knowledge still commits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..backends import get_ocr, get_vlm
from ..backends.base import OCRBackend, VLMBackend
from ..config import Config, Settings
from ..util import slugify
from .frames import FrameSig, bytes_to_signature, extract_keyframe_image, sample_frame_signatures
from .keyframes import select_keyframes
from .scenes import Scene, frame_distance, segment_scenes

__all__ = [
    "FrameSig",
    "bytes_to_signature",
    "sample_frame_signatures",
    "Scene",
    "frame_distance",
    "segment_scenes",
    "select_keyframes",
    "VisualEvent",
    "VisualResult",
    "run",
]


@dataclass
class VisualEvent:
    """A timestamped on-screen event bound to a kept keyframe (spec §4 stage 3)."""

    t_start_s: float
    t_end_s: float
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    embedding: List[float] = field(default_factory=list)
    frame_ref: Optional[str] = None
    scene_index: int = 0
    info_gain: float = 0.0


@dataclass
class VisualResult:
    events: List[VisualEvent] = field(default_factory=list)
    available: bool = False
    reason: str = ""
    n_frames: int = 0
    n_scenes: int = 0
    vlm_backend: str = "none"
    ocr_backend: str = "none"


def run(
    config: Optional[Config],
    meta,
    *,
    settings: Optional[Settings] = None,
    frames: Optional[List[FrameSig]] = None,
    vlm: Optional[VLMBackend] = None,
    ocr: Optional[OCRBackend] = None,
) -> VisualResult:
    """Extract visual events from a source's video stream.

    ``frames`` may be injected (testing / alternate sources); otherwise frames
    are sampled from ``meta.media_path`` via ffmpeg.
    """
    settings = settings or (config.settings if config is not None else Settings())
    if not settings.visual_enabled:
        return VisualResult(available=False, reason="visual track disabled")

    media_path = getattr(meta, "media_path", None) if meta is not None else None
    is_video = getattr(meta, "is_video", False) if meta is not None else False

    if frames is None:
        if not (media_path and is_video):
            return VisualResult(available=False, reason="no video stream")
        frames = sample_frame_signatures(
            media_path,
            fps=settings.frame_sample_fps,
            side=settings.frame_side,
            max_frames=settings.frame_max,
        )
    if not frames:
        return VisualResult(available=False, reason="no frames extracted")

    scenes = segment_scenes(frames, threshold=settings.scene_threshold)
    kept = select_keyframes(
        frames, scenes,
        min_gain=settings.keyframe_min_gain,
        per_scene_cap=settings.keyframe_per_scene_cap,
    )
    scene_of = {i: sc.index for sc in scenes for i in range(sc.start_idx, sc.end_idx + 1)}

    vlm = vlm or get_vlm(settings.vlm_backend, config=config)
    ocr = ocr or get_ocr(settings.ocr_backend, config=config)

    stem = slugify(getattr(meta, "title", "") or "video") if meta is not None else "video"
    events: List[VisualEvent] = []
    for pos, idx in enumerate(kept):
        sig = frames[idx]
        t_start = sig.t
        t_end = frames[kept[pos + 1]].t if pos + 1 < len(kept) else frames[-1].t
        scene_idx = scene_of.get(idx, 0)
        # info_gain is intra-scene novelty; a scene's first keyframe has none.
        is_scene_start = scene_idx < len(scenes) and idx == scenes[scene_idx].start_idx
        gain = 0.0 if is_scene_start else frame_distance(frames[idx - 1].vec, sig.vec)

        image: Optional[Path] = None
        if media_path and config is not None:
            dst = config.frames_dir / f"{stem}_{idx:05d}.jpg"
            image = extract_keyframe_image(media_path, t_start, dst)

        image_arg = str(image) if image else None
        ocr_text = (ocr.extract(image_arg) or "").strip() or None
        caption = (vlm.caption(image_arg, ocr_text=ocr_text) or "").strip() or None

        events.append(
            VisualEvent(
                t_start_s=round(t_start, 3),
                t_end_s=round(max(t_end, t_start), 3),
                caption=caption,
                ocr_text=ocr_text,
                embedding=sig.vec,
                frame_ref=image_arg,
                scene_index=scene_idx,
                info_gain=round(gain, 4),
            )
        )

    return VisualResult(
        events=events,
        available=True,
        n_frames=len(frames),
        n_scenes=len(scenes),
        vlm_backend=vlm.name,
        ocr_backend=ocr.name,
    )
