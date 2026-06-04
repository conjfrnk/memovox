"""Frame sampling and per-frame signatures (Tessera, spec §4 stage 3).

The *signature* of a frame is a small, normalized intensity vector — a
downscaled grayscale thumbnail flattened to ``[0, 1]``. It needs no Python
imaging dependency: ``ffmpeg`` (already required for demux) decodes and
downscales frames to raw grayscale bytes which the standard library turns into
a vector. Signatures drive content-aware scene detection (:mod:`.scenes`) and
information-gain keyframe selection (:mod:`.keyframes`), and double as a free,
deterministic *visual embedding* for retrieval until SigLIP/ColPali is wired.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .. import audio
from ..observe import Span


@dataclass
class FrameSig:
    """A sampled frame: its timestamp and its normalized intensity vector."""

    t: float
    vec: List[float] = field(default_factory=list)


def bytes_to_signature(raw: bytes) -> List[float]:
    """Convert raw 8-bit grayscale bytes to a ``[0, 1]`` intensity vector."""
    return [b / 255.0 for b in raw]


def sample_frame_signatures(
    video_path: "str | Path",
    *,
    fps: float = 1.0,
    side: int = 16,
    max_frames: int = 600,
    span: Optional[Span] = None,
) -> List[FrameSig]:
    """Sample frames from ``video_path`` as signatures via ffmpeg.

    Returns ``[]`` (graceful degradation, spec §9) when ffmpeg is unavailable,
    the file has no video stream, or extraction fails — never raises.
    """
    path = Path(video_path)
    ffmpeg = audio.which_ffmpeg()
    if not ffmpeg or not path.exists():
        return []
    cell = side * side
    cmd = [
        ffmpeg, "-loglevel", "error", "-i", str(path),
        "-vf", f"fps={fps},scale={side}:{side}", "-pix_fmt", "gray",
        "-f", "rawvideo", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        return []
    data = proc.stdout or b""
    available = len(data) // cell if cell else 0
    n = min(available, max_frames) if cell else 0
    if span is not None and cell:
        # frame_max truncates `available` candidate frames to `n`. Note the
        # Settings.frame_max=1200 vs function default 600 mismatch (M0.1 open
        # question) is surfaced here via the actual `max_frames` limit used.
        span.add_cap("frame_max", limit=max_frames, dropped=max(0, available - n))
    sigs: List[FrameSig] = []
    for i in range(n):
        chunk = data[i * cell : (i + 1) * cell]
        sigs.append(FrameSig(t=i / fps if fps else float(i), vec=bytes_to_signature(chunk)))
    return sigs


def extract_keyframe_image(
    video_path: "str | Path", t_seconds: float, dst: "str | Path"
) -> Optional[Path]:
    """Extract a full-resolution keyframe image at ``t_seconds`` for OCR/VLM."""
    return audio.extract_frame(video_path, t_seconds, dst)
