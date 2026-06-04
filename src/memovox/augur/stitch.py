"""Answer-with-video clip stitching (spec §5/§8).

Pure, deterministic arithmetic: merge each video's adjacent/overlapping cited
spans into the fewest non-overlapping watch windows, mint ranged deep links, and
return :class:`Clip` objects. No models, no I/O, no randomness — it only WIDENS to
the union of already-verified citation spans (never narrows or invents), so it
cannot regress provenance. ``render_clip`` is an opt-in ffmpeg cut, a no-op on the
free path (no local media / no ffmpeg).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from ..util import deep_link_range, slugify
from .types import Clip


def stitch_clips(citations, *, videos: dict, merge_gap_s: float = 2.5) -> List[Clip]:
    """Merge cited spans into minimal, non-overlapping, sorted clips per video.

    Groups by ``video_id`` (first-seen citation order), sorts each group by
    ``(t_start, t_end)``, and sweep-merges any two windows whose gap is
    ``<= merge_gap_s`` (overlaps included) into their union. Deterministic.
    """
    by_video: dict = {}
    for c in citations:
        by_video.setdefault(c.video_id, []).append(c)

    clips: List[Clip] = []
    for video_id, group in by_video.items():
        video = videos.get(video_id)
        ordered = sorted(group, key=lambda c: (c.t_start_s, c.t_end_s))
        cur_start = cur_end = None
        cur_idx: List[int] = []
        for c in ordered:
            if cur_start is None:
                cur_start, cur_end, cur_idx = c.t_start_s, c.t_end_s, [c.index]
            elif c.t_start_s <= cur_end + merge_gap_s:
                cur_end = max(cur_end, c.t_end_s)
                cur_idx.append(c.index)
            else:
                clips.append(_make_clip(video_id, video, cur_start, cur_end, cur_idx))
                cur_start, cur_end, cur_idx = c.t_start_s, c.t_end_s, [c.index]
        if cur_start is not None:
            clips.append(_make_clip(video_id, video, cur_start, cur_end, cur_idx))
    return clips


def _make_clip(video_id, video, t_start, t_end, indices) -> Clip:
    source_url = getattr(video, "source_url", None)
    return Clip(
        video_id=video_id,
        t_start_s=t_start,
        t_end_s=t_end,
        title=getattr(video, "title", None),
        deep_link=deep_link_range(source_url, t_start, t_end) if source_url else None,
        citation_indices=sorted(set(indices)),
    )


def render_clip(video, clip: Clip, *, media_path: Optional[str] = None,
                out_dir) -> Optional[Path]:
    """Cut ``clip``'s span from local media to an ``.mp4`` (opt-in, spec §5/§9).

    Returns ``None`` (a silent no-op) unless ``media_path`` exists AND ffmpeg is on
    PATH — the free/CI path. Never writes to stdout (MCP speaks JSON-RPC there);
    graceful-degrades to ``None`` on any ffmpeg/OS error.
    """
    from ..audio import which_ffmpeg

    ffmpeg = which_ffmpeg()
    if not media_path or not Path(media_path).exists() or not ffmpeg:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slugify(clip.video_id)}_{int(clip.t_start_s)}-{int(clip.t_end_s)}.mp4"
    cmd = [ffmpeg, "-y", "-ss", str(clip.t_start_s), "-to", str(clip.t_end_s),
           "-i", str(media_path), "-c", "copy", str(out)]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=120, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out if out.exists() else None
