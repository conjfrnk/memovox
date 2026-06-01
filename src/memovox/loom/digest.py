"""Per-video Markdown digest (the human-readable substrate, spec §2).

You can ``cat`` your knowledge: each ingested video gets an inspectable Markdown
file with its Moments and verified claims, deep-linked to the source.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

from ..util import deep_link, format_span
from .models import Claim, Moment, Video
from .store import LoomStore


def render_digest(video: Video, moments: List[Moment], claims: List[Claim]) -> str:
    by_moment = defaultdict(list)
    for c in claims:
        by_moment[c.moment_id].append(c)

    lines: List[str] = [f"# {video.title or video.video_id}", ""]
    meta = []
    if video.source_url:
        meta.append(f"Source: {video.source_url}")
    if video.channel:
        meta.append(f"Channel: {video.channel}")
    if video.published_at:
        meta.append(f"Published: {video.published_at}")
    meta.append(f"video_id: `{video.video_id}`")
    lines.append("  \n".join(meta))
    lines.append("")

    for m in moments:
        span = format_span(m.t_start_s, m.t_end_s)
        link = deep_link(video.source_url, m.t_start_s)
        header = f"## [{span}]"
        if link:
            header = f"## [{span}]({link})"
        if m.speaker_id:
            header += f" — {m.speaker_id}"
        lines.append(header)
        lines.append("")
        if m.transcript:
            lines.append(f"> {m.transcript}")
            lines.append("")
        if m.ocr_text:
            lines.append(f"*on-screen:* {m.ocr_text}")
            lines.append("")
        moment_claims = by_moment.get(m.moment_id, [])
        if moment_claims:
            lines.append("**Claims:**")
            for c in moment_claims:
                mark = "✓" if c.status == "committed" else "⚠"
                lines.append(f"- {mark} _{c.claim_type}_ {c.text}  (entail={c.entailment_score:.2f})")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_digest(store: LoomStore, video_id: str) -> str:
    video = store.get_video(video_id)
    if video is None:
        raise KeyError(f"No video {video_id!r} in store.")
    moments = store.moments_for_video(video_id)
    claims = store.claims_for_video(video_id, status=None)
    return render_digest(video, moments, claims)
