"""Per-video Markdown digest (the human-readable substrate, spec §2).

You can ``cat`` your knowledge: each ingested video gets an inspectable Markdown
file with its Moments and verified claims, deep-linked to the source.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Optional

from ..util import deep_link, format_span
from .models import Claim, Moment, Video
from .store import LoomStore

# Attacker-influenced fields (remote title/channel, OCR'd on-screen text, claim text)
# are interpolated into Markdown. A line break in any of them lets a hostile source
# break out of its inline context and forge structure — most damagingly a fake
# "- ✓ _fact_ … (entail=1.00)" line byte-identical to a genuine entailment-verified
# claim, which a reader ``cat``-ing the digest cannot tell from the real thing
# (breaks the never-claim-without-citation invariant at the digest surface). Collapse
# every line break / control char to a single space so untrusted text stays inline.
_INLINE_BREAK_RE = re.compile(r"[\r\n  \x0b\x0c\x00-\x08\x0e-\x1f\x7f]+")


def _inline(text: Optional[str]) -> str:
    """Flatten untrusted text to a single Markdown line (no structure breakout)."""
    if not text:
        return ""
    return _INLINE_BREAK_RE.sub(" ", text).strip()


def _safe_link(url: Optional[str]) -> Optional[str]:
    """A deep link safe to drop inside ``[..](url)``: a URL containing whitespace,
    control chars, or an unescaped ``)`` / ``(`` would terminate the link and let the
    rest render as injected Markdown, so reject it (the span text still shows)."""
    if not url:
        return None
    if any(c.isspace() for c in url) or any(c in url for c in "()<>"):
        return None
    return url


def render_digest(video: Video, moments: List[Moment], claims: List[Claim]) -> str:
    by_moment = defaultdict(list)
    for c in claims:
        by_moment[c.moment_id].append(c)

    lines: List[str] = [f"# {_inline(video.title) or video.video_id}", ""]
    meta = []
    if video.source_url:
        meta.append(f"Source: {_inline(video.source_url)}")
    if video.channel:
        meta.append(f"Channel: {_inline(video.channel)}")
    if video.published_at:
        meta.append(f"Published: {_inline(video.published_at)}")
    meta.append(f"video_id: `{video.video_id}`")
    lines.append("  \n".join(meta))
    lines.append("")

    for m in moments:
        span = format_span(m.t_start_s, m.t_end_s)
        link = _safe_link(deep_link(video.source_url, m.t_start_s))
        header = f"## [{span}]"
        if link:
            header = f"## [{span}]({link})"
        if m.speaker_id:
            header += f" — {_inline(m.speaker_id)}"
        lines.append(header)
        lines.append("")
        if m.transcript:
            lines.append(f"> {_inline(m.transcript)}")
            lines.append("")
        if m.ocr_text:
            lines.append(f"*on-screen:* {_inline(m.ocr_text)}")
            lines.append("")
        moment_claims = by_moment.get(m.moment_id, [])
        if moment_claims:
            lines.append("**Claims:**")
            for c in moment_claims:
                mark = "✓" if c.status == "committed" else "⚠"
                lines.append(f"- {mark} _{_inline(c.claim_type)}_ {_inline(c.text)}  "
                             f"(entail={c.entailment_score:.2f})")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_digest(store: LoomStore, video_id: str) -> str:
    video = store.get_video(video_id)
    if video is None:
        raise KeyError(f"No video {video_id!r} in store.")
    moments = store.moments_for_video(video_id)
    claims = store.claims_for_video(video_id, status=None)
    return render_digest(video, moments, claims)
