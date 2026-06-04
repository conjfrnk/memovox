"""Dependency-free helpers: time, identifiers, deep links, text, hashing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlsplit

# --------------------------------------------------------------------------- #
# time
# --------------------------------------------------------------------------- #


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return utcnow().replace(microsecond=0).isoformat()


def parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Accept a bare date (YYYY-MM-DD).
        try:
            dt = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def seconds_to_hms(seconds: Optional[float]) -> str:
    """Format seconds as ``H:MM:SS`` (or ``M:SS`` under an hour)."""
    if seconds is None:
        return "?"
    total = int(round(max(0.0, float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_span(t_start: Optional[float], t_end: Optional[float]) -> str:
    return f"{seconds_to_hms(t_start)}–{seconds_to_hms(t_end)}"


# --------------------------------------------------------------------------- #
# identifiers (deterministic -> idempotent ingestion)
# --------------------------------------------------------------------------- #


def sha1_hex(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def short_hash(data, length: int = 16) -> str:
    return sha1_hex(data)[:length]


def content_hash_file(path, chunk_size: int = 1 << 20) -> str:
    """SHA-1 of a file's bytes, read in chunks (idempotency key for media)."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def youtube_id(url: str) -> Optional[str]:
    """Extract a YouTube video id from common URL shapes, else ``None``."""
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower().lstrip("www.")
    if host == "youtu.be":
        vid = parts.path.lstrip("/").split("/")[0]
        return vid or None
    if host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        if parts.path == "/watch":
            vals = parse_qs(parts.query).get("v")
            return vals[0] if vals else None
        for prefix in ("/embed/", "/shorts/", "/v/", "/live/"):
            if parts.path.startswith(prefix):
                return parts.path[len(prefix):].split("/")[0] or None
    return None


def make_video_id(source: str, *, content_hash: Optional[str] = None) -> str:
    """Stable id for a video. YouTube → ``yt:<id>``; otherwise content/source hash."""
    yt = youtube_id(source)
    if yt:
        return f"yt:{yt}"
    if content_hash:
        return f"vid:{content_hash[:16]}"
    return f"vid:{short_hash(source)}"


def make_moment_id(video_id: str, index: int) -> str:
    return f"{video_id}#m{index:04d}"


def make_claim_id(moment_id: str, index: int) -> str:
    return f"{moment_id}.c{index:02d}"


def deep_link(source_url: Optional[str], t_start: Optional[float]) -> Optional[str]:
    """Build a timestamped deep link into the source, if possible."""
    if not source_url:
        return None
    t = int(t_start or 0)
    yt = youtube_id(source_url)
    if yt:
        return f"https://youtu.be/{yt}?t={t}"
    sep = "&" if ("?" in source_url) else "#"
    return f"{source_url}{sep}t={t}"


def deep_link_range(source_url: Optional[str], t_start: Optional[float],
                    t_end: Optional[float]) -> Optional[str]:
    """A RANGED deep link (M2.3). YouTube → ``watch?v=<id>&start=<t0>&end=<t1>``
    (integer seconds, the only host that honors the range); any other source has no
    standard ranged fragment, so fall back to the start-only :func:`deep_link`."""
    if not source_url:
        return None
    yt = youtube_id(source_url)
    if yt:
        return (f"https://www.youtube.com/watch?v={yt}"
                f"&start={int(t_start or 0)}&end={int(t_end or 0)}")
    return deep_link(source_url, t_start)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #

_slug_re = re.compile(r"[^a-z0-9]+")
_sentence_split_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_word_re = re.compile(r"[a-z0-9]+")


def slugify(text: str, *, max_len: int = 60, default: str = "item") -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _slug_re.sub("-", ascii_text).strip("-")
    if max_len and len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or default


def split_sentences(text: str) -> List[str]:
    """Naive but robust sentence splitter (stdlib only)."""
    text = (text or "").strip()
    if not text:
        return []
    parts = _sentence_split_re.split(text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    return _word_re.findall((text or "").lower())


def truncate(text: str, max_len: int = 80) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
