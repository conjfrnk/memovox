"""Stentor stage 0 — acquire a source (local file or URL).

Local media/transcript files work with zero dependencies. URL acquisition uses
``yt-dlp`` when installed (``pip install "memovox[acquire]"``); otherwise a clear
error explains how to enable it. The acquisition layer is deliberately modular
so yt-dlp breakage stays isolated (spec §4 stage 0).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import audio
from ..config import Config
from ..errors import AcquisitionError
from ..util import content_hash_file, sha1_hex

TRANSCRIPT_EXTENSIONS = {".vtt", ".srt", ".json", ".txt"}


@dataclass
class SourceMeta:
    source_url: Optional[str]
    title: str
    channel: Optional[str] = None
    published_at: Optional[str] = None
    duration: Optional[float] = None
    lang: Optional[str] = None
    content_hash: Optional[str] = None
    media_path: Optional[Path] = None
    captions_path: Optional[Path] = None
    is_video: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class EnumeratedEntry:
    """One child of a channel/playlist (M3.2) — metadata only, no media."""

    video_id: str
    url: str
    title: Optional[str] = None


def enumerate_source(config: Config, url: str) -> list:
    """Expand a channel/playlist URL into its child videos via yt-dlp
    ``--flat-playlist`` (metadata only — NEVER downloads). A bare video URL
    enumerates to a single entry, so sync treats all sources uniformly. Raises
    :class:`AcquisitionError` (with an install hint) when yt-dlp is absent."""
    from ..util import make_video_id, youtube_id

    if not _is_url(url):
        # A local path (free-path sync) enumerates to itself — no yt-dlp needed.
        return [EnumeratedEntry(video_id=make_video_id(url), url=url, title=None)]
    if not shutil.which("yt-dlp"):
        raise AcquisitionError(
            "Subscription enumeration requires yt-dlp, which was not found.\n"
            "Install it with: pip install 'memovox[acquire]'  (or `pip install yt-dlp`)."
        )
    cmd = ["yt-dlp", "--flat-playlist", "--dump-single-json", "--no-warnings", url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        raise AcquisitionError(f"yt-dlp enumeration timed out (>3 min) for {url}.")
    if proc.returncode != 0:
        raise AcquisitionError(f"yt-dlp enumeration failed: {proc.stderr.strip()[:500]}")
    try:
        data = json.loads(proc.stdout)
    except ValueError as exc:
        raise AcquisitionError(f"yt-dlp returned unparseable JSON: {exc}") from exc

    raw_entries = data.get("entries")
    raw_entries = raw_entries if raw_entries is not None else [data]  # single video
    out = []
    for entry in raw_entries:
        if not entry:
            continue
        eid = entry.get("id")
        eurl = entry.get("url") or entry.get("webpage_url")
        # A flat-playlist "url" is sometimes a bare video id -> build a watch URL.
        if eid and (not eurl or not eurl.startswith("http")):
            eurl = f"https://youtu.be/{eid}"
        # Skip an entry with no id AND no real URL — a bare-id url without an id would
        # otherwise mint an unstable vid:<hash> the cursor could never match.
        if not eurl or (not eid and not eurl.startswith("http")):
            continue
        video_id = f"yt:{eid}" if eid else make_video_id(eurl)
        out.append(EnumeratedEntry(video_id=video_id, url=eurl, title=entry.get("title")))
    return out


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def acquire(
    config: Config,
    source: str,
    *,
    source_url: Optional[str] = None,
    title: Optional[str] = None,
    captions: Optional[str] = None,
    cookies: Optional[str] = None,
    want_video: bool = False,
) -> SourceMeta:
    config.ensure()
    if _is_url(source):
        return _acquire_url(config, source, cookies=cookies, title=title, want_video=want_video)
    return _acquire_local(source, source_url=source_url, title=title, captions=captions)


def _sidecar_published_at(path: Path) -> Optional[str]:
    """A local file's publish date from a ``<stem>.meta.json`` sidecar (M3.1), so
    local sources (and the dated golden variant) can carry a date without a network
    fetch. Reads ``published_at`` (ISO) or a yt-dlp-style ``upload_date``."""
    for cand in (path.with_name(path.name + ".meta.json"),
                 path.with_name(path.name.split(".")[0] + ".meta.json")):
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            return data.get("published_at") or _format_date(data.get("upload_date"))
    return None


def _acquire_local(
    source: str,
    *,
    source_url: Optional[str],
    title: Optional[str],
    captions: Optional[str],
) -> SourceMeta:
    if not source or not str(source).strip():
        raise AcquisitionError("Empty source path.")
    path = Path(source).expanduser()
    if not path.exists():
        raise AcquisitionError(f"No such file: {path}")
    if not path.is_file():
        # A directory (or other non-file) would crash later with a cryptic
        # IsADirectoryError in content_hash_file — fail clean here instead.
        raise AcquisitionError(f"Not a file: {path}")
    ext = path.suffix.lower()
    chash = content_hash_file(path)
    display_title = title or path.stem
    published_at = _sidecar_published_at(path)

    if ext in TRANSCRIPT_EXTENSIONS and ext not in audio.MEDIA_EXTENSIONS:
        # Transcript-only ingest — fully free, no media/ffmpeg required.
        return SourceMeta(
            source_url=source_url,
            title=display_title,
            content_hash=chash,
            captions_path=path,
            media_path=None,
            is_video=False,
            published_at=published_at,
        )

    if ext in audio.MEDIA_EXTENSIONS:
        info = audio.probe(path)
        captions_path = Path(captions).expanduser() if captions else _sibling_captions(path)
        return SourceMeta(
            source_url=source_url,
            title=display_title,
            duration=info.get("duration"),
            content_hash=chash,
            media_path=path,
            captions_path=captions_path,
            is_video=info.get("has_video", False),
            published_at=published_at,
        )

    raise AcquisitionError(
        f"Unsupported file type {ext!r}. Provide a media file "
        f"({', '.join(sorted(audio.MEDIA_EXTENSIONS))}) or a transcript "
        f"({', '.join(sorted(TRANSCRIPT_EXTENSIONS))})."
    )


def _sibling_captions(media: Path) -> Optional[Path]:
    for ext in (".vtt", ".srt"):
        for cand in (media.with_suffix(ext), media.with_suffix(f".en{ext}")):
            if cand.exists():
                return cand
    return None


def _acquire_url(
    config: Config, url: str, *, cookies: Optional[str], title: Optional[str],
    want_video: bool = False,
) -> SourceMeta:
    if not shutil.which("yt-dlp"):
        raise AcquisitionError(
            "URL ingestion requires yt-dlp, which was not found.\n"
            "Install it with: pip install 'memovox[acquire]'  (or `pip install yt-dlp`).\n"
            "Alternatively, download the file/captions yourself and ingest the local path."
        )
    out_tmpl = str(config.media_dir / "%(id)s.%(ext)s")
    # Audio-only by default (cheap, and the free ASR path is captions/whisper).
    # ``want_video`` (CLI --with-video) fetches video+audio so Tessera's visual
    # track (keyframes/OCR/VLM) can run — otherwise the whole visual modality is
    # unreachable from a URL. The downloaded stream is probed below, so is_video
    # reflects reality regardless of what yt-dlp actually delivered.
    fmt = "bv*+ba/b" if want_video else "bestaudio/best"
    cmd = [
        "yt-dlp", "-f", fmt, "-o", out_tmpl,
        "--write-info-json", "--write-subs", "--write-auto-subs",
        "--sub-format", "vtt", "--sub-langs", "en.*,en",
        "--no-playlist", "--restrict-filenames",
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    cmd.append(url)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        raise AcquisitionError(f"yt-dlp download timed out (>60 min) for {url}.")
    if proc.returncode != 0:
        raise AcquisitionError(f"yt-dlp failed: {proc.stderr.strip()[:500]}")

    info_files = sorted(config.media_dir.glob("*.info.json"), key=lambda p: p.stat().st_mtime)
    meta_json = {}
    if info_files:
        try:
            meta_json = json.loads(info_files[-1].read_text(encoding="utf-8"))
        except (ValueError, OSError):
            meta_json = {}

    vid = meta_json.get("id", "")
    media_path = _find_media(config.media_dir, vid)
    captions_path = _find_captions(config.media_dir, vid)
    # Probe the ACTUAL downloaded stream rather than assuming video. ``bestaudio``
    # yields an audio-only container, so hardcoding is_video=True made Tessera shell
    # ffmpeg onto a streamless file (a noisy "visual track dropped" error). A real
    # probe means audio-only -> Tessera cleanly skips ("no video stream").
    has_video = bool(media_path and audio.probe(media_path).get("has_video"))
    return SourceMeta(
        source_url=meta_json.get("webpage_url", url),
        title=title or meta_json.get("title") or (media_path.stem if media_path else url),
        channel=meta_json.get("channel") or meta_json.get("uploader"),
        published_at=_format_date(meta_json.get("upload_date")),
        duration=meta_json.get("duration"),
        lang=meta_json.get("language"),
        content_hash=content_hash_file(media_path) if media_path else sha1_hex(url),
        media_path=media_path,
        captions_path=captions_path,
        is_video=has_video,
        extra={"description": meta_json.get("description")},
    )


def _find_media(media_dir: Path, vid: str) -> Optional[Path]:
    for p in sorted(media_dir.glob(f"{vid}.*") if vid else media_dir.glob("*")):
        if p.suffix.lower() in audio.MEDIA_EXTENSIONS:
            return p
    return None


def _find_captions(media_dir: Path, vid: str) -> Optional[Path]:
    cands = sorted(media_dir.glob(f"{vid}*.vtt")) or sorted(media_dir.glob(f"{vid}*.srt"))
    return cands[0] if cands else None


def _format_date(yyyymmdd: Optional[str]) -> Optional[str]:
    if yyyymmdd and len(yyyymmdd) == 8 and yyyymmdd.isdigit():
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    return None
