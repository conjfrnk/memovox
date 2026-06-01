"""Media helpers built on ffmpeg/ffprobe with stdlib fallbacks.

Stentor stage 1 (demux) and the ffprobe validation pre-check live here. WAV
duration is read via the stdlib :mod:`wave` module so the common path needs no
external tooling; everything else degrades to ``None``/``False`` when ffmpeg is
absent rather than raising.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional

from .errors import DemuxError

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".ts"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".aac", ".opus", ".wma"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def which_ffprobe() -> Optional[str]:
    return shutil.which("ffprobe")


def probe(path: "str | Path") -> dict:
    """Return ``{has_audio, has_video, duration, codecs}`` via ffprobe.

    Falls back to a WAV header read when ffprobe is unavailable.
    """
    p = Path(path)
    info = {"has_audio": False, "has_video": False, "duration": None, "codecs": []}
    ffprobe = which_ffprobe()
    if ffprobe:
        with contextlib.suppress(Exception):
            out = subprocess.run(
                [ffprobe, "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(p)],
                capture_output=True, text=True, timeout=60,
            )
            data = json.loads(out.stdout or "{}")
            for stream in data.get("streams", []):
                ctype = stream.get("codec_type")
                if ctype == "audio":
                    info["has_audio"] = True
                elif ctype == "video":
                    info["has_video"] = True
                if stream.get("codec_name"):
                    info["codecs"].append(stream["codec_name"])
            dur = data.get("format", {}).get("duration")
            if dur is not None:
                info["duration"] = float(dur)
            return info
    # Fallback: WAV only.
    if p.suffix.lower() == ".wav":
        with contextlib.suppress(Exception):
            with wave.open(str(p), "rb") as w:
                info["has_audio"] = True
                rate = w.getframerate()
                if rate:
                    info["duration"] = w.getnframes() / float(rate)
    return info


def probe_duration(path: "str | Path") -> Optional[float]:
    return probe(path).get("duration")


def has_audio_stream(path: "str | Path") -> bool:
    return probe(path).get("has_audio", False)


def demux_to_wav(src: "str | Path", dst: "str | Path", *, sample_rate: int = 16000) -> Path:
    """Normalize ``src`` to 16 kHz mono PCM WAV at ``dst`` (Stentor stage 1).

    Raises :class:`DemuxError` if ffmpeg is unavailable or conversion fails.
    """
    src, dst = Path(src), Path(dst)
    ffmpeg = which_ffmpeg()
    if not ffmpeg:
        raise DemuxError("ffmpeg is required to demux media to WAV but was not found on PATH.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise DemuxError(f"ffmpeg demux failed for {src}: {proc.stderr.strip()[:400]}")
    return dst


def extract_frame(src: "str | Path", t_seconds: float, dst: "str | Path") -> Optional[Path]:
    """Extract a single video frame at ``t_seconds`` (Tessera helper)."""
    src, dst = Path(src), Path(dst)
    ffmpeg = which_ffmpeg()
    if not ffmpeg:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-ss", f"{max(0.0, t_seconds):.3f}",
        "-i", str(src), "-frames:v", "1", "-q:v", "3", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        return None
    return dst
