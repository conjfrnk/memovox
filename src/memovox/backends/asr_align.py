"""Opt-in WhisperX forced-alignment (M0.3 W7) — refine word boundaries.

A trailing, is_available-gated upgrade: when ``whisperx`` is installed it refines
the per-word timestamps WhisperASR emits (sharper citation windows); otherwise it
reports unavailable and the pipeline keeps the free word boundaries. The free path
never imports whisperx (lazy import inside ``align``; only ``find_spec`` at gate
time), mirroring ``diarize_voiceprint.py``.
"""

from __future__ import annotations

import importlib.util
from typing import List, Optional


class WhisperXAlign:
    name = "whisperx"

    def __init__(self, config=None, **options) -> None:
        self.config = config
        self.options = options

    @classmethod
    def is_available(cls) -> bool:
        try:
            return importlib.util.find_spec("whisperx") is not None
        except (ImportError, ValueError):
            return False

    def align(self, segments: List, audio_path: str, *, language: Optional[str] = None):  # pragma: no cover - needs whisperx
        """Return ``segments`` with forced-aligned word timings (skeleton, W7)."""
        import whisperx  # noqa: F401  (lazy; only reached when is_available())

        raise NotImplementedError(
            "WhisperXAlign.align is a W7 skeleton; wire whisperx.load_align_model + "
            "whisperx.align here once the [align] extra is installed."
        )
