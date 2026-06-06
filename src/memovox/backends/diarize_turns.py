"""Opt-in pyannote diarization-turns backend (M0.3 W7) — speaker turns.

A trailing, is_available-gated upgrade: when ``pyannote.audio`` is installed it
produces speaker turn segments (and pairs with the existing PyannoteVoiceprint for
cross-video voiceprints); otherwise it reports unavailable and the free path keeps
its caption-derived ``<v Name>`` speakers. Never imported on the free path (lazy
import inside ``diarize``; only ``find_spec`` at gate time).
"""

from __future__ import annotations

import importlib.util
from typing import List


class PyannoteTurns:
    name = "pyannote-turns"

    def __init__(self, config=None, **options) -> None:
        self.config = config
        self.options = options

    @classmethod
    def is_available(cls) -> bool:
        # Unimplemented skeleton (diarize() raises NotImplementedError): report
        # UNAVAILABLE so an explicit request fails clean at the factory rather than
        # crashing mid-pipeline. Restore the find_spec("pyannote.audio") gate when wired.
        return False

    def diarize(self, audio_path: str) -> List:  # pragma: no cover - needs pyannote.audio
        """Return ``(t_start, t_end, speaker)`` turns for ``audio_path`` (skeleton, W7)."""
        from pyannote.audio import Pipeline  # noqa: F401  (lazy; gated by is_available())

        raise NotImplementedError(
            "PyannoteTurns.diarize is a W7 skeleton; wire the pyannote speaker-diarization "
            "pipeline here once the [diarize] extra is installed."
        )
