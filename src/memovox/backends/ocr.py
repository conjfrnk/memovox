"""OCR backends for on-screen text (Tessera, spec §7).

The default :class:`NullOCR` returns no text, so the visual track degrades
gracefully with no dependency. :class:`TesseractOCR` is a free, dependency-light
upgrade: it shells out to the ``tesseract`` binary if it is on ``PATH`` (no
Python package required). Surya/PaddleOCR can slot in behind the same interface.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .base import OCRBackend


class NullOCR(OCRBackend):
    """No-op OCR — the always-available, dependency-free fallback."""

    name = "none"

    def extract(self, image_path) -> str:
        return ""


class TesseractOCR(OCRBackend):
    """Extract on-screen text via the ``tesseract`` CLI (free, no Python dep)."""

    name = "tesseract"

    def __init__(self, config=None, lang: Optional[str] = None, **options):
        super().__init__(config, **options)
        self.lang = lang or options.get("lang") or "eng"

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("tesseract") is not None

    def extract(self, image_path) -> str:
        if not image_path or not Path(image_path).exists():
            return ""
        cmd = ["tesseract", str(image_path), "stdout", "-l", self.lang, "--psm", "6"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return ""
        if proc.returncode != 0:
            return ""
        return " ".join(proc.stdout.split()).strip()
