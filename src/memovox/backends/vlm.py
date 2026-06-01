"""Vision-language captioning backends (Tessera, spec §7).

memovox treats the VLM as *optional*: the default :class:`NullVLM` returns no
caption, so the visual track still works for free (keyframe selection + OCR +
visual embedding). When a local vision model is available it produces dense
on-screen descriptions. The default real backend is an **Ollama** vision model
(local, free, no API key), reached over HTTP with the standard library.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .base import VLMBackend

_CAPTION_PROMPT = (
    "Describe this video frame for a knowledge base: any slide title, bullet "
    "text, code, equations, charts, diagrams, or UI shown. Be dense and factual."
)


class NullVLM(VLMBackend):
    """No-op captioner — the always-available, dependency-free fallback."""

    name = "none"
    is_generative = False

    def caption(self, image_path, *, ocr_text=None, prompt=None) -> str:
        return ""


class OllamaVLM(VLMBackend):
    """Caption frames with a local Ollama vision model (e.g. ``llama3.2-vision``)."""

    name = "ollama"
    is_generative = True

    def __init__(self, config=None, model: Optional[str] = None, host: Optional[str] = None, **options):
        super().__init__(config, **options)
        self.model = (
            model
            or os.environ.get("MEMOVOX_VLM_MODEL")
            or os.environ.get("OLLAMA_VLM_MODEL", "llama3.2-vision")
        )
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")

    @classmethod
    def is_available(cls) -> bool:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        return bool(shutil.which("ollama")) or _ping(host)

    def caption(self, image_path, *, ocr_text=None, prompt=None) -> str:
        if not image_path or not Path(image_path).exists():
            return ""
        try:
            img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        except OSError:
            return ""
        instruction = prompt or _CAPTION_PROMPT
        if ocr_text:
            instruction += f"\nDetected on-screen text: {ocr_text}"
        payload = {
            "model": self.model,
            "prompt": instruction,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return (body.get("response") or "").strip()
        except (urllib.error.URLError, TimeoutError, ValueError):  # pragma: no cover
            return ""


def _ping(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False
