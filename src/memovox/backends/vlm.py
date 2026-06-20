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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .base import VLMBackend

_CAPTION_PROMPT = (
    "Describe this video frame for a knowledge base. Name the setting and the "
    "salient objects, people, and actions visible in the scene, and transcribe any "
    "on-screen text: slide titles, bullet text, code, equations, charts, diagrams, "
    "UI, captions, or signage. Be dense and factual; describe only what is visible, "
    "do not speculate."
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
        # Reachable server only (see OllamaLLM.is_available) — the binary on PATH
        # does not mean the daemon is up, and memovox never starts it.
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        return _ping(host)

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
            # A non-dict JSON body (unexpected server response) must degrade to no caption,
            # not raise AttributeError and drop the whole visual track.
            return (body.get("response") or "").strip() if isinstance(body, dict) else ""
        except (urllib.error.URLError, TimeoutError, ValueError):  # pragma: no cover
            return ""


def _ping(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


class Qwen25VL(VLMBackend):
    """Named §7 default: Qwen2.5-VL dense frame captioner (local, via transformers).
    Opt-in [visual] extra; is_available-gated, falls back to NullVLM on a bare
    machine — the free path is unchanged."""

    name = "qwen2.5-vl"
    is_generative = True

    @classmethod
    def is_available(cls) -> bool:
        # Unimplemented skeleton (caption() raises NotImplementedError): report
        # UNAVAILABLE so `auto` never selects it and an explicit request fails clean
        # with BackendUnavailable at the factory instead of crashing mid-ingest.
        # Restore the find_spec(transformers)+find_spec(qwen_vl_utils) gate when the
        # Qwen2.5-VL model wiring lands.
        return False

    def caption(self, image_path, *, ocr_text=None, prompt=None) -> str:  # pragma: no cover - needs qwen
        from transformers import AutoModelForVision2Seq  # noqa: F401  (lazy; gated)

        raise NotImplementedError(
            "Qwen25VL is a named-default skeleton; wire the Qwen2.5-VL processor + "
            "generate() here once the [visual] extra is installed (free path stays NullVLM)."
        )
