"""LLM backends for claim extraction and answer synthesis.

memovox is designed so the LLM is *optional*: Assay (claim extraction) and Augur
(answer synthesis) have deterministic, extractive fallbacks that need no model.
When a real generative backend is configured it is used for higher-quality
phrasing. The default generative backend is **Ollama** (local, free, no API key)
reached over HTTP with the standard library — no extra Python dependency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from .base import LLMBackend


class OllamaLLM(LLMBackend):
    name = "ollama"
    is_generative = True

    def __init__(self, config=None, model: Optional[str] = None, host: Optional[str] = None, **options):
        super().__init__(config, **options)
        self.model = model or os.environ.get("MEMOVOX_LLM_MODEL") or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")

    @classmethod
    def is_available(cls) -> bool:
        # Ollama is reached over HTTP — availability means a REACHABLE server, not
        # just the binary on PATH. memovox never starts the daemon, so a
        # binary-present-but-server-down host must report unavailable; otherwise
        # auto-selects ollama and every generate call fails (Connection refused)
        # before degrading to the deterministic fallback.
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        return _ping(host)

    def complete(self, prompt, *, system=None, max_tokens=512, temperature=0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return (body.get("response") or "").strip()
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:  # pragma: no cover
            raise RuntimeError(f"Ollama request failed: {exc}") from exc


def _ping(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False
