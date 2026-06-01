"""Backend registries and auto-selection.

``get_*("auto", config)`` picks the best installed backend and otherwise falls
back to the always-available, dependency-free implementation.
"""

from __future__ import annotations

from typing import Optional

from ..errors import BackendUnavailable
from .base import (
    ASRBackend,
    ASRResult,
    Backend,
    Embedder,
    LLMBackend,
    NLIBackend,
    NLIResult,
    Segment,
    Word,
)
from .embed import HashingEmbedder, SentenceTransformerEmbedder
from .llm import OllamaLLM
from .nli import LexicalNLI, TransformersNLI

_EMBEDDERS = {
    "hashing": HashingEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}
_EMBED_ALIASES = {"st": "sentence-transformers", "bge": "sentence-transformers", "bge-m3": "sentence-transformers"}

_NLI = {
    "lexical": LexicalNLI,
    "deberta-nli": TransformersNLI,
}
_NLI_ALIASES = {"transformers": "deberta-nli", "deberta": "deberta-nli", "nli": "deberta-nli"}

_LLMS = {"ollama": OllamaLLM}


def get_embedder(name: str = "auto", *, config=None, **options) -> Embedder:
    if name == "auto":
        name = "sentence-transformers" if SentenceTransformerEmbedder.is_available() else "hashing"
    name = _EMBED_ALIASES.get(name, name)
    cls = _EMBEDDERS.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown embedder backend {name!r}. Options: {list(_EMBEDDERS)} or 'auto'.")
    if not cls.is_available():
        raise BackendUnavailable(
            f"Embedder {name!r} is not installed. Try: pip install 'memovox[embed]'"
        )
    return cls(config=config, **options)


def get_nli(name: str = "auto", *, config=None, **options) -> NLIBackend:
    if name == "auto":
        name = "deberta-nli" if TransformersNLI.is_available() else "lexical"
    name = _NLI_ALIASES.get(name, name)
    cls = _NLI.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown NLI backend {name!r}. Options: {list(_NLI)} or 'auto'.")
    if not cls.is_available():
        raise BackendUnavailable(
            f"NLI backend {name!r} is not installed. Try: pip install 'memovox[nli]'"
        )
    return cls(config=config, **options)


def get_llm(name: str = "auto", *, config=None, **options) -> Optional[LLMBackend]:
    """Return a generative LLM backend, or ``None`` if none is available/desired."""
    if name in ("none", "off", "false", ""):
        return None
    if name == "auto":
        return OllamaLLM(config=config, **options) if OllamaLLM.is_available() else None
    cls = _LLMS.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown LLM backend {name!r}. Options: {list(_LLMS)}, 'auto', or 'none'.")
    if not cls.is_available():
        raise BackendUnavailable(f"LLM backend {name!r} is not reachable (is the local server running?).")
    return cls(config=config, **options)


def backend_status() -> dict:
    """Snapshot of which backends are available (for `memovox backends`)."""
    from .asr_whisper import WhisperASR

    return {
        "asr": {
            "whisper": WhisperASR.is_available(),
            "captions": True,
            "fake": True,
        },
        "embed": {
            "sentence-transformers": SentenceTransformerEmbedder.is_available(),
            "hashing": True,
        },
        "nli": {
            "deberta-nli": TransformersNLI.is_available(),
            "lexical": True,
        },
        "llm": {
            "ollama": OllamaLLM.is_available(),
        },
    }


__all__ = [
    "Backend", "ASRBackend", "ASRResult", "Segment", "Word",
    "Embedder", "NLIBackend", "NLIResult", "LLMBackend",
    "HashingEmbedder", "SentenceTransformerEmbedder",
    "LexicalNLI", "TransformersNLI", "OllamaLLM",
    "get_embedder", "get_nli", "get_llm", "backend_status",
]
