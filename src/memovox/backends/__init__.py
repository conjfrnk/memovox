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
    OCRBackend,
    Segment,
    VLMBackend,
    Word,
)
from .embed import HashingEmbedder, SentenceTransformerEmbedder
from .entity_link import Canonical, EntityLinker, NullLinker, WikidataLinker
from .llm import OllamaLLM
from .nli import LexicalNLI, TransformersNLI
from .ocr import NullOCR, TesseractOCR
from .vlm import NullVLM, OllamaVLM

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

_VLMS = {"none": NullVLM, "ollama": OllamaVLM}
_OCRS = {"none": NullOCR, "tesseract": TesseractOCR}
_OCR_ALIASES = {"surya": "tesseract"}  # graceful: Surya not yet wired, use tesseract

_LINKERS = {"none": NullLinker, "wikidata": WikidataLinker}


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


def get_vlm(name: str = "auto", *, config=None, **options) -> VLMBackend:
    """Return a vision-language captioner; falls back to the no-op NullVLM."""
    if name in ("none", "off", "false", ""):
        return NullVLM(config=config, **options)
    if name == "auto":
        return OllamaVLM(config=config, **options) if OllamaVLM.is_available() else NullVLM(config=config)
    cls = _VLMS.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown VLM backend {name!r}. Options: {list(_VLMS)}, 'auto', or 'none'.")
    if not cls.is_available():
        raise BackendUnavailable(f"VLM backend {name!r} is not reachable (is the local vision model running?).")
    return cls(config=config, **options)


def get_ocr(name: str = "auto", *, config=None, **options) -> OCRBackend:
    """Return an OCR backend; falls back to the no-op NullOCR."""
    if name in ("none", "off", "false", ""):
        return NullOCR(config=config, **options)
    if name == "auto":
        return TesseractOCR(config=config, **options) if TesseractOCR.is_available() else NullOCR(config=config)
    name = _OCR_ALIASES.get(name, name)
    cls = _OCRS.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown OCR backend {name!r}. Options: {list(_OCRS)}, 'auto', or 'none'.")
    if not cls.is_available():
        raise BackendUnavailable(
            f"OCR backend {name!r} is not installed. Install the `tesseract` binary."
        )
    return cls(config=config, **options)


def get_entity_linker(name: str = "auto", *, config=None, **options) -> EntityLinker:
    """Return an entity linker; falls back to the slug-based NullLinker."""
    if name in ("none", "off", "false", ""):
        return NullLinker(config=config, **options)
    if name == "auto":
        return WikidataLinker(config=config, **options) if WikidataLinker.is_available() else NullLinker(config=config)
    cls = _LINKERS.get(name)
    if cls is None:
        raise BackendUnavailable(f"Unknown entity linker {name!r}. Options: {list(_LINKERS)}, 'auto', or 'none'.")
    if not cls.is_available():
        raise BackendUnavailable(f"Entity linker {name!r} is offline (no network to Wikidata).")
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
        "vlm": {
            "ollama": OllamaVLM.is_available(),
            "none": True,
        },
        "ocr": {
            "tesseract": TesseractOCR.is_available(),
            "none": True,
        },
        "entity_link": {
            "wikidata": WikidataLinker.is_available(),
            "none": True,
        },
    }


__all__ = [
    "Backend", "ASRBackend", "ASRResult", "Segment", "Word",
    "Embedder", "NLIBackend", "NLIResult", "LLMBackend", "VLMBackend", "OCRBackend",
    "EntityLinker", "Canonical",
    "HashingEmbedder", "SentenceTransformerEmbedder",
    "LexicalNLI", "TransformersNLI", "OllamaLLM",
    "NullVLM", "OllamaVLM", "NullOCR", "TesseractOCR",
    "NullLinker", "WikidataLinker",
    "get_embedder", "get_nli", "get_llm", "get_vlm", "get_ocr",
    "get_entity_linker", "backend_status",
]
