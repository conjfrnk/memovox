"""Backend registries and auto-selection.

``get_*("auto", config)`` picks the best installed backend and otherwise falls
back to the always-available, dependency-free implementation.
"""

from __future__ import annotations

import sys
from typing import Optional

from ..errors import BackendUnavailable

#: Optional backends whose MODEL failed to load this process (offline / uncached /
#: OOM / corrupt). Memoized so ``auto`` selection neither retries the doomed load
#: nor re-warns — it degrades straight to the free fallback. Cleared only on restart.
_DEGRADED: set = set()


def _try_optional(opt_cls, probe, *, config, options):
    """For the ``auto`` path only: construct + PROBE the optional backend (the probe
    must force the lazy model load). Return the instance, or ``None`` if it can't
    load — degrading to the free fallback is the caller's job. The decision is
    memoized in ``_DEGRADED`` and a one-time warning goes to STDERR (never stdout —
    MCP owns stdout). Explicit (non-auto) selection never calls this, so opting into
    an optional backend still fails loud."""
    if opt_cls.name in _DEGRADED or not opt_cls.is_available():
        return None
    try:
        inst = opt_cls(config=config, **options)
        probe(inst)  # forces the model load; raises offline / uncached / OOM
        return inst
    except Exception as exc:  # noqa: BLE001 - ANY load/runtime failure -> free fallback
        _DEGRADED.add(opt_cls.name)
        print(f"memovox: optional backend {opt_cls.name!r} could not load "
              f"({type(exc).__name__}: {str(exc)[:120]}); using the free fallback.",
              file=sys.stderr)
        return None
from .base import (
    ASRBackend,
    ASRResult,
    Backend,
    Embedder,
    LLMBackend,
    NLIBackend,
    NLIResult,
    OCRBackend,
    Reranker,
    Segment,
    VisualEmbedder,
    VLMBackend,
    Word,
)
from .diarize_voiceprint import PyannoteVoiceprint
from .embed import HashingEmbedder, SentenceTransformerEmbedder
from .entity_link import Canonical, EntityLinker, NullLinker, WikidataLinker
from .llm import OllamaLLM
from .nli import LexicalNLI, TransformersNLI
from .ocr import NullOCR, SuryaOCR, TesseractOCR
from .rerank import CrossEncoderReranker, IdentityReranker
from .visual_embed import ColPaliVisualEmbedder, SignatureVisualEmbedder
from .vlm import NullVLM, OllamaVLM, Qwen25VL

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

_VLMS = {"none": NullVLM, "ollama": OllamaVLM, "qwen2.5-vl": Qwen25VL}
_OCRS = {"none": NullOCR, "tesseract": TesseractOCR, "surya": SuryaOCR}
_VISUAL_EMBEDDERS = {"signature": SignatureVisualEmbedder, "colpali": ColPaliVisualEmbedder}
_RERANKERS = {"identity": IdentityReranker, "cross-encoder": CrossEncoderReranker}
_RERANK_ALIASES = {"ce": "cross-encoder", "none": "identity", "": "identity"}
_OCR_ALIASES = {}  # M1.1: the surya->tesseract placeholder is replaced by a real SuryaOCR

_LINKERS = {"none": NullLinker, "wikidata": WikidataLinker}


def get_embedder(name: str = "auto", *, config=None, **options) -> Embedder:
    if name == "auto":
        inst = _try_optional(SentenceTransformerEmbedder, lambda e: e.embed_one("ok"),
                             config=config, options=options)
        return inst if inst is not None else HashingEmbedder(config=config, **options)
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
        inst = _try_optional(TransformersNLI, lambda n: n.classify("ok", "ok"),
                             config=config, options=options)
        return inst if inst is not None else LexicalNLI(config=config, **options)
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


def get_reranker(name: str = "auto", *, config=None, **options) -> Reranker:
    """Return a reranker; ``auto`` is the free IdentityReranker unless a cross-encoder
    is installed. ``none``/``""``/``identity`` -> IdentityReranker (spec §5)."""
    if name == "auto":
        inst = _try_optional(
            CrossEncoderReranker,
            lambda r: r.rerank("ok", [("m", 1.0)], texts={"m": "ok"}),
            config=config, options=options)
        return inst if inst is not None else IdentityReranker(config=config, **options)
    name = _RERANK_ALIASES.get(name, name)
    cls = _RERANKERS.get(name)
    if cls is None:
        raise BackendUnavailable(
            f"Unknown reranker {name!r}. Options: {list(_RERANKERS)}, 'auto', or 'none'."
        )
    if not cls.is_available():
        raise BackendUnavailable(
            f"Reranker {name!r} is not installed (try the [rerank] extra)."
        )
    return cls(config=config, **options)


def get_visual_embedder(name: str = "auto", *, config=None, **options) -> VisualEmbedder:
    """Return a visual embedder; ``auto`` is the free signature embedder (M1.1)."""
    if name in ("auto", "signature", ""):
        return SignatureVisualEmbedder(config=config, **options)
    cls = _VISUAL_EMBEDDERS.get(name)
    if cls is None:
        raise BackendUnavailable(
            f"Unknown visual embedder {name!r}. Options: {list(_VISUAL_EMBEDDERS)} or 'auto'."
        )
    if not cls.is_available():
        raise BackendUnavailable(
            f"Visual embedder {name!r} is not installed (try the [visual] extra)."
        )
    return cls(config=config, **options)


def get_voiceprint_backend(
    name: str = "auto", *, config=None, **options
) -> Optional[PyannoteVoiceprint]:
    """Return an OPTIONAL voiceprint backend, or ``None`` if none is available/desired.

    Mirrors :func:`get_llm`: ``"none"``/``""`` always yields ``None`` (the free
    path), and ``"auto"`` returns a :class:`PyannoteVoiceprint` only when
    ``pyannote.audio`` is installed, else ``None``. A ``None`` return means the
    voice-based speaker merge is skipped entirely — the free name-based path
    (W4.1) is unaffected.
    """
    if name in ("none", "off", "false", ""):
        return None
    if name == "auto":
        return PyannoteVoiceprint(config=config, **options) if PyannoteVoiceprint.is_available() else None
    if name in ("pyannote", "voiceprint"):
        if not PyannoteVoiceprint.is_available():
            raise BackendUnavailable(
                "Voiceprint backend 'pyannote' is not installed. "
                "Try: pip install 'memovox[diarize]'"
            )
        return PyannoteVoiceprint(config=config, **options)
    raise BackendUnavailable(
        f"Unknown voiceprint backend {name!r}. Options: 'pyannote', 'auto', or 'none'."
    )


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
            "qwen2.5-vl": Qwen25VL.is_available(),
            "none": True,
        },
        "ocr": {
            "tesseract": TesseractOCR.is_available(),
            "surya": SuryaOCR.is_available(),
            "none": True,
        },
        "visual_embed": {
            "signature": True,
            "colpali": ColPaliVisualEmbedder.is_available(),
        },
        "rerank": {
            "identity": True,
            "cross-encoder": CrossEncoderReranker.is_available(),
        },
        "entity_link": {
            "wikidata": WikidataLinker.is_available(),
            "none": True,
        },
        "voiceprint": {
            "pyannote": PyannoteVoiceprint.is_available(),
            "none": True,
        },
    }


__all__ = [
    "Backend", "ASRBackend", "ASRResult", "Segment", "Word",
    "Embedder", "NLIBackend", "NLIResult", "LLMBackend", "VLMBackend", "OCRBackend",
    "VisualEmbedder", "Reranker",
    "EntityLinker", "Canonical",
    "HashingEmbedder", "SentenceTransformerEmbedder",
    "LexicalNLI", "TransformersNLI", "OllamaLLM",
    "NullVLM", "OllamaVLM", "NullOCR", "TesseractOCR",
    "IdentityReranker", "CrossEncoderReranker",
    "NullLinker", "WikidataLinker", "PyannoteVoiceprint",
    "get_embedder", "get_nli", "get_llm", "get_vlm", "get_ocr",
    "get_visual_embedder", "get_reranker",
    "get_entity_linker", "get_voiceprint_backend", "backend_status",
]
