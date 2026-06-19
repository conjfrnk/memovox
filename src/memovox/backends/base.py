"""Backend interfaces and shared result types (spec §7).

ASR, embedder, NLI, and LLM each implement a thin interface so the engine never
hard-depends on a vendor and any backend is A/B-benchmarkable against another.
Every interface has a deterministic, dependency-free fallback implementation so
the whole pipeline runs for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------- #
# transcript value types (produced by ASR, consumed by Escapement)
# --------------------------------------------------------------------------- #


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[Word] = field(default_factory=list)
    kind: str = "speech"  # speech | event (e.g. [music], [applause])


@dataclass
class ASRResult:
    segments: List[Segment]
    language: Optional[str] = None
    duration: Optional[float] = None
    backend: str = ""


# --------------------------------------------------------------------------- #
# abstract backends
# --------------------------------------------------------------------------- #


class Backend(ABC):
    name: str = "base"

    def __init__(self, config=None, **options) -> None:
        self.config = config
        self.options = options

    @classmethod
    def is_available(cls) -> bool:
        return True


class ASRBackend(Backend):
    @abstractmethod
    def transcribe(
        self,
        audio_path: Optional[str] = None,
        *,
        captions_path: Optional[str] = None,
        language: Optional[str] = None,
    ) -> ASRResult:
        raise NotImplementedError


class Embedder(Backend):
    dim: int = 0
    #: True iff the embedding geometry is SEMANTIC (a real sentence-transformer), not the
    #: deterministic hashing free fallback. Gates the dense (topic-cluster + cosine)
    #: candidate generation in consolidation/synthesis — meaningless on hashing geometry.
    is_semantic: bool = False

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


@dataclass
class NLIResult:
    label: str  # entailment | neutral | contradiction
    entail: float
    neutral: float
    contradict: float


class NLIBackend(Backend):
    #: True iff this NLI is PRECISE (a real entailment model, e.g. DeBERTa) rather than the
    #: lexical free fallback. When True the dense candidate path drops the Jaccard prefilter
    #: and lets the model judge low-lexical-overlap pairs directly.
    is_semantic: bool = False

    @abstractmethod
    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        raise NotImplementedError

    def entailment(self, premise: str, hypothesis: str) -> float:
        return self.classify(premise, hypothesis).entail


class LLMBackend(Backend):
    #: Whether this backend actually generates text (vs. a non-generative stub).
    is_generative: bool = True

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError


class VLMBackend(Backend):
    """Vision-language captioner for keyframes (Tessera, spec §7).

    ``caption()`` produces a dense description of an on-screen frame. The
    always-available fallback returns ``""`` (no caption); real backends
    (a local Ollama vision model, hosted VLMs) are optional upgrades.
    """

    #: Whether this backend actually produces captions (vs. the empty fallback).
    is_generative: bool = True

    @abstractmethod
    def caption(
        self,
        image_path: Optional[str],
        *,
        ocr_text: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        raise NotImplementedError


class VisualEmbedder(Backend):
    """Embed a keyframe into a retrievable VISUAL vector (Tessera, spec §4.3).

    The free fallback (:class:`SignatureVisualEmbedder`) is the grayscale
    signature already produced at ingest. The interface ALLOWS multi-vector
    late-interaction embedders (ColPali/MaxSim) as an opt-in upgrade, but this
    track materializes only the single-vector signature path. Every embedded
    vector carries a ``space`` tag so it can never be cosined against a text vector.
    """

    dim: int = 0
    space: str = "visual_sig"

    @abstractmethod
    def embed_image(self, image) -> List[float]:
        raise NotImplementedError


class OCRBackend(Backend):
    """On-screen text extraction for keyframes (Tessera, spec §7).

    Reads slide text, code, and equations that exist nowhere in the audio. The
    always-available fallback returns ``""``; ``tesseract`` / Surya are optional.
    """

    @abstractmethod
    def extract(self, image_path: Optional[str]) -> str:
        raise NotImplementedError


class Reranker(Backend):
    """Reorder the RRF-fused candidate set by query↔candidate relevance (spec §5).

    Sits between fused retrieval and answer synthesis. The free fallback is
    IDENTITY (returns the candidates untouched — byte-identical to today); an
    optional cross-encoder is an is_available-gated upgrade. ``rerank`` must return
    the SAME ``(moment_id, score)`` set with no additions or removals — only a
    reorder. ``needs_text`` tells the caller whether to supply candidate texts.
    """

    #: Whether ``rerank`` needs a ``texts`` map (False for the free identity path).
    needs_text: bool = False

    @abstractmethod
    def rerank(self, query: str, candidates: List[Tuple[str, float]], *,
               texts: Optional[dict] = None) -> List[Tuple[str, float]]:
        raise NotImplementedError
