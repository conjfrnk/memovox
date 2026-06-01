"""Backend interfaces and shared result types (spec §7).

ASR, embedder, NLI, and LLM each implement a thin interface so the engine never
hard-depends on a vendor and any backend is A/B-benchmarkable against another.
Every interface has a deterministic, dependency-free fallback implementation so
the whole pipeline runs for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


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
