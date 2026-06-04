"""Storage-backend interfaces for the four-index store (spec §6, §7).

Mirrors the model-backend seam in ``memovox/backends/base.py``: each storage slot
(vector / lexical / graph) is an abstract interface with an always-available,
dependency-free SQLite default, so an ANN / Tantivy / Kùzu upgrade can be dropped
in behind ``is_available()`` without rewriting callers.

These are *seams*, not new dependencies — the free path stays SQLite-only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple


class StorageBackend(ABC):
    """Common parent (mirrors ``memovox.backends.base.Backend``)."""

    name: str = "base"

    @classmethod
    def is_available(cls) -> bool:
        return True


class VectorIndex(StorageBackend):
    """Dense (vector) leg: store a moment's embedding, search by cosine."""

    @abstractmethod
    def add(self, moment_id: str, embedding: Sequence[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vec: Sequence[float],
        top_k: int = 20,
        *,
        video_id: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError


class LexicalIndex(StorageBackend):
    """Sparse (lexical) leg: index a moment's text, search by term match."""

    @abstractmethod
    def add(self, moment_id: str, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        raise NotImplementedError


class GraphStore(StorageBackend):
    """Temporal knowledge graph leg: provenanced, timestamped edges."""

    @abstractmethod
    def add_edge(
        self, src: str, rel: str, dst: str, *,
        src_type: str = "", dst_type: str = "", video_id: Optional[str] = None,
        t_start_s: float = 0.0, t_end_s: float = 0.0, modality: str = "speech",
        confidence: float = 1.0, props: Optional[dict] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node: str, *, rel: Optional[str] = None,
                  direction: str = "out") -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def edges(self, *, rel: Optional[str] = None) -> List[dict]:
        raise NotImplementedError
