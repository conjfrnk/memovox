"""Optional ANN vector backends (M0.2 W7) — opt-in, NEVER in the CI gate.

Skeletons behind ``is_available()``: ``get_vector_index("lance"|"qdrant")`` raises
``BackendUnavailable`` (never crashes, never silently falls into the gate) when the
dependency is absent — exactly the model-backend ``auto → free fallback`` idiom.
They are measured only by ``eval/scale.py`` (synthetic-N recall@k vs the exact
free index), never by ``make test`` or ``--assert-thresholds``.
"""

from __future__ import annotations

import importlib.util
from typing import List, Optional, Sequence, Tuple

from .base import VectorIndex


class LanceVectorIndex(VectorIndex):
    """LanceDB-backed approximate vector index (embedded, on-disk)."""

    name = "lance"

    def __init__(self, conn=None, **options) -> None:
        self.conn = conn
        self.options = options

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("lancedb") is not None

    def add(self, moment_id: str, embedding: Sequence[float]) -> None:  # pragma: no cover - needs lancedb
        raise NotImplementedError(
            "LanceVectorIndex is a W7 skeleton; wire the lancedb table here once "
            "the [vectors] extra is installed (measured via eval/scale.py)."
        )

    def search(self, query_vec: Sequence[float], top_k: int = 20, *,
               video_id: Optional[str] = None, query_text: Optional[str] = None,
               space: Optional[str] = None) -> List[Tuple[str, float]]:  # pragma: no cover
        raise NotImplementedError("LanceVectorIndex.search is a W7 skeleton.")


class QdrantVectorIndex(VectorIndex):
    """Qdrant-backed approximate vector index (local or remote server)."""

    name = "qdrant"

    def __init__(self, conn=None, **options) -> None:
        self.conn = conn
        self.options = options

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("qdrant_client") is not None

    def add(self, moment_id: str, embedding: Sequence[float]) -> None:  # pragma: no cover - needs qdrant
        raise NotImplementedError(
            "QdrantVectorIndex is a W7 skeleton; wire the qdrant collection here once "
            "qdrant-client is installed (measured via eval/scale.py)."
        )

    def search(self, query_vec: Sequence[float], top_k: int = 20, *,
               video_id: Optional[str] = None, query_text: Optional[str] = None,
               space: Optional[str] = None) -> List[Tuple[str, float]]:  # pragma: no cover
        raise NotImplementedError("QdrantVectorIndex.search is a W7 skeleton.")
