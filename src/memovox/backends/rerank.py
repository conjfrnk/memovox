"""Reranker backends (the spec §5/§3 rerank stage).

``IdentityReranker`` is the always-available free default — it returns the fused
candidates untouched, so the rerank seam is byte-identical to today. The optional
``CrossEncoderReranker`` (a sentence-transformers CrossEncoder) re-scores each
``(query, moment_text)`` pair and stable-sorts by descending relevance; it is
is_available-gated and never runs in CI, mirroring ``TransformersNLI``.
"""

from __future__ import annotations

import importlib.util
from typing import List, Optional, Tuple

from .base import Reranker

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class IdentityReranker(Reranker):
    """The free default: no reorder (same set, same order, same scores)."""

    name = "identity"
    needs_text = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    def rerank(self, query: str, candidates: List[Tuple[str, float]], *,
               texts: Optional[dict] = None) -> List[Tuple[str, float]]:
        return list(candidates)


class CrossEncoderReranker(Reranker):
    """Opt-in cross-encoder relevance reranker (sentence-transformers)."""

    name = "cross-encoder"
    needs_text = True
    _model_cache: dict = {}

    def __init__(self, config=None, model: Optional[str] = None, **options) -> None:
        super().__init__(config, **options)
        self.model_name = options.get("model", model) or DEFAULT_CROSS_ENCODER

    @classmethod
    def is_available(cls) -> bool:
        return (importlib.util.find_spec("sentence_transformers") is not None
                and importlib.util.find_spec("torch") is not None)

    def _model(self):  # pragma: no cover - needs sentence-transformers
        cached = self._model_cache.get(self.model_name)
        if cached is not None:
            return cached
        from sentence_transformers import CrossEncoder  # type: ignore

        cache_folder = str(self.config.models_dir) if self.config is not None else None
        model = CrossEncoder(self.model_name, cache_folder=cache_folder)
        self._model_cache[self.model_name] = model
        return model

    def rerank(self, query: str, candidates: List[Tuple[str, float]], *,
               texts: Optional[dict] = None) -> List[Tuple[str, float]]:  # pragma: no cover - needs model
        texts = texts or {}
        scored = self._model().predict([(query, texts.get(mid, "")) for mid, _ in candidates])
        # stable sort by descending cross-encoder relevance (ties keep RRF order)
        order = sorted(range(len(candidates)), key=lambda i: -float(scored[i]))
        return [candidates[i] for i in order]
