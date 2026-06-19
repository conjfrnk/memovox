"""Embedding backends.

Default: a deterministic, dependency-free **hashing embedder** (signed
feature-hashing over word unigrams+bigrams, L2-normalized). It needs no model
download and gives meaningful cosine similarity for lexical overlap — enough to
drive dense retrieval for free. Optional upgrade: sentence-transformers (BGE-M3).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from typing import List

from ..util import tokenize
from .base import Embedder


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, config=None, dim: int = None, **options) -> None:
        super().__init__(config, **options)
        if dim is None:
            dim = config.settings.embed_dim if config is not None else 256
        self.dim = int(dim)

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        grams = list(toks)
        grams += [f"{toks[i]}_{toks[i + 1]}" for i in range(len(toks) - 1)]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if (digest[4] & 1) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"
    is_semantic = True
    _model_cache: dict = {}

    def __init__(self, config=None, model: str = "BAAI/bge-m3", **options) -> None:
        super().__init__(config, **options)
        self.model_name = options.get("model", model)
        self.dim = 0  # set after load

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("sentence_transformers") is not None

    def _load(self):
        cached = self._model_cache.get(self.model_name)
        if cached is not None:
            return cached
        from sentence_transformers import SentenceTransformer  # type: ignore

        cache_folder = str(self.config.models_dir) if self.config is not None else None
        model = SentenceTransformer(self.model_name, cache_folder=cache_folder)
        self._model_cache[self.model_name] = model
        return model

    def embed(self, texts: List[str]) -> List[List[float]]:
        model = self._load()
        embs = model.encode(list(texts), normalize_embeddings=True)
        result = [[float(x) for x in row] for row in embs]
        if result:
            self.dim = len(result[0])
        return result
