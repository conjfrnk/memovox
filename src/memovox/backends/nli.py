"""Natural-language-inference backends (Assay's verification gate, spec §5).

Default: a dependency-free **lexical entailment** estimator — measures how much
of the hypothesis's content is contained in the premise, with a negation-flip
heuristic for contradiction. Because Assay extracts claims *from* their source
span, genuine claims score high containment and pass the gate, while
hallucinated content would not. Optional upgrade: a transformers DeBERTa-NLI model.
"""

from __future__ import annotations

import importlib.util
from typing import List

from ..util import tokenize
from .base import NLIBackend, NLIResult

_NEGATIONS = {
    "not", "no", "never", "cannot", "cant", "wont", "dont", "doesnt", "didnt",
    "isnt", "arent", "wasnt", "werent", "none", "neither", "nor", "without",
    "n't", "false", "incorrect",
}
# Very small stopword set so containment focuses on content words.
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "as", "by", "with", "from", "we", "you", "they", "he", "she", "i",
}


def _content(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _STOP and len(t) > 1]


class LexicalNLI(NLIBackend):
    name = "lexical"

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        p_tokens = tokenize(premise)
        h_tokens = tokenize(hypothesis)
        p_set = set(p_tokens)
        h_content = _content(h_tokens)
        if not h_content:
            return NLIResult("neutral", 0.34, 0.33, 0.33)

        overlap = sum(1 for w in h_content if w in p_set) / len(h_content)
        p_neg = bool(p_set & _NEGATIONS)
        h_neg = bool(set(h_tokens) & _NEGATIONS)

        contradict = 0.0
        if overlap >= 0.5 and (p_neg != h_neg):
            contradict = 0.5 * overlap
        entail = max(0.0, overlap - contradict)
        neutral = max(0.0, 1.0 - entail - contradict)

        if contradict > entail and contradict > neutral:
            label = "contradiction"
        elif entail >= 0.5 and entail >= neutral:
            label = "entailment"
        else:
            label = "neutral"
        return NLIResult(label, round(entail, 4), round(neutral, 4), round(contradict, 4))


class TransformersNLI(NLIBackend):
    name = "deberta-nli"
    _pipe_cache: dict = {}
    DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def __init__(self, config=None, model: str = None, **options) -> None:
        super().__init__(config, **options)
        self.model_name = options.get("model", model) or self.DEFAULT_MODEL

    @classmethod
    def is_available(cls) -> bool:
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def _pipe(self):
        cached = self._pipe_cache.get(self.model_name)
        if cached is not None:
            return cached
        from transformers import pipeline  # type: ignore

        pipe = pipeline("text-classification", model=self.model_name, top_k=None)
        self._pipe_cache[self.model_name] = pipe
        return pipe

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        pipe = self._pipe()
        scores = pipe({"text": premise, "text_pair": hypothesis})
        # Normalize the various label spellings MNLI models emit.
        mapping = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        rows = scores[0] if scores and isinstance(scores[0], list) else scores
        for row in rows:
            label = str(row["label"]).lower()
            for key in mapping:
                if key.startswith(label[:4]):
                    mapping[key] = float(row["score"])
        best = max(mapping, key=mapping.get)
        return NLIResult(best, mapping["entailment"], mapping["neutral"], mapping["contradiction"])
