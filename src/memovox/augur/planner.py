"""Augur query planner (spec §5).

A lightweight intent classifier that picks a retrieval strategy and modality.
The agentic, multi-step planner is a later upgrade; this deterministic version
already routes contradiction / temporal / procedure / visual questions sensibly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class SubQuery:
    """One decomposed clause of a (possibly multi-part) question (spec §5)."""

    text: str
    strategy: str = "hybrid"
    modality: str = "any"
    contradiction: bool = False
    temporal: bool = False

    def to_dict(self) -> dict:
        return {"text": self.text, "strategy": self.strategy, "modality": self.modality}


@dataclass
class QueryPlan:
    strategy: str = "hybrid"     # hybrid | contradiction | temporal | procedure | visual
    modality: str = "any"        # any | speech | visual
    contradiction: bool = False
    temporal: bool = False
    subqueries: List[SubQuery] = field(default_factory=list)


_CONTRADICTION = ("contradict", "disagree", "conflict", "inconsistent", "dispute")
_TEMPORAL = ("over time", "change", "evolve", "evolution", "history", "revised",
             "used to", "originally", "no longer", "these days")
_PROCEDURE = ("how do i", "how to", "steps", "step by step", "install", "set up",
              "configure", "demo", "walkthrough")
_VISUAL = ("slide", "diagram", "chart", "graph", "figure", "on screen", "shown",
           "screenshot", "code on")


def plan(query: str) -> QueryPlan:
    q = (query or "").lower()
    if any(w in q for w in _CONTRADICTION):
        return QueryPlan(strategy="contradiction", contradiction=True)
    if any(w in q for w in _TEMPORAL):
        return QueryPlan(strategy="temporal", temporal=True)
    if any(w in q for w in _PROCEDURE):
        return QueryPlan(strategy="procedure")
    if any(w in q for w in _VISUAL):
        return QueryPlan(strategy="visual", modality="visual")
    return QueryPlan(strategy="hybrid")


# Conservative clause boundaries (deterministic): a comma+"and"/"or" coordination,
# a semicolon, or a sentence-final "?"/"." followed by a new clause. NOT a bare
# " and " (which joins noun-phrase lists like "scaling laws and compute").
_BOUNDARY = re.compile(r",\s+(?:and|or)\s+|;\s*|(?<=[?.])\s+(?=\w)")
_WH = ("what", "which", "who", "whom", "whose", "when", "where", "why", "how")


def _classify(text: str) -> SubQuery:
    p = plan(text)
    return SubQuery(text=text, strategy=p.strategy, modality=p.modality,
                    contradiction=p.contradiction, temporal=p.temporal)


def _looks_like_query(fragment: str) -> bool:
    toks = fragment.lower().split()
    return (any(w in toks for w in _WH)
            or len([t for t in toks if len(t) > 3]) >= 2)


def decompose(query: str) -> QueryPlan:
    """Decompose a (possibly multi-part) question into ordered sub-queries (spec §5).

    Deterministic and conservative: splits ONLY on clear coordinating boundaries
    where EVERY resulting fragment independently looks like a query. A single-clause
    question yields exactly one sub-query equal to the verbatim input — so ``ask``
    degrades to byte-identical today's behavior.
    """
    q = (query or "").strip()
    fragments = [f.strip().strip(" ?.,") for f in _BOUNDARY.split(q)]
    fragments = [f for f in fragments if f]
    if len(fragments) >= 2 and all(_looks_like_query(f) for f in fragments):
        subs = [_classify(f) for f in fragments]
    else:
        subs = [_classify(q)]
    first = subs[0]
    return QueryPlan(strategy=first.strategy, modality=first.modality,
                     contradiction=first.contradiction, temporal=first.temporal,
                     subqueries=subs)


_DECOMPOSE_SYSTEM = (
    "Split the user's question into its minimal independent sub-questions. Return "
    "ONLY a JSON array of strings (one per sub-question); a single-part question "
    "returns a 1-element array with the question verbatim."
)


def llm_decompose(llm, query: str) -> QueryPlan:
    """LLM query decomposer (opt-in, spec §5) with a guaranteed DETERMINISTIC
    fallback: any transport/parse error or empty result returns ``decompose(query)``
    so the planner can never fail or become non-deterministic on the free path."""
    import json

    try:
        raw = llm.complete(f"QUESTION: {query}", system=_DECOMPOSE_SYSTEM, temperature=0.0)
        start, end = raw.find("["), raw.rfind("]")
        parts = json.loads(raw[start:end + 1]) if 0 <= start < end else []
        parts = [str(p).strip() for p in parts if str(p).strip()]
        if not parts:
            return decompose(query)
        subs = [_classify(p) for p in parts]
        first = subs[0]
        return QueryPlan(strategy=first.strategy, modality=first.modality,
                         contradiction=first.contradiction, temporal=first.temporal,
                         subqueries=subs)
    except Exception:
        return decompose(query)
