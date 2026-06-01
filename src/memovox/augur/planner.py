"""Augur query planner (spec §5).

A lightweight intent classifier that picks a retrieval strategy and modality.
The agentic, multi-step planner is a later upgrade; this deterministic version
already routes contradiction / temporal / procedure / visual questions sensibly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryPlan:
    strategy: str = "hybrid"     # hybrid | contradiction | temporal | procedure | visual
    modality: str = "any"        # any | speech | visual
    contradiction: bool = False
    temporal: bool = False


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
