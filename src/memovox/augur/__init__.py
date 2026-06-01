"""Augur — agentic retrieval + grounded, cited answer synthesis (spec §3)."""

from .answer import ask
from .planner import QueryPlan, plan
from .retrieve import retrieve, rrf_fuse
from .types import Answer, Citation

__all__ = ["ask", "retrieve", "rrf_fuse", "plan", "QueryPlan", "Answer", "Citation"]
