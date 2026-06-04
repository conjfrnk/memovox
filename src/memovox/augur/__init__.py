"""Augur — agentic retrieval + grounded, cited answer synthesis (spec §3)."""

from .answer import ask
from .planner import QueryPlan, plan
from .retrieve import retrieve, rrf_fuse
from .stitch import render_clip, stitch_clips
from .synthesize import Synthesis, synthesize
from .types import Answer, Citation, Clip

__all__ = ["ask", "retrieve", "rrf_fuse", "plan", "QueryPlan", "Answer", "Citation",
           "synthesize", "Synthesis", "Clip", "stitch_clips", "render_clip"]
