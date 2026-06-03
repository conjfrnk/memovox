"""Assay — claim extraction + verification + typing (spec §3 stage 5)."""

from __future__ import annotations

from typing import List, Optional

from ..backends.base import LLMBackend, NLIBackend
from ..config import Settings
from ..loom.models import Claim, Moment
from .claims import epistemic_type, extract_claims, salience_score
from .spans import span_text
from .verify import verify_claim


def run(
    moment: Moment,
    *,
    nli: NLIBackend,
    llm: Optional[LLMBackend] = None,
    settings: Optional[Settings] = None,
) -> List[Claim]:
    """Extract, type, score, and verify the claims in a Moment.

    Returns all claims (both ``committed`` and ``unsupported``); the caller
    decides what to persist. By design, unsupported claims are retained but
    flagged, never silently dropped.
    """
    settings = settings or Settings()
    claims = extract_claims(moment, llm=llm)
    for claim in claims:
        # Verify each claim against the source text of its OWN located span
        # (W1.2), not the whole Moment — so a hallucinated claim whose tokens
        # appear nowhere in its span is rejected. Segment-less Moments (old
        # fixtures, store-reloaded Moments) fall back to the whole-Moment text,
        # preserving legacy behaviour.
        premise = span_text(moment.segments, claim.t_start_s, claim.t_end_s) or moment.text_for_embedding()
        verify_claim(nli, claim, premise, threshold=settings.entailment_threshold)
    return claims


__all__ = ["run", "extract_claims", "verify_claim", "epistemic_type", "salience_score"]
