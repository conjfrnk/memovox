"""Assay — claim extraction + verification + typing (spec §3 stage 5)."""

from __future__ import annotations

from typing import List, Optional

from ..backends.base import LLMBackend, NLIBackend
from ..config import Settings
from ..loom.models import STATUS_COMMITTED, STATUS_UNSUPPORTED, Claim, Moment
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
        # PROVENANCE INVARIANT (M0.3 W3): the premise stays SEGMENT-granular even
        # when locate_span narrowed the claim's citation window to a word span —
        # span_text selects by overlap, so a narrowed window still pulls its whole
        # parent cue. The displayed citation window is thus always ⊆ the text NLI
        # verified; it can never drift narrower than the premise. Pinned in
        # tests/test_span_premise_invariant.py.
        premise = span_text(moment.segments, claim.t_start_s, claim.t_end_s) or moment.text_for_embedding()
        verify_claim(nli, claim, premise, threshold=settings.entailment_threshold)
        # Salience floor (spec §5): salience drives retrieval priority + summary
        # inclusion, so a low-salience claim is not worth committing to the
        # trusted layer even when entailed. Default floor 0.0 demotes nothing.
        if (settings.salience_floor > 0.0 and claim.status == STATUS_COMMITTED
                and claim.salience < settings.salience_floor):
            claim.status = STATUS_UNSUPPORTED
    return claims


__all__ = ["run", "extract_claims", "verify_claim", "epistemic_type", "salience_score"]
