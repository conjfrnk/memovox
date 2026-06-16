"""Assay — claim extraction + verification + typing (spec §3 stage 5)."""

from __future__ import annotations

from typing import List, Optional

from ..backends.base import LLMBackend, NLIBackend
from ..config import Settings
from ..loom.models import STATUS_COMMITTED, STATUS_UNSUPPORTED, Claim, Moment
from .claims import (
    epistemic_type,
    extract_claims,
    is_low_value_claim,
    salience_score,
    transcript_is_punctuated,
)
from .spans import is_contiguous_in, premise_covers, span_text
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
    # W5.2: fragment detection keys off leading case, which is only valid when the
    # transcript is actually punctuated (manual captions). Computed once per Moment.
    punctuated = transcript_is_punctuated(moment.transcript)
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
        # Rolling-caption sentences are split across two ~3s cues, so locate_span's
        # single best cue can omit the claim's own tail — which a strict NLI model
        # (DeBERTa) then rejects at ~0 entailment even though the claim is verbatim
        # in the Moment. When the located premise doesn't cover the claim's content
        # AND the claim is a CONTIGUOUS run of the Moment's words (a genuine split
        # sentence — NOT a recombination of tokens scattered across spans), widen
        # the premise to the whole Moment. This recovers the split sentence while
        # still rejecting recombination hallucinations (their tokens never form a
        # contiguous run) and true hallucinations (tokens nowhere in the Moment).
        # The displayed citation window stays narrow, so the provenance invariant
        # (window ⊆ verified premise) holds.
        if not premise_covers(premise, claim.text) and is_contiguous_in(claim.text, moment.transcript):
            premise = moment.transcript
        verify_claim(nli, claim, premise, threshold=settings.entailment_threshold)
        # W5.2: demote low-value claims (greetings, ad reads, navigational
        # imperatives, and continuation fragments split across a Moment boundary).
        # Conservative + high-precision; retained as unsupported, never dropped.
        if claim.status == STATUS_COMMITTED and is_low_value_claim(claim.text, punctuated=punctuated):
            claim.status = STATUS_UNSUPPORTED
        # Salience floor (spec §5): salience drives retrieval priority + summary
        # inclusion, so a low-salience claim is not worth committing to the
        # trusted layer even when entailed. Default floor 0.0 demotes nothing.
        if (settings.salience_floor > 0.0 and claim.status == STATUS_COMMITTED
                and claim.salience < settings.salience_floor):
            claim.status = STATUS_UNSUPPORTED
    return claims


__all__ = ["run", "extract_claims", "verify_claim", "epistemic_type", "salience_score"]
