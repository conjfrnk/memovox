"""Assay verification gate — the anti-hallucination guarantee (spec §5).

Every extracted claim is entailment-checked against its source span *before* it
is committed. Claims not entailed are marked ``unsupported`` and never enter the
graph as facts.
"""

from __future__ import annotations

from ..backends.base import NLIBackend
from ..loom.models import STATUS_COMMITTED, STATUS_UNSUPPORTED, Claim


def verify_claim(nli: NLIBackend, claim: Claim, source_text: str, *, threshold: float) -> Claim:
    """Set ``entailment_score`` and ``status`` for one claim against its source.

    ``source_text`` is the premise the claim is entailment-checked against. The
    caller (``assay.run``) passes the text of the claim's OWN located span (W1.2)
    so the gate has teeth: a claim whose content tokens are absent from its span
    scores low containment and is marked ``unsupported`` rather than committed.
    """
    result = nli.classify(source_text, claim.text)
    claim.entailment_score = round(result.entail, 4)
    if result.label == "contradiction":
        claim.status = STATUS_UNSUPPORTED
    elif result.entail >= threshold:
        claim.status = STATUS_COMMITTED
    else:
        claim.status = STATUS_UNSUPPORTED
    return claim
