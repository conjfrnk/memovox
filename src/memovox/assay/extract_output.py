"""X.4 — schema-targeted structured extraction OUTPUT mode (spec §5).

Distinct from ``Answer.to_dict()`` / ``Synthesis.to_dict()`` (a *generated
answer*): this emits the *extracted knowledge structure* — typed claims
(subject/predicate/object/type + timing + salience) and the resolved entity
surface forms — validated against a declared schema, deterministic on the free
(rule-based, no-LLM) path. The generative path is opt-in via the same
``is_generative`` gate ``extract_claims`` already uses.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from ..backends.base import LLMBackend
from ..loom.models import CLAIM_TYPES, Claim, Moment
from .claims import extract_claims, extract_mentions

SCHEMA_VERSION = "1.0"
_CLAIM_KEYS = {"claim_id", "text", "subject", "predicate", "object", "type",
               "t_start_s", "t_end_s", "salience"}
_DOC_KEYS = {"version", "claims", "entities"}


def _claim_doc(c: Claim) -> dict:
    return {
        "claim_id": c.claim_id, "text": c.text, "subject": c.subject,
        "predicate": c.predicate, "object": c.object, "type": c.claim_type,
        "t_start_s": c.t_start_s, "t_end_s": c.t_end_s,
        "salience": round(c.salience, 4),
    }


def validate_document(doc: dict) -> None:
    """Stdlib dict-shape validation (no jsonschema dependency)."""
    if set(doc) != _DOC_KEYS:
        raise ValueError(f"extraction doc keys {sorted(doc)} != {sorted(_DOC_KEYS)}")
    if not isinstance(doc["claims"], list) or not isinstance(doc["entities"], list):
        raise ValueError("'claims' and 'entities' must be lists")
    for c in doc["claims"]:
        if set(c) != _CLAIM_KEYS:
            raise ValueError(f"claim keys {sorted(c)} != {sorted(_CLAIM_KEYS)}")
        if c["type"] not in CLAIM_TYPES:
            raise ValueError(f"claim type {c['type']!r} not in {CLAIM_TYPES}")


def _build(claims: List[Claim]) -> dict:
    entities: set = set()
    docs = []
    for c in sorted(claims, key=lambda x: x.claim_id):  # deterministic ordering
        docs.append(_claim_doc(c))
        entities.update(extract_mentions(c))
    doc = {"version": SCHEMA_VERSION, "claims": docs, "entities": sorted(entities)}
    validate_document(doc)
    return doc


def extract_document(moment: Moment, *, llm: Optional[LLMBackend] = None,
                     min_words: int = 4) -> dict:
    """Schema-validated extraction document for a single Moment."""
    return _build(extract_claims(moment, llm=llm, min_words=min_words))


def extract_video_document(moments: Iterable[Moment], *, llm: Optional[LLMBackend] = None,
                           min_words: int = 4) -> dict:
    """Schema-validated extraction document merged across a video's Moments."""
    claims: List[Claim] = []
    for m in moments:
        claims.extend(extract_claims(m, llm=llm, min_words=min_words))
    return _build(claims)
