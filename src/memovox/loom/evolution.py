"""First-class claim-evolution tracking (Phase 3, spec §5).

"How did the view on X change over time?" — collect an entity's or a topic's
claims and order them by source publish date, so a position's trajectory (and the
moments where it flips) is a first-class query rather than a side effect of the
temporal answer strategy.

Each :class:`EvolutionStep` annotates how its claim relates to the PREVIOUS step
(``CONTRADICTS`` / ``CORRECTS`` / ``SUPPORTS``) by reading the persisted graph
edges, and flags superseded claims. The historical record is preserved: committed
AND superseded claims appear (superseded ones never deleted — spec §2/§4.7); only
``unsupported`` claims, which never entered the graph as facts, are excluded.

Undated sources sort LAST, matching :func:`memovox.augur.answer.ask`'s temporal
ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..util import deep_link, slugify
from .consolidate import _content_tokens
from .models import STATUS_SUPERSEDED, STATUS_UNSUPPORTED, Claim

#: Relations checked, in priority order, when annotating a step against its
#: predecessor. CONTRADICTS first so a disagreement is never masked by a weaker
#: agreement edge that may also exist between the same claims.
_TRANSITION_RELS = ("CONTRADICTS", "CORRECTS", "SUPPORTS")


@dataclass
class EvolutionStep:
    claim: Claim
    video_id: str
    published_at: Optional[str]
    deep_link: Optional[str]
    relation: Optional[str] = None  # vs the previous step's claim
    superseded: bool = False

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim.claim_id,
            "video_id": self.video_id,
            "published_at": self.published_at,
            "text": self.claim.text,
            "claim_type": self.claim.claim_type,
            "relation": self.relation,
            "superseded": self.superseded,
            "deep_link": self.deep_link,
            "t_start_s": self.claim.t_start_s,
        }


def _relation(store, a_id: str, b_id: str) -> Optional[str]:
    """The first transition relation linking claim ``a_id`` and ``b_id`` (either
    direction), or ``None``."""
    for rel in _TRANSITION_RELS:
        for e in store.neighbors(a_id, rel=rel, direction="out"):
            if e["dst"] == b_id:
                return rel
        for e in store.neighbors(a_id, rel=rel, direction="in"):
            if e["src"] == b_id:
                return rel
    return None


def _entity_claims(store, entity_id: str) -> List[Claim]:
    claims = store.get_claims(store.entity_mentions(entity_id))
    return [c for c in claims if c.status != STATUS_UNSUPPORTED]


def _topic_claims(store, topic: str) -> List[Claim]:
    wanted = _content_tokens(topic)
    if not wanted:
        return []
    out = []
    for c in store.list_claims(status=None):
        if c.status == STATUS_UNSUPPORTED:
            continue
        if _content_tokens(c.text) & wanted:
            out.append(c)
    return out


def claim_evolution(
    store, *, entity_id: Optional[str] = None, topic: Optional[str] = None
) -> List[EvolutionStep]:
    """Ordered claim trajectory for an entity or a topic (spec §5).

    Supply exactly one of ``entity_id`` (an ``ent:<slug>`` id, or a surface name
    which is slugified to one) or ``topic`` (free text matched on content tokens).
    Claims are ordered by ``(undated last, published_at, t_start_s, claim_id)``.
    """
    if entity_id:
        if not entity_id.startswith("ent:"):
            entity_id = "ent:" + slugify(entity_id)
        claims = _entity_claims(store, entity_id)
    elif topic:
        claims = _topic_claims(store, topic)
    else:
        raise ValueError("claim_evolution requires entity_id or topic")

    dates: Dict[str, Optional[str]] = {}
    urls: Dict[str, Optional[str]] = {}

    def _meta(video_id: str):
        if video_id not in dates:
            v = store.get_video(video_id)
            dates[video_id] = v.published_at if v else None
            urls[video_id] = v.source_url if v else None
        return dates[video_id], urls[video_id]

    for c in claims:
        _meta(c.video_id)

    ordered = sorted(
        claims,
        key=lambda c: (dates[c.video_id] is None, dates[c.video_id] or "",
                       c.t_start_s, c.claim_id),
    )

    steps: List[EvolutionStep] = []
    prev: Optional[Claim] = None
    for c in ordered:
        pub, url = dates[c.video_id], urls[c.video_id]
        steps.append(EvolutionStep(
            claim=c,
            video_id=c.video_id,
            published_at=pub,
            deep_link=deep_link(url, c.t_start_s),
            relation=_relation(store, c.claim_id, prev.claim_id) if prev else None,
            superseded=(c.status == STATUS_SUPERSEDED),
        ))
        prev = c
    return steps
