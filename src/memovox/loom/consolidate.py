"""Loom cross-corpus consolidation (spec §3 stage 7).

Phase-0 implementation of contradiction/agreement detection: cluster candidate
claim pairs by shared content tokens (an inverted-index prefilter so it isn't
O(n^2)), then run pairwise NLI. Contradictions/supports are written as
provenanced graph edges and returned with deep links. With a real DeBERTa-NLI
backend this becomes precise; the lexical fallback already surfaces clear
polarity flips ("X holds" vs "X does not hold") for free.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from ..backends.base import NLIBackend
from ..util import deep_link, tokenize
from .models import Claim
from .store import LoomStore

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "as", "by", "with", "from", "we", "you", "they", "i", "not", "no",
}


@dataclass
class ContradictionPair:
    claim_a: Claim
    claim_b: Claim
    relation: str  # CONTRADICTS | SUPPORTS
    score: float
    deep_link_a: Optional[str] = None
    deep_link_b: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "relation": self.relation,
            "score": self.score,
            "a": {"claim_id": self.claim_a.claim_id, "video_id": self.claim_a.video_id,
                  "text": self.claim_a.text, "t_start_s": self.claim_a.t_start_s,
                  "deep_link": self.deep_link_a},
            "b": {"claim_id": self.claim_b.claim_id, "video_id": self.claim_b.video_id,
                  "text": self.claim_b.text, "t_start_s": self.claim_b.t_start_s,
                  "deep_link": self.deep_link_b},
        }


def _content_tokens(text: str) -> set:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 2}


def find_contradictions(
    store: LoomStore,
    *,
    nli: NLIBackend,
    topic: Optional[str] = None,
    threshold: float = 0.55,
    min_shared: int = 2,
    max_claims: int = 600,
    write_edges: bool = True,
    include_supports: bool = False,
) -> List[ContradictionPair]:
    claims = store.list_claims(status="committed")[:max_claims]
    if topic:
        topic_tokens = _content_tokens(topic)
        if topic_tokens:
            claims = [c for c in claims if _content_tokens(c.text) & topic_tokens]

    token_set = {c.claim_id: _content_tokens(c.text) for c in claims}
    by_id = {c.claim_id: c for c in claims}

    # Inverted index: token -> claim_ids, to generate only overlapping candidates.
    inverted = defaultdict(list)
    for cid, toks in token_set.items():
        for t in toks:
            inverted[t].append(cid)

    seen_pairs = set()
    video_cache = {}
    results: List[ContradictionPair] = []

    def link(claim: Claim):
        v = video_cache.get(claim.video_id)
        if v is None:
            v = store.get_video(claim.video_id)
            video_cache[claim.video_id] = v
        return deep_link(v.source_url, claim.t_start_s) if v else None

    for toks in token_set.values():
        candidates = {cid for t in toks for cid in inverted[t]}
        for cid_a in candidates:
            for cid_b in candidates:
                if cid_a >= cid_b:
                    continue
                a, b = by_id[cid_a], by_id[cid_b]
                if a.video_id == b.video_id:
                    continue  # cross-corpus only
                if len(token_set[cid_a] & token_set[cid_b]) < min_shared:
                    continue
                key = (cid_a, cid_b)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                res = nli.classify(a.text, b.text)
                if res.label == "contradiction" and res.contradict >= threshold:
                    if write_edges:
                        store.add_edge(a.claim_id, "CONTRADICTS", b.claim_id,
                                       src_type="Claim", dst_type="Claim",
                                       confidence=res.contradict)
                    results.append(ContradictionPair(a, b, "CONTRADICTS", round(res.contradict, 4),
                                                     link(a), link(b)))
                elif include_supports and res.label == "entailment" and res.entail >= threshold:
                    if write_edges:
                        store.add_edge(a.claim_id, "SUPPORTS", b.claim_id,
                                       src_type="Claim", dst_type="Claim",
                                       confidence=res.entail)
                    results.append(ContradictionPair(a, b, "SUPPORTS", round(res.entail, 4),
                                                     link(a), link(b)))

    results.sort(key=lambda p: p.score, reverse=True)
    return results
