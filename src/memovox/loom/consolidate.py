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
from ..config import Settings
from ..util import deep_link, tokenize
from .models import STATUS_COMMITTED, Claim
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
                # Stamp the cross-video edge with the (deterministic, since a<b by
                # claim_id) source claim's video so the edges table's
                # UNIQUE(src, rel, dst, video_id) actually dedups on re-run — a
                # NULL video_id is treated as distinct by SQLite, which would
                # duplicate the edge every consolidation pass.
                if res.label == "contradiction" and res.contradict >= threshold:
                    if write_edges:
                        store.add_edge(a.claim_id, "CONTRADICTS", b.claim_id,
                                       src_type="Claim", dst_type="Claim",
                                       video_id=a.video_id, confidence=res.contradict)
                    results.append(ContradictionPair(a, b, "CONTRADICTS", round(res.contradict, 4),
                                                     link(a), link(b)))
                elif include_supports and res.label == "entailment" and res.entail >= threshold:
                    if write_edges:
                        store.add_edge(a.claim_id, "SUPPORTS", b.claim_id,
                                       src_type="Claim", dst_type="Claim",
                                       video_id=a.video_id, confidence=res.entail)
                    results.append(ContradictionPair(a, b, "SUPPORTS", round(res.entail, 4),
                                                     link(a), link(b)))

    results.sort(key=lambda p: p.score, reverse=True)
    return results


# --------------------------------------------------------------------------- #
# dedup / decay — the LIVE caller of the supersede lifecycle (W1.4 / spec §4.7)
# --------------------------------------------------------------------------- #


def _normalize_text(text: str) -> str:
    """Case/punctuation/whitespace-insensitive key for exact-duplicate detection."""
    return " ".join(tokenize(text))


def dedup_claims(store: LoomStore) -> int:
    """Supersede duplicate / corrected claims, returning how many were superseded.

    Conservative by construction (spec §4.7 "merge duplicate claims … superseded
    claims are versioned, never deleted"):

    * **within-video exact duplicates** — committed claims with identical
      normalized text in the SAME video: keep the earliest (by ``t_start_s`` then
      ``claim_id``), supersede the rest. Cross-video duplicates are NEVER touched —
      the same claim in two videos is consensus evidence (handled by
      :func:`cluster_claims`), not a duplicate.
    * **CORRECTS edges** — for each ``(correction)-[:CORRECTS]->(corrected)`` edge
      (emitted by :func:`link_claim_relations`), supersede the corrected claim by
      the correction.

    Idempotent: a claim already superseded drops out of the committed set / is
    skipped on the next pass, so re-running supersedes nothing new.
    """
    superseded = 0

    # 1) within-video exact duplicates
    buckets: dict = defaultdict(list)
    for c in store.list_claims(status=STATUS_COMMITTED):
        buckets[(c.video_id, _normalize_text(c.text))].append(c)
    for (_vid, norm), dups in buckets.items():
        if not norm or len(dups) < 2:
            continue
        dups.sort(key=lambda c: (c.t_start_s, c.claim_id))
        keeper = dups[0]
        for dup in dups[1:]:
            store.supersede_claim(dup.claim_id, keeper.claim_id)
            superseded += 1

    # 2) CORRECTS edges: the correction (src) supersedes the corrected (dst)
    for e in store.edges(rel="CORRECTS"):
        corrected = store.get_claim(e["dst"])
        correction = store.get_claim(e["src"])
        if corrected is None or corrected.status != STATUS_COMMITTED:
            continue
        if correction is None or correction.status != STATUS_COMMITTED:
            continue
        store.supersede_claim(corrected.claim_id, correction.claim_id)
        superseded += 1

    return superseded


# --------------------------------------------------------------------------- #
# consolidation orchestrator (spec §4 stage 7 — the background job)
# --------------------------------------------------------------------------- #


@dataclass
class ConsolidationReport:
    topics: int = 0
    contradictions: int = 0
    supports: int = 0
    consensus_clusters: int = 0
    superseded: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def consolidate(
    store: LoomStore, *, nli: NLIBackend, settings: Optional[Settings] = None
) -> ConsolidationReport:
    """Run the full cross-corpus consolidation pass (spec §4 stage 7).

    Topic induction → NLI-verified contradiction/agreement detection (CONTRADICTS
    + SUPPORTS edges) → consensus clustering → dedup. Intended to run as a
    background job after ingest, NOT on the per-video ingest path. Idempotent:
    every leg is deterministic and its writes are ``UNIQUE``-guarded or
    status-gated.
    """
    settings = settings or store.config.settings
    # Local imports to avoid a circular import (topics/consensus import this module).
    from .consensus import cluster_claims
    from .topics import induce_topics

    topics = induce_topics(store, settings=settings)

    # All graph agreement/disagreement edges come from the NLI here (correct even
    # for lexically near-identical-but-negated pairs); cluster_claims is used only
    # to count consensus clusters, with no edge writes.
    pairs = find_contradictions(
        store, nli=nli, threshold=settings.contradiction_threshold,
        include_supports=True, write_edges=True,
    )
    contradictions = sum(1 for p in pairs if p.relation == "CONTRADICTS")
    supports = sum(1 for p in pairs if p.relation == "SUPPORTS")

    clusters = cluster_claims(store, write_edges=False)
    consensus_clusters = sum(1 for c in clusters if c.support_count >= 2)

    superseded = dedup_claims(store)

    return ConsolidationReport(
        topics=len(topics), contradictions=contradictions, supports=supports,
        consensus_clusters=consensus_clusters, superseded=superseded,
    )
