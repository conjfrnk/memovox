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
from dataclasses import dataclass, field
from typing import List, Optional

from ..backends.base import NLIBackend
from ..config import Settings
from ..observe import Span, Tracer
from ..util import deep_link, tokenize
from .models import STATUS_COMMITTED, Claim
from .store import LoomStore

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "as", "by", "with", "from", "we", "you", "they", "i", "not", "no",
    # Interrogative/auxiliary function words: never CONTENT. Keeping them let a topic
    # phrased as a question ("what is AGI?") rope the whole corpus — incl. "I don't know
    # what ..." discourse filler — into the contradiction candidate set, where near-mirror
    # filler pairs passed the precision gate as fabricated CONTRADICTS. ask() already
    # strips these via _rel_tokens; align consolidate so topic-scoped contradiction /
    # consensus / synthesize candidate selection matches.
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "would", "should", "will",
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


#: Default universe ceiling for the offline consolidation job. High enough that real
#: corpora are scanned IN FULL (every video participates in cross-video discovery);
#: the per-bucket cap below — not this — is what bounds cost. The old default of 600
#: silently excluded ~94% of a 10k-claim corpus from the persistent contradiction graph.
DEFAULT_MAX_CLAIMS = 50000

#: Inverted-index blocking cap: a content token shared by more than this many claims
#: is too common to signal a specific agreement/contradiction, and its O(bucket^2)
#: pairs would dominate cost. Skipping such buckets keeps the cross-video scan
#: near-linear at scale. Small/medium corpora (every bucket <= cap) generate exactly
#: the same candidate pairs as the old exhaustive per-claim scan, so results — and the
#: golden contradiction/incremental-equivalence gates — are unchanged.
_BUCKET_CAP = 200


def _candidate_pairs(token_set: dict, *, bucket_cap: int = _BUCKET_CAP):
    """Yield unique ``(cid_a, cid_b)`` (cid_a < cid_b) claim-id pairs that share at
    least one content token, via inverted-index blocking with a per-bucket size cap.

    This replaces the per-claim ``candidates = union(buckets); for a in C: for b in C``
    double loop, which was O(sum |C|^2) and therefore only affordable over a truncated
    prefix of the corpus. Bucket iteration with a hard cap is near-linear and covers
    every claim, so the whole corpus participates in cross-video discovery."""
    inverted: dict = defaultdict(list)
    for cid, toks in token_set.items():
        for t in toks:
            inverted[t].append(cid)
    seen: set = set()
    for ids in inverted.values():
        if len(ids) > bucket_cap:
            continue  # ubiquitous token: low signal, would dominate cost
        for i in range(len(ids)):
            a = ids[i]
            for j in range(i + 1, len(ids)):
                b = ids[j]
                pair = (a, b) if a < b else (b, a)
                if pair in seen:
                    continue
                seen.add(pair)
                yield pair


def find_contradictions(
    store: LoomStore,
    *,
    nli: NLIBackend,
    topic: Optional[str] = None,
    threshold: float = 0.55,
    min_shared: int = 3,
    min_jaccard: float = 0.5,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    write_edges: bool = True,
    include_supports: bool = False,
    span: Optional[Span] = None,
    scope: Optional[set] = None,
    bucket_cap: int = _BUCKET_CAP,
) -> List[ContradictionPair]:
    all_committed = store.list_claims(status="committed")
    # Topic filter BEFORE the cap: a topic-scoped query wants the most relevant
    # claims, not whatever happens to fall in the first ``max_claims`` of ingest
    # order. Capping first silently empties any topic whose claims arrive late
    # (e.g. with 10k claims the saturated-fat videos sit past index 4500, so a
    # cap-then-filter returned zero candidates and *looked* like an NLI miss).
    if topic:
        topic_tokens = _content_tokens(topic)
        if topic_tokens:
            all_committed = [c for c in all_committed
                             if _content_tokens(c.text) & topic_tokens]
    # Universe: bound the PRIOR side by max_claims, but ALWAYS include the scope (new)
    # claims regardless of their rowid, so the incremental new-vs-ALL guarantee holds
    # at scale (else a new claim past index max_claims is absent from the candidate
    # universe entirely and paired against nothing — the M0.2 docstring's guarantee
    # was silently false once total committed > max_claims).
    claims = all_committed[:max_claims]
    if scope:
        have = {c.claim_id for c in claims}
        claims = claims + [c for c in all_committed
                           if c.claim_id in scope and c.claim_id not in have]
    if span is not None:
        span.add_counter("candidates", len(all_committed))
        span.add_cap("max_claims", limit=max_claims,
                     dropped=max(0, len(all_committed) - len(claims)))

    token_set = {c.claim_id: _content_tokens(c.text) for c in claims}
    by_id = {c.claim_id: c for c in claims}

    video_cache = {}
    results: List[ContradictionPair] = []
    nli_calls = 0

    def link(claim: Claim):
        v = video_cache.get(claim.video_id)
        if v is None:
            v = store.get_video(claim.video_id)
            video_cache[claim.video_id] = v
        return deep_link(v.source_url, claim.t_start_s) if v else None

    # Inverted-index blocking (bucket-capped) generates each cross-video candidate pair
    # once. The whole universe participates — no per-claim O(|C|^2) over a truncated
    # prefix — so cross-video discovery is no longer blind past the first max_claims.
    for cid_a, cid_b in _candidate_pairs(token_set, bucket_cap=bucket_cap):
        # Incremental scope (M0.2): when scanning only NEW claims, skip any pair where
        # neither side is new — that pair was already scored in an earlier pass. The
        # candidate UNIVERSE includes all scope claims, so a new claim is still paired
        # against every prior claim (new-vs-ALL).
        if scope is not None and cid_a not in scope and cid_b not in scope:
            continue
        a, b = by_id[cid_a], by_id[cid_b]
        if a.video_id == b.video_id:
            continue  # cross-corpus only
        # PRECISION GATE (free/lexical path): only NLI-compare claims that are
        # substantive NEAR-MIRRORS — they must share >= min_shared content tokens AND
        # overlap by >= min_jaccard. Lexical NLI reliably scores opposing-polarity
        # near-duplicates ("X is harmful" / "X is not harmful") but FALSELY flags
        # unrelated short fragments that share a couple of generic tokens plus a
        # negation cue. Without this gate a full-corpus scan emits hundreds of garbage
        # cross-video edges (the 600-claim cap used to hide them by barely scanning
        # cross-video pairs). Genuine differently-phrased contradictions need a real
        # [nli] backend; the free path cannot tell them from coincidence.
        shared = token_set[cid_a] & token_set[cid_b]
        if len(shared) < min_shared:
            continue
        union = token_set[cid_a] | token_set[cid_b]
        if union and len(shared) / len(union) < min_jaccard:
            continue

        nli_calls += 1
        res = nli.classify(a.text, b.text)
        # Stamp the cross-video edge with the (deterministic, since a<b by claim_id)
        # source claim's video so the edges table's UNIQUE(src, rel, dst, video_id)
        # actually dedups on re-run — a NULL video_id is treated as distinct by SQLite,
        # which would duplicate the edge every consolidation pass.
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

    if span is not None:
        span.add_counter("nli_calls", nli_calls)  # actual NLI work (sparse under scope)
    # Stable tiebreak by (claim_a, claim_b) ids: ties on score must order the same way
    # across runs (PYTHONHASHSEED-independent), so sort by id within equal scores.
    results.sort(key=lambda p: (-p.score, p.claim_a.claim_id, p.claim_b.claim_id))
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
    claims_scanned: int = 0    # M0.2: candidate claims kept after the cap
    claims_skipped: int = 0    # M0.2: candidate claims dropped by the cap (reported, not silent)
    capped: bool = False       # M0.2: True iff the max_claims cap engaged
    metrics: dict = field(default_factory=dict)  # M0.1 per-stage trace (volatile wall_ms)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


_WATERMARK_KEY = "consolidation_watermark"


def consolidate(
    store: LoomStore, *, nli: NLIBackend, settings: Optional[Settings] = None,
    tracer: Optional[Tracer] = None, since_watermark: Optional[int] = None,
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> ConsolidationReport:
    """Run cross-corpus consolidation (spec §4 stage 7), incrementally (M0.2 W5).

    Topic induction → NLI-verified contradiction/agreement detection (CONTRADICTS
    + SUPPORTS edges) → consensus clustering → dedup. Intended as a background job
    after ingest, NOT on the per-video ingest path. Idempotent: every leg is
    deterministic and its writes are ``UNIQUE``-guarded or status-gated.

    **Incremental:** only NEW claims (rowid past ``consolidation_watermark``) are
    scanned for contradictions, each against the FULL committed set (never
    new-vs-new) — so the cumulative graph is identical to a single full pass while
    the expensive NLI runs only on new-involving pairs. Pass ``since_watermark=0``
    to force a full pass. The ``max_claims`` cap is now *reported*
    (``claims_scanned``/``claims_skipped``/``capped``), never silent. This is the
    single owner of incremental consolidation — M3.2/M3.3 call it, never reimplement.

    The report's ``contradictions``/``supports`` are TOTAL graph edge counts (stable
    across idempotent re-runs — a second pass with no new claims reports the same
    totals and does zero NLI work, surfaced as ``new_*`` span counters), not just
    this-pass deltas. **Re-ingest note:** re-ingesting a video re-inserts its claims
    at higher rowids, so they re-enter ``scope`` and that video is re-scanned vs all
    others — correctness-safe (edges are ``UNIQUE``-guarded) but not free.
    """
    settings = settings or store.config.settings
    tracer = tracer or Tracer("consolidate", otel_enabled=settings.otel_enabled)
    # Local imports to avoid a circular import (topics/consensus import this module).
    from .consensus import cluster_claims
    from .topics import induce_topics

    watermark = (
        since_watermark if since_watermark is not None
        else int(store.get_meta(_WATERMARK_KEY, "0") or 0)
    )
    scope = store.committed_claim_ids_since(watermark)
    new_high_water = store.max_claim_rowid()

    with tracer.span("topics") as _sp:
        topics = induce_topics(store, settings=settings)
        _sp.add_counter("topics", len(topics))

    # All graph agreement/disagreement edges come from the NLI here (correct even
    # for lexically near-identical-but-negated pairs); cluster_claims is used only
    # to count consensus clusters, with no edge writes.
    with tracer.span("contradictions") as _sp:
        pairs = find_contradictions(
            store, nli=nli, threshold=settings.contradiction_threshold,
            include_supports=True, write_edges=True, span=_sp,
            scope=scope, max_claims=max_claims,
        )
        # This pass's NEW findings (0 on an idempotent re-run); the report's totals
        # below reflect the whole graph so a re-run never misreports "0".
        _sp.add_counter("new_contradictions", sum(1 for p in pairs if p.relation == "CONTRADICTS"))
        _sp.add_counter("new_supports", sum(1 for p in pairs if p.relation == "SUPPORTS"))
        # Reported paging derived from the cap event (M0.1) on this span. Both this
        # cap site AND cluster_claims below page the SAME list_claims[:max_claims]
        # universe, so this single capped/skipped figure covers both.
        candidates_total = int(_sp.counters.get("candidates", 0))
        cap = next((c for c in _sp.caps if c["name"] == "max_claims"), None)
        claims_skipped = int(cap["dropped"]) if cap else 0
        claims_scanned = candidates_total - claims_skipped
        capped = claims_skipped > 0

    # Totals over the whole consolidation graph (stable, idempotent-safe).
    contradictions = store.count_edges(rel="CONTRADICTS")
    supports = store.count_edges(rel="SUPPORTS")

    with tracer.span("consensus") as _sp:
        clusters = cluster_claims(store, write_edges=False, max_claims=max_claims)
        consensus_clusters = sum(1 for c in clusters if c.support_count >= 2)
        _sp.add_counter("clusters", consensus_clusters)

    with tracer.span("dedup") as _sp:
        superseded = dedup_claims(store)
        _sp.add_counter("superseded", superseded)

    # Advance the watermark only forward (never below an explicitly cold pass).
    store.set_meta(_WATERMARK_KEY, str(new_high_water))

    return ConsolidationReport(
        topics=len(topics), contradictions=contradictions, supports=supports,
        consensus_clusters=consensus_clusters, superseded=superseded,
        claims_scanned=claims_scanned, claims_skipped=claims_skipped, capped=capped,
        metrics=tracer.to_dict(),
    )
