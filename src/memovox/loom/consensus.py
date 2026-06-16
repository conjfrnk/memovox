"""Cross-corpus claim clustering + consensus scoring (Phase 3, spec §4.7).

The agreement half of "contradiction & agreement detection": cluster
semantically-equivalent committed claims across the corpus, and turn each cluster
into a confidence estimate weighted by **source count × recency × authority**.

Free + deterministic: equivalence is content-token **Jaccard** over the same
stopword-stripped tokens used by contradiction detection (no NLI/model needed on
the free path), and clustering is a union-find over the inverted-index candidate
pairs (so it is not O(n²)). Cross-video agreeing pairs get a provenanced
``SUPPORTS`` edge (``UNIQUE``-guarded → idempotent); within-video equivalence is
left to dedup (W5), since agreement is a *cross-corpus* signal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..util import parse_iso
from .consolidate import _BUCKET_CAP, DEFAULT_MAX_CLAIMS, _candidate_pairs, _content_tokens
from .models import Claim

# Consensus weighting. Source count dominates (more independent sources asserting
# the same thing is the strongest agreement signal); recency and authority refine.
_W_SOURCES = 0.5
_W_AUTHORITY = 0.25
_W_RECENCY = 0.25
#: Distinct-source count at which the source term saturates to 1.0.
_SOURCE_SATURATION = 3.0
#: Recency half-life: a claim one half-life older than the reference date scores
#: half the recency weight of one published on the reference date.
_RECENCY_HALFLIFE_DAYS = 365.0


@dataclass
class ClaimCluster:
    """A set of equivalent claims plus their per-source publish dates.

    ``dates`` maps ``video_id -> published_at`` (ISO string or ``None``) for the
    cluster's sources; it is kept alongside ``claims`` because claims do not carry
    their video's publish date. ``consensus`` is filled in by
    :func:`score_consensus` (0.0 until scored).
    """

    claims: List[Claim]
    dates: Dict[str, Optional[str]] = field(default_factory=dict)
    consensus: float = 0.0

    @property
    def videos(self) -> List[str]:
        return sorted({c.video_id for c in self.claims})

    @property
    def support_count(self) -> int:
        return len(self.videos)

    @property
    def newest_date(self) -> Optional[str]:
        dated = [d for d in self.dates.values() if d]
        return max(dated) if dated else None

    @property
    def representative(self) -> str:
        """The highest-salience claim's text (ties broken by claim_id)."""
        if not self.claims:
            return ""
        rep = max(self.claims, key=lambda c: (c.salience, c.claim_id))
        return rep.text


def recency_weight(published_at: Optional[str], reference_date: Optional[str], *,
                   halflife: float = _RECENCY_HALFLIFE_DAYS, default: float = 0.5) -> float:
    """Exponential half-life recency in [0,1] (the single shared decay model, reused
    by consensus scoring AND retrieval decay — M3.1). ``default`` is returned when a
    date is missing: consensus uses 0.5 (neutral term in a weighted sum); retrieval
    uses 1.0 (a no-op multiplier, so an all-undated corpus is byte-identical)."""
    if not published_at or not reference_date:
        return default
    ref, nd = parse_iso(reference_date), parse_iso(published_at)
    if not ref or not nd:
        return default
    age_days = max(0.0, (ref - nd).total_seconds() / 86400.0)
    return 0.5 ** (age_days / halflife)


def _recency_term(cluster: ClaimCluster, reference_date: Optional[str]) -> float:
    """Exponential-decay recency in [0,1]; neutral 0.5 when dates are absent."""
    return recency_weight(cluster.newest_date, reference_date, default=0.5)


def score_consensus(cluster: ClaimCluster, *, reference_date: Optional[str] = None) -> float:
    """A [0,1] consensus/confidence estimate for a cluster (spec §4.7).

    ``score = w_src·source_term + w_auth·authority_term + w_rec·recency_term``:

    * **source term** — distinct sources, saturating at :data:`_SOURCE_SATURATION`;
    * **authority term** — the cluster's max claim salience (the available
      speaker-authority proxy; salience already ∈ [0,1]);
    * **recency term** — exponential decay of the newest source's age relative to
      ``reference_date`` (neutral 0.5 when publish dates are unavailable).
    """
    source_term = min(1.0, cluster.support_count / _SOURCE_SATURATION)
    saliences = [c.salience for c in cluster.claims]
    authority_term = max(saliences) if saliences else 0.0
    recency_term = _recency_term(cluster, reference_date)
    score = _W_SOURCES * source_term + _W_AUTHORITY * authority_term + _W_RECENCY * recency_term
    return round(min(1.0, max(0.0, score)), 4)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Attach the larger id under the smaller so the root is deterministic.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def _cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def partition_claims(
    claims: List[Claim], *, min_shared: int = 2, jaccard: float = 0.5,
    cosine: float = 0.0, vectors: Optional[Dict[str, list]] = None,
    bucket_cap: int = _BUCKET_CAP,
):
    """Union-find partition of claims into equivalence groups (pure, no store).

    Two claims are equivalent when they share at least ``min_shared`` content
    tokens AND their content-token Jaccard is ``>= jaccard`` — OR (W5.6, opt-in)
    when ``cosine > 0`` and their embedding cosine is ``>= cosine``. The cosine
    fallback groups paraphrases/synonyms that token-Jaccard misses ("AGI is coming
    soon" / "AGI will arrive imminently" share only the token 'agi'); it only fires
    among the inverted-index candidate pairs (claims sharing >=1 token), and is a
    no-op by default (``cosine=0``) and with the lexical hashing embedder (whose
    geometry is not semantic). Returns ``(groups, cross_video_pairs)``.
    """
    by_id = {c.claim_id: c for c in claims}
    tokens = {c.claim_id: _content_tokens(c.text) for c in claims}
    vectors = vectors or {}

    uf = _UnionFind(by_id.keys())
    cross_video: List[tuple] = []
    # Inverted-index blocking (bucket-capped) over claims sharing >=1 token — the same
    # candidate generator the contradiction pass uses, so consensus clustering also
    # scales to the full corpus instead of a 600-claim prefix.
    for cid_a, cid_b in _candidate_pairs(tokens, bucket_cap=bucket_cap):
        toks_a, toks_b = tokens[cid_a], tokens[cid_b]
        token_equiv = (len(toks_a & toks_b) >= min_shared
                       and _jaccard(toks_a, toks_b) >= jaccard)
        cos_equiv = (cosine > 0.0 and cid_a in vectors and cid_b in vectors
                     and _cosine(vectors[cid_a], vectors[cid_b]) >= cosine)
        if not (token_equiv or cos_equiv):
            continue
        uf.union(cid_a, cid_b)
        a, b = by_id[cid_a], by_id[cid_b]
        if a.video_id != b.video_id:
            cross_video.append((a, b))

    grouped: Dict[str, List[Claim]] = defaultdict(list)
    for cid, claim in by_id.items():
        grouped[uf.find(cid)].append(claim)
    groups = [sorted(grouped[root], key=lambda c: c.claim_id) for root in sorted(grouped)]
    return groups, cross_video


def clusters_from_groups(store, groups: List[List[Claim]]) -> List[ClaimCluster]:
    """Score a partition into :class:`ClaimCluster`s, filling per-source dates.

    Recency is measured relative to the newest publish date across ALL groups, so
    consensus scores are comparable within one run.
    """
    video_dates: Dict[str, Optional[str]] = {}

    def _date(video_id: str) -> Optional[str]:
        if video_id not in video_dates:
            v = store.get_video(video_id)
            video_dates[video_id] = v.published_at if v else None
        return video_dates[video_id]

    reference_date = max(
        (d for g in groups for d in (_date(c.video_id) for c in g) if d), default=None
    )
    clusters: List[ClaimCluster] = []
    for members in groups:
        dates = {c.video_id: _date(c.video_id) for c in members}
        cluster = ClaimCluster(claims=members, dates=dates)
        cluster.consensus = score_consensus(cluster, reference_date=reference_date)
        clusters.append(cluster)
    return clusters


def cluster_claims(
    store,
    *,
    min_shared: int = 2,
    jaccard: Optional[float] = None,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    write_edges: bool = True,
) -> List[ClaimCluster]:
    """Partition committed claims into scored clusters of equivalent claims.

    Returns ALL clusters (including singletons), each scored. For every
    CROSS-video equivalent pair a provenanced ``SUPPORTS`` edge is written
    (``write_edges``); within-video equivalence is left to dedup (W5). ``jaccard``
    defaults to ``settings.consensus_jaccard``.
    """
    if jaccard is None:
        jaccard = getattr(store.config.settings, "consensus_jaccard", 0.5)

    claims = store.list_claims(status="committed")[:max_claims]
    groups, cross_video = partition_claims(claims, min_shared=min_shared, jaccard=jaccard)
    if write_edges:
        for a, b in cross_video:
            store.add_edge(
                a.claim_id, "SUPPORTS", b.claim_id,
                src_type="Claim", dst_type="Claim", video_id=a.video_id,
                t_start_s=a.t_start_s, t_end_s=a.t_end_s,
            )
    return clusters_from_groups(store, groups)
