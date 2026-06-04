"""Augur hybrid retrieval (spec §5).

Dense (vector) + sparse (lexical/FTS) retrieval fused with Reciprocal Rank
Fusion — the empirically strong baseline. The graph leg (multi-hop over
SUPPORTS/CONTRADICTS edges) is layered on for synthesis questions.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..backends.base import Embedder
from ..config import Settings
from ..loom.store import LoomStore
from ..observe import Span
from .traverse import expand


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    *,
    k: int = 60,
    top_k: int = 10,
    span: Optional[Span] = None,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion over several ranked (id, score) lists."""
    scores: dict = {}
    for lst in ranked_lists:
        for rank, (item_id, _score) in enumerate(lst):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if span is not None:
        span.add_cap("top_k", limit=top_k, dropped=max(0, len(fused) - top_k))
    return fused[:top_k]


def retrieve(
    store: LoomStore,
    query: str,
    *,
    embedder: Embedder,
    settings: Optional[Settings] = None,
    video_id: Optional[str] = None,
    use_graph: bool = False,
    graph_rels: Optional[Sequence[str]] = None,
    graph_hops: int = 1,
    span: Optional[Span] = None,
    use_visual: bool = False,
    visual_query_vec: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Return fused (moment_id, rrf_score) for the query.

    With ``use_graph=True`` a third GRAPH leg is fused in: from the dense+lexical
    seed moments we follow claim->claim SUPPORTS/CONTRADICTS/ELABORATES edges and
    surface the linked moments (spec §5). The default is off, so the existing
    dense+lexical behavior — and every current retrieval/answer test — is
    unchanged; the planner (W3.3) turns it on for synthesis questions.
    """
    settings = settings or Settings()
    pool = max(settings.top_k * 3, 12)
    query_vec = embedder.embed_one(query)
    dense = store.vector_search(query_vec, pool, video_id=video_id, query_text=query)
    lexical = store.lexical_search(query, pool)
    if video_id:
        lexical = [(mid, s) for (mid, s) in lexical if mid.startswith(video_id)]
    if span is not None:
        # pool is the per-leg candidate cap; saturation (len == pool) means the
        # store had at least this many and some may have been dropped upstream.
        span.add_counter("pool", pool)
        span.add_counter("dense_candidates", len(dense))
        span.add_counter("lexical_candidates", len(lexical))
    legs = [dense, lexical]
    if use_graph:
        seeds = [
            mid
            for mid, _ in rrf_fuse([dense, lexical], k=settings.rrf_k, top_k=settings.top_k)
        ]
        rels = list(graph_rels) if graph_rels else ["SUPPORTS", "CONTRADICTS", "ELABORATES"]
        graph = expand(store, seeds, rels=rels, hops=graph_hops, video_id=video_id)
        legs.append(graph)
    # 4th VISUAL leg (M1.1), DEFAULT OFF. Fires only when the plan routes to visual
    # AND a visual query vector exists (e.g. an image query); a text query with no
    # visual representation skips it gracefully, like the empty-graph case.
    if use_visual and visual_query_vec:
        visual = store.visual_search(visual_query_vec, pool, video_id=video_id)
        if span is not None:
            span.add_counter("visual_candidates", len(visual))
        if visual:
            legs.append(visual)
    # span passed only to the final fuse (the user-visible top_k cut), not the seed fuse
    fused = rrf_fuse(legs, k=settings.rrf_k, top_k=settings.top_k, span=span)
    # M3.1 decay (default OFF -> early skip -> byte-identical). When ON, re-weight by
    # recency and drop fully-superseded moments; an all-undated corpus stays
    # byte-identical because every recency multiplier is 1.0.
    if settings.decay_enabled:
        fused = _apply_decay(store, fused, settings)
    return fused


def _video_of(moment_id: str) -> str:
    """The video_id embedded in a moment id (``<video_id>#m####``)."""
    return moment_id.split("#", 1)[0]


def _apply_decay(store: LoomStore, fused, settings: Settings):
    """Recency re-weight + fully-superseded demotion (M3.1, decay path only)."""
    from ..loom.consensus import recency_weight

    vids = {_video_of(mid) for mid, _ in fused}
    dates = {v: (store.get_video(v).published_at if store.get_video(v) else None) for v in vids}
    reference_date = max((d for d in dates.values() if d), default=None)

    reweighted = []
    for mid, score in fused:
        # Demote (exclude) a moment whose claims are ALL superseded — it has claims
        # but none committed (§2: the stored claim is untouched, only its ranking).
        committed = store.claims_for_moment(mid, status="committed")
        if not committed and store.claims_for_moment(mid, status=None):
            continue
        w = recency_weight(dates.get(_video_of(mid)), reference_date,
                           halflife=settings.decay_halflife_days, default=1.0)
        reweighted.append((mid, score * w))
    reweighted.sort(key=lambda x: x[1], reverse=True)  # stable: undated keeps order
    return reweighted
