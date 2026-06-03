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
from .traverse import expand


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]], *, k: int = 60, top_k: int = 10
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion over several ranked (id, score) lists."""
    scores: dict = {}
    for lst in ranked_lists:
        for rank, (item_id, _score) in enumerate(lst):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
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
    dense = store.vector_search(query_vec, pool, video_id=video_id)
    lexical = store.lexical_search(query, pool)
    if video_id:
        lexical = [(mid, s) for (mid, s) in lexical if mid.startswith(video_id)]
    legs = [dense, lexical]
    if use_graph:
        seeds = [
            mid
            for mid, _ in rrf_fuse([dense, lexical], k=settings.rrf_k, top_k=settings.top_k)
        ]
        rels = list(graph_rels) if graph_rels else ["SUPPORTS", "CONTRADICTS", "ELABORATES"]
        graph = expand(store, seeds, rels=rels, hops=graph_hops, video_id=video_id)
        legs.append(graph)
    return rrf_fuse(legs, k=settings.rrf_k, top_k=settings.top_k)
