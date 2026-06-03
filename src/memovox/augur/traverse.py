"""Graph-expansion retrieval leg (W3.2, spec §5).

A BFS over the claim->claim layer of the temporal knowledge graph. Starting from
the seed moments produced by dense+lexical retrieval, we walk SUPPORTS /
CONTRADICTS / ELABORATES edges between their claims and map the reachable claims
back to *their* moments. This surfaces moments that share no query terms with the
question but are semantically linked through a claim relation — the synthesis
("where do these sources contradict?", "what builds on this?") leg.

The walk follows edges in BOTH directions for every relation. SUPPORTS and
CONTRADICTS are effectively symmetric — ``talk_b CONTRADICTS talk_a`` means a
seed sitting in talk_a must follow the *incoming* edge to reach talk_b — so we
must not assume the seed is always the edge source.

Moments are scored by hop distance (closer hops score higher), and seed moments
are excluded so the leg only contributes *new* moments to the RRF fusion.

Forward note: today ``ELABORATES`` (from ``link_claim_relations``) only links
claims WITHIN the same moment, so following it never reaches a NEW moment (the
neighbor's moment == the seed's moment, already visited) — it is a no-op for
moment-level expansion. It stays in the default rels (harmless, forward
compatible) and will start contributing once ELABORATES spans moments.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..loom.models import STATUS_COMMITTED


def expand(
    store,
    seed_moment_ids: Sequence[str],
    *,
    rels: Sequence[str],
    hops: int = 1,
    video_id: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Surface moments graph-reachable from ``seed_moment_ids``.

    For each seed moment, find its committed claims, follow ``neighbors`` along
    every relation in ``rels`` (out AND in), map neighbor claims back to their
    moments, and score each newly-reached moment by ``1 / (hop + 1)`` — nearer
    moments rank higher. Seed moments are never returned. When ``video_id`` is
    set the walk stays in-video (cross-video edges are not followed), matching
    the in-video scoping of the dense/lexical legs.
    """
    scored: dict = {}
    frontier = list(seed_moment_ids)
    visited = set(seed_moment_ids)
    for hop in range(1, hops + 1):
        nxt: List[str] = []
        for mid in frontier:
            for claim in store.claims_for_moment(mid):
                neighbor_claim_ids = set()
                for rel in rels:
                    for e in store.neighbors(claim.claim_id, rel=rel, direction="out"):
                        neighbor_claim_ids.add(e["dst"])
                    for e in store.neighbors(claim.claim_id, rel=rel, direction="in"):
                        neighbor_claim_ids.add(e["src"])
                for ncid in neighbor_claim_ids:
                    nc = store.get_claim(ncid)
                    # Both ends committed-only: seeds come from claims_for_moment
                    # (committed by default); guard the neighbor too so a
                    # once-committed-then-superseded claim (W1.4 lifecycle) cannot
                    # surface its moment via a stale edge.
                    if nc is None or nc.status != STATUS_COMMITTED or nc.moment_id in visited:
                        continue
                    if video_id and not nc.moment_id.startswith(video_id):
                        continue
                    score = 1.0 / (hop + 1)
                    if score > scored.get(nc.moment_id, 0.0):
                        scored[nc.moment_id] = score
                    visited.add(nc.moment_id)
                    nxt.append(nc.moment_id)
        frontier = nxt
    return sorted(scored.items(), key=lambda x: x[1], reverse=True)
