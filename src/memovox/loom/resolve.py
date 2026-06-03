"""Cross-corpus entity resolution (spec §4.6, W2.3).

Wires three previously-scaffolded store primitives into a real knowledge graph:
each committed claim's surface mentions (W2.1's :func:`extract_mentions`) are
canonicalized (W2.2's :class:`EntityLinker`) into ``Entity`` nodes, linked to the
claim via the ``mentions`` table, and stamped onto a provenanced ``MENTIONS``
edge. The canonical id is a deterministic ``ent:<slug>`` (see the linker), so the
SAME entity mentioned in two videos collapses onto ONE node whose mentions span
both — that is what makes the eval's ``entity_f1`` move off 0.0.

Idempotent by construction: :meth:`LoomStore.upsert_entity` is an UPSERT that
UPDATES the row IN PLACE on conflict (it must NOT ``INSERT OR REPLACE`` — that
DELETEs the row first, and the ``mentions.entity_id ON DELETE CASCADE`` FK would
then orphan every mention already linked to a shared entity, so the second video
to mention it would lose the first video's link), the mention link is
``INSERT OR IGNORE``, and the ``MENTIONS`` edge is guarded by the edges table's
``UNIQUE(src, rel, dst, video_id)`` constraint. Re-ingesting / re-resolving an
unchanged corpus is therefore a graph no-op.
"""

from __future__ import annotations

from typing import Iterable

from ..assay.claims import extract_mentions
from ..backends.entity_link import EntityLinker
from .models import STATUS_COMMITTED, Claim, Entity


def resolve_entities(store, claims: Iterable[Claim], *, linker: EntityLinker) -> None:
    """Resolve each committed claim's mentions into canonical graph entities.

    Only committed claims contribute to the graph (unsupported/superseded claims
    have no place in the trusted knowledge layer). Every ``MENTIONS`` edge carries
    the claim's own ``(video_id, t_start_s, t_end_s)`` provenance.
    """
    for claim in claims:
        if claim.status != STATUS_COMMITTED:
            continue
        for surface in extract_mentions(claim):
            can = linker.canonicalize(surface)
            store.upsert_entity(
                Entity(
                    entity_id=can.entity_id,
                    canonical_name=can.name,
                    wikidata_qid=can.wikidata_qid,
                    aliases=[surface],
                )
            )
            # Two intentionally-distinct writes, NOT duplicates: the ``mentions``
            # table is the fast claim<->entity lookup (powers entity_mentions);
            # the ``MENTIONS`` edge is the provenanced graph edge (carries the
            # claim's span/video for traversal). Keep both.
            store.link_mention(claim.claim_id, can.entity_id)
            store.add_edge(
                claim.claim_id, "MENTIONS", can.entity_id,
                src_type="Claim", dst_type="Entity", video_id=claim.video_id,
                t_start_s=claim.t_start_s, t_end_s=claim.t_end_s,
            )
