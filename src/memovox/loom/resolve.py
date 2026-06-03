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

from typing import Iterable, List

from ..assay.claims import extract_mentions
from ..backends.entity_link import EntityLinker
from .consolidate import _content_tokens
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


def link_claim_relations(store, claims: Iterable[Claim]) -> None:
    """Emit claim->claim discourse edges within a video (spec §6, W3.1).

    Two typed edges capture how a speaker's claims relate over time:

    * **ELABORATES** — claim N elaborates the immediately-following claim N+1
      when they sit in the SAME Moment and share a speaker (claims within a
      Moment are contiguous and time-ordered, so this links the discourse
      flow inside a single span). The edge spans ``a.t_start_s .. b.t_end_s``.
    * **CORRECTS** — a ``CORRECTION``-typed claim supersedes the NEAREST prior
      claim (scanning backward in pipeline/time order) that shares a subject
      (CONTENT-word overlap on ``subject`` via :func:`_content_tokens`, which
      strips stopwords/short tokens so "the model" and "the method" do NOT
      match on a bare "the") or an entity (overlap of :func:`extract_mentions`).
      If nothing matches, no edge is emitted — a correction with no antecedent
      is left dangling rather than forced.

    Operates on COMMITTED claims only (consistent with :func:`resolve_entities`):
    only trusted facts belong in the graph. Idempotent — every edge is
    provenance-stamped and guarded by the edges table's
    ``UNIQUE(src, rel, dst, video_id)`` constraint, so re-linking is a no-op.
    """
    committed: List[Claim] = [c for c in claims if c.status == STATUS_COMMITTED]

    # --- ELABORATES: consecutive same-speaker claims inside one Moment -----
    for a, b in zip(committed, committed[1:]):
        if a.moment_id != b.moment_id:
            continue
        if a.speaker_id != b.speaker_id:
            continue
        store.add_edge(
            a.claim_id, "ELABORATES", b.claim_id,
            src_type="Claim", dst_type="Claim", video_id=a.video_id,
            t_start_s=a.t_start_s, t_end_s=b.t_end_s,
        )

    # --- CORRECTS: a CORRECTION -> nearest prior sharer in the same video --
    # Precompute each claim's content-subject tokens and entity surface forms
    # once, so the backward scan below is a set lookup rather than re-tokenizing.
    subj_tokens = [_content_tokens(c.subject) for c in committed]
    entities = [set(extract_mentions(c)) for c in committed]
    for i, c in enumerate(committed):
        if c.claim_type != "CORRECTION":
            continue
        c_subject = subj_tokens[i]
        c_entities = entities[i]
        # Scan backward; link the nearest prior claim that shares a CONTENT
        # subject token or an entity, then stop. Stopword-only overlap ("the")
        # does NOT count — a spurious CORRECTS can suppress a still-valid claim.
        for j in range(i - 1, -1, -1):
            p = committed[j]
            if p.video_id != c.video_id:
                continue
            shares_subject = bool(c_subject & subj_tokens[j])
            shares_entity = bool(c_entities & entities[j])
            if shares_subject or shares_entity:
                store.add_edge(
                    c.claim_id, "CORRECTS", p.claim_id,
                    src_type="Claim", dst_type="Claim", video_id=c.video_id,
                    t_start_s=c.t_start_s, t_end_s=c.t_end_s,
                )
                break
