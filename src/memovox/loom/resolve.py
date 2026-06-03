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

import difflib
import re
from typing import Iterable, List

from ..assay.claims import extract_mentions
from ..backends.entity_link import EntityLinker
from ..util import slugify
from .consolidate import _content_tokens
from .models import STATUS_COMMITTED, Claim, Entity, Speaker


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


# --------------------------------------------------------------------------- #
# cross-video speaker resolution (spec §4.6 / §12, W4.1)
# --------------------------------------------------------------------------- #

#: A raw label is ANONYMOUS (a bare diarization turn id, no real name) when it
#: matches one of these shapes — e.g. ``spk_0``, ``SPEAKER_00``, ``speaker-1``.
#: Anonymous speakers are NEVER merged across videos (§12: cross-video voice
#: identity is hard; refusing to merge is the conservative, correct default).
_ANON_RE = re.compile(r"^(spk_?\d+|speaker[_-]?\d+)$", re.IGNORECASE)

#: Names this close (difflib ratio) are treated as the same person, so trivial
#: punctuation/whitespace differences ("Dr. Lee" vs "Dr Lee") merge while
#: genuinely-different names ("Dr. Lee" vs "Prof. Kim") stay apart. Kept HIGH so
#: resolution errs toward NOT merging when unsure (§12: never over-merge).
_NAME_SIMILARITY = 0.9


def _speaker_name(spk: Speaker) -> str:
    """The human name to resolve by (resolved_name preferred over the label)."""
    return (spk.resolved_name or spk.label or "").strip()


def _normalize_name(name: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for name matching."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_anonymous(spk: Speaker) -> bool:
    name = _speaker_name(spk)
    if not name:
        return True
    return bool(_ANON_RE.match(name))


def resolve_speakers(store) -> None:
    """Resolve same-named speakers across the corpus to ONE canonical identity.

    The speaker analog of :func:`resolve_entities`. Today speakers are namespaced
    PER-VIDEO (``f"{video_id}:{raw}"``) and never unified; this collapses the
    SAME named speaker across videos onto a single deterministic ``spk:<slug>``
    identity so the eval's ``der`` becomes a real cross-video signal, while
    leaving the per-video rows in place (linked by ``SAME_AS``) so provenance is
    preserved.

    Conservative by construction (spec §12):

    * **Anonymous** speakers (bare diarization ids like ``SPEAKER_00``) are never
      merged across videos — each stays self-canonical with no ``SAME_AS`` edge.
    * Named speakers are clustered by NORMALIZED name with a high
      :data:`_NAME_SIMILARITY` floor, so when unsure we do NOT merge.

    Clustering is a **greedy single pass**: each speaker (in a stable
    sorted-by-speaker_id order) is compared against the representative name of
    each existing cluster and joins the FIRST cluster above the
    :data:`_NAME_SIMILARITY` floor, else starts its own. A transitive name
    "chain" (A~B, B~C but A≁C) is therefore partitioned greedily — but
    deterministically, because the input order is fixed.

    The canonical identity key is the **normalized-name slug** of the cluster's
    representative — chosen as the member whose normalized name sorts first
    (``min`` over :func:`_normalize_name`), so the canonical ``spk:<slug>`` is a
    pure function of the NAMES present and is decoupled from video-hash ordering.
    A consequence: two speakers whose names normalize to the SAME slug unify onto
    one identity even if difflib happened to place them in different clusters.
    This is INTENTIONAL free-path behavior (spec §12: "name/metadata merge first;
    voiceprints optional") — name-based resolution deliberately merges
    same-named / spelling-variant speakers. We do NOT add a uniqueness suffix to
    split them (that would wrongly fracture accent/spelling variants of the same
    person); disambiguating a genuine same-name-different-person collision is
    deferred to W4.2 voiceprints.

    Idempotent + deterministic: the canonical id is ``spk:<slug>``, the canonical
    row is upserted in place, and ``SAME_AS`` edges are guarded by the edges
    table's ``UNIQUE(src, rel, dst, video_id)`` constraint — re-running over an
    unchanged corpus is a no-op.
    """
    # Skip canonical rows (spk:*) so re-resolution doesn't treat them as inputs.
    speakers = [s for s in store.list_speakers() if not s.speaker_id.startswith("spk:")]
    named = [s for s in speakers if not _is_anonymous(s)]

    # Cluster named speakers by normalized name with a fuzzy floor. Process in a
    # stable order (by speaker_id) so the greedy single-pass partition is
    # deterministic; the canonical slug itself is name-based (see below).
    named.sort(key=lambda s: s.speaker_id)
    clusters: List[List[Speaker]] = []
    cluster_norms: List[str] = []
    for spk in named:
        norm = _normalize_name(_speaker_name(spk))
        placed = False
        for i, existing in enumerate(cluster_norms):
            if norm == existing or difflib.SequenceMatcher(
                None, norm, existing
            ).ratio() >= _NAME_SIMILARITY:
                clusters[i].append(spk)
                placed = True
                break
        if not placed:
            clusters.append([spk])
            cluster_norms.append(norm)

    for members in clusters:
        # Pick the representative BY NAME (lowest normalized name) so the
        # canonical slug depends only on the names present, NOT on which
        # video-hash sorted first. Near-identical names yield the same slug, so
        # the golden der is unchanged; this just decouples the id from hash order.
        rep = min(members, key=lambda s: _normalize_name(_speaker_name(s)))
        rep_name = _speaker_name(rep)
        canonical_id = "spk:" + slugify(rep_name, default="speaker")
        # Upsert the canonical identity row once.
        store.upsert_speaker(
            Speaker(
                speaker_id=canonical_id,
                label=rep_name,
                resolved_name=rep_name,
                canonical_id=canonical_id,
            )
        )
        for member in members:
            # Stamp the per-video speaker with the canonical id, preserving its
            # own label / resolved_name / voiceprint.
            store.upsert_speaker(
                Speaker(
                    speaker_id=member.speaker_id,
                    label=member.label,
                    voiceprint_ref=member.voiceprint_ref,
                    resolved_name=member.resolved_name,
                    canonical_id=canonical_id,
                )
            )
            # SAME_AS edge carries the per-video speaker's OWN video id. The id is
            # ``{video_id}:{raw}`` and video ids themselves contain a colon
            # (``vid:hash``), so split on the LAST ":" to recover the video id.
            video_id = member.speaker_id.rsplit(":", 1)[0]
            store.add_edge(
                member.speaker_id, "SAME_AS", canonical_id,
                src_type="Speaker", dst_type="Speaker", video_id=video_id,
            )
