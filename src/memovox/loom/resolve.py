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
from typing import Dict, Iterable, List, Optional, Sequence

from ..assay.claims import extract_mentions
from ..backends.entity_link import EntityLinker
from ..util import short_hash, slugify
from ..vectormath import cosine
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

#: Cosine floor for the OPTIONAL voiceprint merge (W4.2). Two anonymous speakers
#: whose voiceprints are at least this similar are treated as the same voice.
#: Kept HIGH so voice-merging errs toward NOT merging (§12: never over-merge);
#: this only ever runs when a voiceprint backend supplies vectors — the free
#: name-based path (W4.1) never reaches it.
_VOICEPRINT_SIMILARITY = 0.75


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


def cluster_by_voiceprint(
    voiceprints: Dict[str, Sequence[float]], *, threshold: float
) -> List[List[str]]:
    """Greedy cosine clustering of speaker voiceprints (PURE, no I/O).

    The load-bearing W4.2 logic, deliberately dependency-free and hermetically
    testable. A **greedy single pass** (mirroring W4.1's name clustering): each
    speaker, iterated in SORTED speaker-id order so the result is deterministic,
    is compared against the REPRESENTATIVE vector of every existing cluster and
    joins the FIRST cluster whose cosine similarity is ``>= threshold``, else it
    starts a new cluster. The representative is the cluster's first (= lowest-id)
    member — consistent with W4.1's "lowest sorts first" convention. A transitive
    "chain" (A~B, B~C but A≁C) is therefore partitioned greedily — but
    deterministically, because the iteration order is fixed.

    Defensive (the function is documented as pure/defensive): vectors of
    DIFFERENT dimensions are treated as a non-match and never compared via
    cosine. The shared :func:`memovox.vectormath.cosine` silently truncates to the
    common prefix for the dot product while taking each norm over the full vector
    (so ``cosine([1,0], [1,0,0,0]) == 1.0``), which would wrongly merge
    different-dim vectors; the explicit length guard forecloses that. Same-window
    pyannote embeddings are same-dim today, so this is belt-and-suspenders.

    Returns a list of clusters, each a list of speaker ids with the representative
    (lowest id) first. Singleton clusters are included; callers decide whether a
    lone speaker is canonicalized (W4.2 leaves lone anonymous voices unresolved).
    """
    clusters: List[List[str]] = []
    reps: List[Sequence[float]] = []
    for sid in sorted(voiceprints):
        vec = voiceprints[sid]
        placed = False
        for i, rep in enumerate(reps):
            # Length guard: a dimension mismatch is a non-match (see docstring) —
            # never feed it to cosine, which would compare only the shared prefix.
            if len(vec) != len(rep):
                continue
            if cosine(vec, rep) >= threshold:
                clusters[i].append(sid)
                placed = True
                break
        if not placed:
            clusters.append([sid])
            reps.append(vec)
    return clusters


def resolve_speakers(store, *, voiceprints: Optional[Dict[str, Sequence[float]]] = None) -> None:
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

    **Optional voiceprint merge (W4.2, §12).** ``voiceprints`` is an OPTIONAL
    ``{speaker_id: vector}`` map supplied only when a voiceprint backend is
    available (:func:`memovox.backends.get_voiceprint_backend`). When ``None``
    (the FREE path) this function behaves EXACTLY as the name-only W4.1 logic
    above — byte-for-byte unchanged. When provided, AFTER name resolution the
    speakers that name-resolution left unresolved (still-anonymous, ``canonical``
    still their own id) AND that have a voiceprint are clustered by
    :func:`cluster_by_voiceprint`; each MULTI-member voice cluster is assigned a
    deterministic ``spk:voice-<hash>`` canonical id (distinct from the name-based
    ``spk:<slug>``) with ``SAME_AS`` edges. Name-resolved speakers are NEVER
    touched, and a LONE anonymous voice stays unresolved (conservative §12).

    Idempotency caveat (voice path only): the ``spk:voice-<hash>`` id hashes the
    cluster's MEMBER SET, so it is idempotent for a FIXED corpus but NOT stable
    across cluster growth — a later ingest that adds another same-voice speaker
    grows the cluster, changes the hash, and ORPHANS the old ``spk:voice-<oldhash>``
    row + its ``SAME_AS`` edges (queries still resolve correctly via
    :meth:`LoomStore.canonical_speaker` at read time; the stale id just dangles).
    The name-based ``spk:<slug>`` id has no such instability. See
    :func:`_merge_by_voiceprint`.
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
        # An EMPTY normalized name carries no comparable ASCII signal: _normalize_name strips
        # every non-Latin script (CJK, Cyrillic, Arabic, Greek, Hebrew, ...) to "", so two
        # such names matched via "" == "" (and difflib 1.0 on two empties) and collapsed EVERY
        # non-Latin-named speaker across all videos into ONE fictitious identity. Never
        # fuzzy-cluster on an empty norm (each non-Latin name stays its own cluster, the
        # conservative §12 posture); exact-name unification still happens via the canonical id
        # below (a content hash of the original name), so two videos with the SAME non-Latin
        # speaker still resolve together while two DIFFERENT ones stay distinct.
        if norm:
            for i, existing in enumerate(cluster_norms):
                if not existing:
                    continue
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
        # slugify() drops all non-ASCII, so a non-Latin name degenerates to the default and
        # EVERY such speaker would collide on one canonical id ("spk:speaker"). Fall back to a
        # content hash of the original name when the slug is empty, so distinct non-Latin
        # speakers get distinct ids while two videos sharing the SAME name still unify (the
        # hash is a pure function of the name). Latin names keep their slug -> byte-identical.
        slug = slugify(rep_name, default="")
        canonical_id = "spk:" + (slug or ("u-" + short_hash(rep_name)))
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

    # --- OPTIONAL voiceprint merge (W4.2) -----------------------------------
    # Strictly gated: with no voiceprints supplied (the free path) this is a true
    # no-op — the function returns having done ONLY the W4.1 name-based work above.
    if not voiceprints:
        return
    _merge_by_voiceprint(store, voiceprints)


def _merge_by_voiceprint(
    store, voiceprints: Dict[str, Sequence[float]]
) -> None:
    """Voice-merge the ANONYMOUS speakers that name-resolution left unresolved.

    Conservative (§12): only speakers whose name-based canonical is still their
    OWN id (i.e. never merged by name — anonymous diarization labels) and that
    HAVE a voiceprint are eligible. Multi-member voice clusters get a
    deterministic ``spk:voice-<hash(sorted member ids)>`` canonical; lone voices
    stay unresolved.

    Idempotent for a FIXED corpus: the voice canonical is a pure function of the
    cluster's member-id SET, the canonical row is upserted in place, and
    ``SAME_AS`` edges are UNIQUE-guarded — re-running over the same speakers is a
    no-op (already-merged members have a ``spk:voice-*`` canonical, so they fall
    out of the eligible set on the next pass). Known limitation: it is NOT stable
    across cluster GROWTH — a later ingest adding a same-voice speaker changes the
    member set, hence the hash, so the voice-group canonical id changes and the
    old ``spk:voice-<oldhash>`` row + its ``SAME_AS`` edges are orphaned (harmless
    to queries, which resolve via ``canonical_speaker`` at read time). This is the
    trade-off of the optional voice path vs the stable name-based ``spk:<slug>``.
    """
    # Re-read current state so we see the canonical ids name-resolution just set.
    speakers = [s for s in store.list_speakers() if not s.speaker_id.startswith("spk:")]
    eligible: Dict[str, Sequence[float]] = {}
    by_id = {}
    for spk in speakers:
        vec = voiceprints.get(spk.speaker_id)
        if not vec:
            continue
        # Only speakers NAME-resolution did NOT canonicalize are eligible. A
        # speaker whose canonical_id is set (and not its own id) was merged by
        # name — NEVER override that with a voice merge.
        canonical = store.canonical_speaker(spk.speaker_id)
        if canonical != spk.speaker_id:
            continue
        eligible[spk.speaker_id] = vec
        by_id[spk.speaker_id] = spk

    for cluster in cluster_by_voiceprint(eligible, threshold=_VOICEPRINT_SIMILARITY):
        if len(cluster) < 2:
            # A lone anonymous voice stays unresolved (do NOT canonicalize it).
            continue
        members = sorted(cluster)
        canonical_id = "spk:voice-" + short_hash("|".join(members))
        store.upsert_speaker(
            Speaker(
                speaker_id=canonical_id,
                label=canonical_id,
                resolved_name=None,
                canonical_id=canonical_id,
            )
        )
        for sid in members:
            member = by_id[sid]
            store.upsert_speaker(
                Speaker(
                    speaker_id=member.speaker_id,
                    label=member.label,
                    voiceprint_ref=member.voiceprint_ref,
                    resolved_name=member.resolved_name,
                    canonical_id=canonical_id,
                )
            )
            video_id = member.speaker_id.rsplit(":", 1)[0]
            store.add_edge(
                member.speaker_id, "SAME_AS", canonical_id,
                src_type="Speaker", dst_type="Speaker", video_id=video_id,
            )
