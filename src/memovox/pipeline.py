"""The ingestion pipeline — a resumable, idempotent DAG (spec §4).

Wires the subsystems: Stentor (acquire/ASR/diarize) → Escapement (Moments) →
Assay (claims + verify) → Loom (triple write + graph + digest). Re-ingesting an
unchanged source is a no-op; re-ingesting changed content replaces it cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import assay, escapement, stentor, tessera
from .backends import (
    get_embedder,
    get_entity_linker,
    get_llm,
    get_nli,
    get_voiceprint_backend,
)
from .config import PIPELINE_VERSION, Config, Settings
from .errors import IngestionError
from .loom import Claim, LoomStore, Speaker, Video
from .loom.digest import render_digest
from .loom.models import STATUS_COMMITTED
from .loom.resolve import link_claim_relations, resolve_entities, resolve_speakers
from .observe import Tracer
from .util import make_video_id, slugify


@dataclass
class IngestReport:
    video_id: str
    title: str
    status: str  # ingested | unchanged | replaced
    n_moments: int = 0
    n_claims_committed: int = 0
    n_claims_unsupported: int = 0
    asr_backend: str = ""
    embed_backend: str = ""
    nli_backend: str = ""
    duration_s: Optional[float] = None
    visual_available: bool = False
    n_visual_events: int = 0
    vlm_backend: str = ""
    ocr_backend: str = ""
    metrics: dict = field(default_factory=dict)  # M0.1 per-stage trace (volatile wall_ms)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _namespace_speaker(video_id: str, speaker: Optional[str]) -> Optional[str]:
    if not speaker:
        return None
    prefix = f"{video_id}:"
    return speaker if speaker.startswith(prefix) else prefix + speaker


def _extract_voiceprints(backend, *, video_id, segments, audio_path) -> Optional[dict]:
    """Build a ``{namespaced_speaker_id: vector}`` map via the OPTIONAL backend.

    Fully gated (W4.2): returns ``None`` unless a voiceprint backend is present
    AND a local audio file exists, so the free/captions path (no backend, no
    media) is always a clean no-op and ``resolve_speakers`` runs name-only. The
    actual per-speaker embedding runs inside ``backend.embed`` — the only code
    that imports pyannote — and is skipped entirely when the backend is absent.

    One representative span per speaker (the longest speech segment) is embedded,
    which is enough to cluster anonymous same-voice speakers across videos.
    """
    if backend is None or not audio_path:
        return None
    best: dict = {}  # speaker_id -> (duration, t_start, t_end)
    for seg in segments:
        if getattr(seg, "kind", "speech") != "speech" or not seg.speaker:
            continue
        dur = (seg.end or 0.0) - (seg.start or 0.0)
        prev = best.get(seg.speaker)
        if prev is None or dur > prev[0]:
            best[seg.speaker] = (dur, seg.start, seg.end)
    voiceprints: dict = {}
    for speaker, (_dur, t0, t1) in best.items():
        try:
            vec = backend.embed(str(audio_path), float(t0 or 0.0), float(t1 or 0.0))
        except Exception:  # pragma: no cover - backend/model failure must not break ingest
            continue
        if vec:
            voiceprints[_namespace_speaker(video_id, speaker)] = vec
    return voiceprints or None


def ingest(
    config: Config,
    source: str,
    *,
    source_url: Optional[str] = None,
    title: Optional[str] = None,
    captions: Optional[str] = None,
    cookies: Optional[str] = None,
    language: Optional[str] = None,
    glossary: Optional[List[str]] = None,
    force: bool = False,
    settings: Optional[Settings] = None,
    store: Optional[LoomStore] = None,
    tracer: Optional[Tracer] = None,
    published_at: Optional[str] = None,
    visual_result: Optional[object] = None,
    modality: Optional[str] = None,
    resolve_corpus: bool = True,
    with_video: bool = False,
) -> IngestReport:
    # CANONICAL ingest() signature (M0.3, single-owner). Keyword-only Phase-4 seams:
    #   published_at: override the source's publication date (local files have none).
    #   visual_result: inject a precomputed Tessera result instead of recomputing
    #                  (reserved for M1.1; None -> tessera.run as today).
    #   modality: reserved citation-routing hint (reserved for M1.1; inert here).
    # Adding a Phase-4 keyword? Add it here, keyword-only with an inert default —
    # do NOT churn the positional list. None reproduces today's free-path behavior.
    config.ensure()
    settings = settings or config.settings
    tracer = tracer or Tracer("ingest", otel_enabled=settings.otel_enabled)

    # M3.3: private-by-default — refuse network egress BEFORE any fetch when on.
    if settings.local_only and source.startswith(("http://", "https://")):
        raise IngestionError(
            f"local_only is set: refusing to acquire the remote source {source!r}. "
            "Ingest a downloaded local file instead, or unset local_only."
        )

    # --- Stentor: acquire + ASR + diarize -------------------------------
    with tracer.span("asr") as _sp:
        st = stentor.run(
            config, source, source_url=source_url, title=title, captions=captions,
            cookies=cookies, asr_backend=settings.asr_backend, language=language, glossary=glossary,
            asr_device=settings.asr_device, asr_compute_type=settings.asr_compute_type,
            asr_allow_cpu=settings.asr_allow_cpu, captions_as_prior=settings.captions_as_prior,
            want_video=with_video,
        )
        _sp.add_counter("segments", len(st.segments))
    meta = st.meta
    video_id = make_video_id(meta.source_url or source, content_hash=meta.content_hash)

    video = Video(
        video_id=video_id,
        source_url=meta.source_url,
        title=meta.title,
        channel=meta.channel,
        # None -> use the source's date; an explicit "" is a real blank override (M0.3).
        published_at=published_at if published_at is not None else meta.published_at,
        duration_s=st.duration,
        lang=st.language,
        content_hash=meta.content_hash,
        pipeline_version=PIPELINE_VERSION,
    )

    owns_store = store is None
    store = store or LoomStore(config)
    try:
        if not force and store.is_unchanged(video):
            return IngestReport(video_id, video.title, "unchanged",
                                asr_backend=st.asr_backend, metrics=tracer.to_dict())

        existed = store.get_video(video_id) is not None
        if existed:
            store.delete_video(video_id)
        store.upsert_video(video)

        embedder = get_embedder(settings.embed_backend, config=config)
        nli = get_nli(settings.nli_backend, config=config)
        llm = get_llm(settings.llm_backend, config=config)
        store.set_meta("embed_backend", embedder.name)

        # --- Tessera: visual track (degrades gracefully, spec §9) -------
        with tracer.span("visual") as _sp:
            # M1.1 may inject a precomputed visual_result; else compute as today.
            visual = visual_result if visual_result is not None else \
                tessera.run(config, meta, settings=settings, span=_sp)
            _sp.add_counter("frames", visual.n_frames)
            _sp.add_counter("scenes", visual.n_scenes)
            _sp.add_counter("events", len(visual.events))

        # --- Escapement: Moments (fused with visual events) -------------
        with tracer.span("moments") as _sp:
            moments = escapement.build_moments(
                video_id, st.segments, embedder=embedder, settings=settings,
                visual_events=visual.events,
            )
            for m in moments:
                m.speaker_id = _namespace_speaker(video_id, m.speaker_id)

            # --- Loom: speakers -----------------------------------------
            for raw, name in (st.speaker_names or {}).items():
                sid = _namespace_speaker(video_id, raw)
                store.upsert_speaker(Speaker(speaker_id=sid, label=raw, resolved_name=name))
            for m in moments:
                if m.speaker_id and store.get_speaker(m.speaker_id) is None:
                    store.upsert_speaker(Speaker(speaker_id=m.speaker_id, label=m.speaker_id.split(":")[-1]))
            _sp.add_counter("moments", len(moments))

        # --- Loom: triple write (vector + lexical via add_moment) -------
        with tracer.span("embed") as _sp:
            if moments:
                embeddings = embedder.embed([m.text_for_embedding() for m in moments])
            else:
                embeddings = []
            for m, emb in zip(moments, embeddings):
                vis_emb = escapement.moment_visual_embedding(m, visual.events) if visual.events else None
                store.add_moment(m, emb, visual_embedding=vis_emb)

            # PRECEDES edges along the timeline.
            for prev, nxt in zip(moments, moments[1:]):
                store.add_edge(prev.moment_id, "PRECEDES", nxt.moment_id,
                               src_type="Moment", dst_type="Moment", video_id=video_id,
                               t_start_s=prev.t_start_s, t_end_s=nxt.t_end_s)
            _sp.add_counter("embedded", len(moments))

        # --- Assay: claims + verify + graph edges -----------------------
        with tracer.span("claims") as _sp:
            committed = unsupported = 0
            all_claims: List[Claim] = []
            for m in moments:
                for claim in assay.run(m, nli=nli, llm=llm, settings=settings):
                    store.add_claim(claim)
                    all_claims.append(claim)
                    if claim.status == STATUS_COMMITTED:
                        committed += 1
                        if claim.speaker_id:
                            store.add_edge(claim.speaker_id, "STATES", claim.claim_id,
                                           src_type="Speaker", dst_type="Claim", video_id=video_id,
                                           t_start_s=claim.t_start_s, t_end_s=claim.t_end_s,
                                           confidence=claim.entailment_score)
                            store.add_edge(claim.claim_id, "ATTRIBUTED_TO", claim.speaker_id,
                                           src_type="Claim", dst_type="Speaker", video_id=video_id)
                    else:
                        unsupported += 1
            _sp.add_counter("committed", committed)
            _sp.add_counter("unsupported", unsupported)
            _sp.add_counter("claims", len(all_claims))

        # --- Loom: entity + speaker resolution + discourse edges --------
        # M3.2: resolve_corpus=False defers the WHOLE-CORPUS resolve passes so a
        # batch sync runs them ONCE (via resolve_corpus_pass) instead of once per
        # video. Default True -> byte-identical single-video ingest.
        with tracer.span("resolve") as _sp:
            if not resolve_corpus:
                # Deferred: the batch sync runs resolve_corpus_pass once for all videos.
                _sp.add_counter("deferred", 1)
            else:
                # Canonicalize each committed claim's mentions into Entity nodes +
                # provenanced MENTIONS edges, so the SAME entity across videos is ONE
                # node. resolve_entities filters to committed claims internally.
                linker = get_entity_linker(settings.entity_backend, config=config)
                resolve_entities(store, all_claims, linker=linker)

                # Unify the SAME named speaker across videos onto one canonical
                # ``spk:<slug>`` identity (per-video rows preserved, linked by SAME_AS).
                # Re-resolves the WHOLE corpus each ingest (idempotent); anonymous
                # diarization labels are never merged across videos by NAME.
                #
                # OPTIONAL voiceprint merge (W4.2, §12): only when a voiceprint backend
                # is installed AND local audio exists do we extract per-speaker
                # voiceprints and pass them in, letting anonymous same-voice speakers
                # merge. On the free/captions path the backend is None and media is
                # absent, so voiceprints is None and resolve_speakers stays name-only
                # (no pyannote import, no behavior change). Voiceprints are transient.
                voiceprint_backend = get_voiceprint_backend(
                    getattr(settings, "voiceprint_backend", "auto"), config=config
                )
                voiceprints = _extract_voiceprints(
                    voiceprint_backend,
                    video_id=video_id,
                    segments=st.segments,
                    audio_path=meta.media_path,
                )
                resolve_speakers(store, voiceprints=voiceprints)

                # Link claim->claim within the video: ELABORATES (adjacent same-speaker
                # claims inside a Moment) and CORRECTS (a CORRECTION -> nearest prior
                # claim sharing a subject/entity). Committed-only; provenance-stamped.
                link_claim_relations(store, all_claims)
                _sp.add_counter("claims", len(all_claims))

        # --- human-readable digest --------------------------------------
        with tracer.span("digest") as _sp:
            digest = render_digest(video, moments, all_claims)
            (config.digests_dir / f"{slugify(video_id)}.md").write_text(digest, encoding="utf-8")
            _sp.add_counter("bytes", len(digest))

        # --- persist per-stage metrics + bump the cumulative ledger ------
        store.record_stage_metrics(video_id, tracer)
        store.bump_ledger({
            "videos": 1,
            "moments": len(moments),
            "claims_committed": committed,
            "claims_unsupported": unsupported,
            "visual_events": len(visual.events),
            "frames": visual.n_frames,
        })

        return IngestReport(
            video_id=video_id,
            title=video.title,
            status="replaced" if existed else "ingested",
            n_moments=len(moments),
            n_claims_committed=committed,
            n_claims_unsupported=unsupported,
            asr_backend=st.asr_backend,
            embed_backend=embedder.name,
            nli_backend=nli.name,
            duration_s=st.duration,
            visual_available=visual.available,
            n_visual_events=len(visual.events),
            vlm_backend=visual.vlm_backend,
            ocr_backend=visual.ocr_backend,
            metrics=tracer.to_dict(),
        )
    finally:
        if owns_store:
            store.close()


def resolve_corpus_pass(config: Config, store: LoomStore,
                        settings: Optional[Settings] = None) -> None:
    """Run the WHOLE-CORPUS resolve passes ONCE over the store (M3.2): entity
    resolution, speaker unification, and per-video discourse edges. A batch sync
    ingests each video with ``resolve_corpus=False`` then calls this once, instead
    of re-resolving the whole corpus on every video. Idempotent."""
    from collections import defaultdict

    settings = settings or config.settings
    linker = get_entity_linker(settings.entity_backend, config=config)
    all_claims = store.list_claims(status="committed")
    resolve_entities(store, all_claims, linker=linker)
    resolve_speakers(store)  # name-only (batch path has no per-video voiceprints)
    by_video = defaultdict(list)
    for c in all_claims:
        by_video[c.video_id].append(c)
    for claims in by_video.values():
        link_claim_relations(store, claims)
