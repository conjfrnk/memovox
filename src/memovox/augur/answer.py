"""Augur answer synthesis (spec §5).

Composes a grounded answer **strictly from retrieved Moments**, with every
sentence carrying a citation to ``(video, timestamp, modality)``. The default
synthesizer is extractive (free, deterministic); a generative LLM is used when
configured. Low-evidence queries are flagged rather than confabulated.
"""

from __future__ import annotations

from typing import List, Optional

from ..backends.base import Embedder, LLMBackend
from ..config import Settings
from ..loom.models import make_provenance
from ..loom.store import LoomStore
from ..observe import Tracer
from ..util import split_sentences, tokenize, truncate
from .planner import plan as plan_query
from .retrieve import retrieve
from .types import Answer, Citation

_LOW_EVIDENCE_MSG = (
    "I don't have enough indexed evidence to answer that confidently. "
    "Try ingesting more sources or rephrasing."
)


def _best_sentence(text: str, query: str) -> str:
    q_tokens = set(tokenize(query))
    best, best_score = "", -1
    for sentence in split_sentences(text) or [text]:
        overlap = sum(1 for w in tokenize(sentence) if w in q_tokens)
        if overlap > best_score:
            best, best_score = sentence, overlap
    return best.strip()


def _synthesize_extractive(citations: List[Citation], *, limit: int = 4) -> str:
    parts = []
    for c in citations[:limit]:
        snippet = c.snippet.strip()
        if not snippet:
            continue
        if snippet[-1] not in ".!?":
            snippet += "."
        parts.append(f"{snippet} [{c.index}]")
    return " ".join(parts)


_LLM_SYSTEM = (
    "Answer the question using ONLY the numbered sources. Cite every sentence "
    "with the matching [n]. If the sources don't answer it, say so. Be concise."
)


def _synthesize_llm(llm: LLMBackend, query: str, citations: List[Citation]) -> str:
    sources = "\n".join(f"[{c.index}] {c.snippet}" for c in citations)
    prompt = f"SOURCES:\n{sources}\n\nQUESTION: {query}\n\nANSWER (with [n] citations):"
    return llm.complete(prompt, system=_LLM_SYSTEM, temperature=0.0).strip()


def ask(
    store: LoomStore,
    query: str,
    *,
    embedder: Embedder,
    llm: Optional[LLMBackend] = None,
    settings: Optional[Settings] = None,
    video_id: Optional[str] = None,
    tracer: Optional[Tracer] = None,
    modality: str = "any",
    visual_query_vec: Optional[List[float]] = None,
) -> Answer:
    settings = settings or Settings()
    tracer = tracer or Tracer("ask", otel_enabled=settings.otel_enabled)
    qp = plan_query(query)
    # The VISUAL leg (M1.1) turns on when the plan routes to visual OR the caller
    # explicitly asks for modality="visual"; it only fires if a visual query vector
    # is supplied (e.g. an image query), so a plain text ask is byte-identical.
    use_visual = (qp.modality == "visual" or qp.strategy == "visual"
                  or modality == "visual")
    # Consume the plan: the strategy chooses the retrieval mode (and, below, the
    # citation ordering) — it is NOT decorative. Only the contradiction route
    # turns on the graph leg today, walking CONTRADICTS/SUPPORTS edges to surface
    # the OTHER side of a disagreement (a moment that shares no query terms). We
    # deliberately do NOT follow ELABORATES: it is emitted intra-moment only, so
    # following it never reaches a new moment. hybrid/procedure/visual keep the
    # dense+lexical baseline — which is what the (factual) eval queries route to,
    # so the retrieval gates stay green.
    use_graph = qp.strategy == "contradiction"
    # SUPPORTS is included alongside CONTRADICTS on purpose: a contradiction
    # answer can then surface both the disagreement AND corroborating context.
    # The extractive synthesizer cites every surfaced moment neutrally (it does
    # not editorialize the relation), so adding SUPPORTS only widens evidence.
    graph_rels = ["CONTRADICTS", "SUPPORTS"] if use_graph else None
    with tracer.span("retrieve") as _sp:
        fused = retrieve(
            store, query, embedder=embedder, settings=settings, video_id=video_id,
            use_graph=use_graph, graph_rels=graph_rels, span=_sp,
            use_visual=use_visual, visual_query_vec=visual_query_vec,
        )
        _sp.add_counter("results", len(fused))
    if not fused:
        return Answer(text=_LOW_EVIDENCE_MSG, citations=[], strategy=qp.strategy,
                      low_evidence=True, metrics=tracer.to_dict())

    with tracer.span("synthesize") as _sp:
        moment_ids = [mid for mid, _ in fused]
        score_by_id = dict(fused)
        moments = store.get_moments(moment_ids)

        citations: List[Citation] = []
        video_cache: dict = {}
        for i, moment in enumerate(moments, start=1):
            video = video_cache.get(moment.video_id)
            if video is None:
                video = store.get_video(moment.video_id)
                video_cache[moment.video_id] = video
            prov = make_provenance(
                video, moment.t_start_s, moment.t_end_s,
                modality=moment.modality, speaker=moment.speaker_id,
                confidence=round(min(1.0, score_by_id.get(moment.moment_id, 0.0) * 30), 4),
            ) if video else None
            snippet = _best_sentence(moment.text_for_embedding(), query)
            citations.append(
                Citation(
                    index=i,
                    video_id=moment.video_id,
                    moment_id=moment.moment_id,
                    t_start_s=moment.t_start_s,
                    t_end_s=moment.t_end_s,
                    modality=moment.modality,
                    speaker=moment.speaker_id,
                    title=video.title if video else None,
                    deep_link=prov.deep_link if prov else None,
                    snippet=truncate(snippet, 240),
                    score=round(score_by_id.get(moment.moment_id, 0.0), 6),
                )
            )

        if qp.strategy == "temporal":
            # Multi-hop temporal synthesis: order the cited moments chronologically
            # by their video's published_at (ascending), missing dates last, then
            # RE-INDEX so the [n] markers stay aligned with the new order. This must
            # happen BEFORE synthesis, because the extractive synthesizer emits
            # [c.index] and iterates citations in list order — both must reflect it.
            def _published_at(c: Citation) -> str:
                video = video_cache.get(c.video_id)
                return (video.published_at or "") if video else ""

            citations.sort(key=lambda c: (_published_at(c) == "", _published_at(c)))
            for new_index, c in enumerate(citations, start=1):
                c.index = new_index

        if llm is not None and getattr(llm, "is_generative", False):
            try:
                text = _synthesize_llm(llm, query, citations)
            except Exception:
                text = _synthesize_extractive(citations)
        else:
            text = _synthesize_extractive(citations)

        low_evidence = not text.strip()
        if low_evidence:
            text = _LOW_EVIDENCE_MSG
        _sp.add_counter("citations", len(citations))
    return Answer(text=text, citations=citations, strategy=qp.strategy,
                  low_evidence=low_evidence, metrics=tracer.to_dict())
