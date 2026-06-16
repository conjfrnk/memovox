"""Augur answer synthesis (spec §5).

Composes a grounded answer **strictly from retrieved Moments**, with every
sentence carrying a citation to ``(video, timestamp, modality)``. The default
synthesizer is extractive (free, deterministic); a generative LLM is used when
configured. Low-evidence queries are flagged rather than confabulated.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ..backends.base import Embedder, LLMBackend, Reranker
from ..config import Settings
from ..loom.models import make_provenance
from ..loom.store import LoomStore
from ..observe import Tracer
from ..util import split_sentences, tokenize, truncate
from .planner import decompose, llm_decompose
from .retrieve import retrieve
from .stitch import stitch_clips
from .types import Answer, Citation

_LOW_EVIDENCE_MSG = (
    "I don't have enough indexed evidence to answer that confidently. "
    "Try ingesting more sources or rephrasing."
)


# W5.1/W5.4 relevance gate. The signal asks: do the query's DISTINCTIVE terms appear,
# jointly, in a single cited moment? "Distinctive" = not a function word and not a
# common English word — so a generic word that is merely absent from a niche corpus
# ("arrive", "capital") neither vetoes a real question nor fakes relevance for an
# out-of-corpus one. The COMMON_WORDS list is deliberately generic (no corpus topic
# terms). NOTE: English-only; a non-English corpus falls back to df-based weighting.
_REL_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "as", "by", "with", "from", "we", "you", "they", "i", "not", "no",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "say", "said", "tell", "about", "their", "his", "her",
}

# Common English content words (verbs/nouns/adjectives) that are NOT topic-distinctive.
# Dropped from the relevance signal so they cannot veto a real question (a df=0 generic
# verb like "arrive") or fake relevance for an absent topic (an incidentally-rare word
# like "capital"/"boiling"). Curated to contain zero domain/topic terms.
_COMMON_WORDS = frozenset("""
able about above across act actual add after again against age ago agree air all
allow almost alone along already also although always among amount another answer any
anyone anything appear area around arrive ask available away back bad base because
become been before begin behind believe below best better between big bit body both
break bring build business call came can cannot car care carry case catch cause center
certain chance change check child choose city claim class clear close come common
community company complete consider continue control cost could country couple course
cover create current cut data day deal decide deep develop die difference different
discuss done door down draw drive drop during each early easy eat effect either else
end enough enter even ever every example expect experience explain face fact fall family
far feel few field figure fill final find fine first follow food foot force form found
free friend full game general get give glad goes going gone good got great group grow
guess guy had half hand happen happy hard has have head hear heart held help here high
hold home hope hour house however huge idea important include increase inside instead
interest into issue item job join just keep kind knew know land large last late later
lead learn least leave left less let level life light like line list little live local
long look lose lot love low made main make man many matter may mean meet member might
mind minute miss moment money month more most move much must name near need never new
next nice night nor note nothing now number off offer office often okay old once only
open order other our out over own page part pass past pay people perhaps person pick
piece place plan play point possible power present press pretty probably problem process
provide pull purpose push put question quick quite rather reach read ready real reason
receive recent record remember report rest result return right rise room round run safe
said same save saw school season second see seem seen sell send sense set several shall
share short should show side simple since single small social some something sometimes
soon sort sound space speak special spend stand start state stay step still stop story
street strong study stuff such sure system take talk teach team tell thank their them
then there thing think this those though thought three through time today together told
too took top total touch toward tried true try turn type under understand until upon
use used usual very view visit wait walk want watch water way week well went were what
when where whether while white whole why will win window wish within without word work
world would write wrong year yes yet young
""".split())


def _rel_tokens(text: str) -> set:
    return {t for t in tokenize(text)
            if t not in _REL_STOP and t not in _COMMON_WORDS and len(t) > 2}


# Minimum distinctive query mass (as a fraction of the max possible IDF, log(n+1))
# required for an answer to clear the relevance gate. Calibrated on the stress corpus
# so a query whose only in-corpus term is generic is refused. See _relevance_coverage.
_RELEVANCE_MIN_MASS_FRAC = 0.35


def _relevance_coverage(store, query: str, citations: List["Citation"],
                        min_moments: int = 0) -> float:
    """IDF-weighted fraction of the query's distinctive content tokens that the
    CITED moments actually cover. ~1.0 when the citations contain the query's rare
    terms; ~0 when the query asks about something absent from the corpus. This is
    the signal behind the W5.1 gate: out-of-corpus questions match only generic
    tokens (their distinctive terms have no corpus support), so coverage collapses.
    Returns 1.0 (never gate) when the query has no distinctive tokens."""
    q = _rel_tokens(query)
    if not q:
        return 1.0
    try:
        n = max(1, int(store.stats().get("moments", 1)))
    except Exception:  # noqa: BLE001 - stats is best-effort; never break an answer
        n = 1
    if n < min_moments:
        return 1.0  # corpus too small for IDF to be a reliable relevance signal

    # IDF over the query's DISTINCTIVE tokens (function + common words already removed
    # by _rel_tokens). KEEP tokens absent from the corpus (df==0) at max IDF: an absent
    # distinctive term ("mercury", "mongolia") is exactly what should sink coverage for
    # an out-of-corpus question. (Generic absent words like "arrive" were already
    # dropped as common, so they can't wrongly veto a real question.)
    weights = {t: math.log((n + 1) / (store.doc_freq(t) + 1)) for t in q}
    total = sum(weights.values())
    if total < 1e-6:
        return 1.0  # only ubiquitous (idf~0) terms -> maximally on-topic, never gate

    # Backstop: a real query carries a MINIMUM distinctive mass; floor the denominator
    # as a fraction of the maximum possible IDF (log(n+1)) so it is corpus-size-robust.
    denom = max(total, _RELEVANCE_MIN_MASS_FRAC * math.log(n + 1))

    # JOINT single-moment coverage (max over citations), NOT a union across all results.
    # Scattered generic tokens each in a different moment must not add up to fake
    # relevance; a genuinely answerable query has ONE moment that covers its distinctive
    # terms together. The cited moment's video TITLE is included so questions naming the
    # speaker/event ("Peter Attia ... saturated fat") match metadata too.
    best = 0.0
    for c in citations:
        ctoks = _rel_tokens(" ".join(filter(None, (c.source_text or c.snippet or "", c.title or ""))))
        covered = sum(w for t, w in weights.items() if t in ctoks)
        if covered > best:
            best = covered
    return best / denom


def _best_sentence(text: str, query: str) -> str:
    q_tokens = set(tokenize(query))
    best, best_score = "", -1
    for sentence in split_sentences(text) or [text]:
        overlap = sum(1 for w in tokenize(sentence) if w in q_tokens)
        if overlap > best_score:
            best, best_score = sentence, overlap
    return best.strip()


def _citation_text(moment) -> str:
    """The answerable CONTENT of a moment for snippet selection + LLM synthesis:
    the spoken transcript and any literal on-screen OCR text. The VLM's prose
    *description* of the frame (``visual_caption``) is deliberately excluded — it is
    a retrieval aid, not content, and a verbose caption ("The image shows a man
    wearing sunglasses…") otherwise wins the snippet and makes the synthesizer
    reason about the picture instead of what was said/shown. Falls back to the
    caption only for a pure-visual moment with no transcript/OCR, so such a moment
    still yields a non-empty citation."""
    parts = [p for p in (getattr(moment, "transcript", None),
                         getattr(moment, "ocr_text", None)) if p]
    if parts:
        return "\n".join(parts).strip()
    return (getattr(moment, "visual_caption", None) or "").strip()


def _includes_unverified_visual(moment) -> bool:
    """True when a moment's answerable content includes on-screen/visual material
    that did NOT pass the entailment gate. Claims are extracted from the spoken
    transcript only (assay.claims), so OCR text and a pure-visual caption are never
    verify-before-commit checked — a poisoned slide could otherwise reach an answer
    indistinguishable from vetted speech. Citations carrying such content are flagged
    ``ocr_unverified`` so clients can mark it lower-trust. Mirrors _citation_text's
    content rule: content = transcript + OCR, falling back to the caption only when
    there is no transcript and no OCR."""
    transcript = (getattr(moment, "transcript", None) or "").strip()
    ocr = (getattr(moment, "ocr_text", None) or "").strip()
    caption = (getattr(moment, "visual_caption", None) or "").strip()
    return bool(ocr) or (not transcript and bool(caption))


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
    # Give the LLM each citation's FULL content (source_text) rather than the
    # one-sentence display snippet, so an answer-bearing sentence with no
    # query-token overlap is still visible. Falls back to the snippet if unset.
    sources = "\n".join(f"[{c.index}] {c.source_text or c.snippet}" for c in citations)
    prompt = f"SOURCES:\n{sources}\n\nQUESTION: {query}\n\nANSWER (with [n] citations):"
    return llm.complete(prompt, system=_LLM_SYSTEM, temperature=0.0).strip()


def _apply_rerank(reranker, query, fused, store):
    """Rerank a fused (id, score) list (no-op for identity / empty / None)."""
    if reranker is None or not fused:
        return fused
    texts = None
    if reranker.needs_text:
        texts = {m.moment_id: m.text_for_embedding()
                 for m in store.get_moments([mid for mid, _ in fused])}
    return reranker.rerank(query, fused, texts=texts)


def _merge_round_robin(legs, top_k):
    """Interleave per-sub-query fused lists (rank-0 of each, then rank-1, …),
    de-duplicating moments and capping at ``top_k`` — so every clause contributes
    its top result(s) before any clause's deeper results (per-clause coverage WHEN
    ``top_k >= len(legs)``; with fewer slots than clauses the later clauses are
    intentionally dropped, the deterministic cost of a small top_k)."""
    merged: List[Tuple[str, float]] = []
    seen = set()
    depth = 0
    while len(merged) < top_k and any(depth < len(leg) for leg in legs):
        for leg in legs:
            if depth < len(leg):
                mid, score = leg[depth]
                if mid not in seen:
                    seen.add(mid)
                    merged.append((mid, score))
                    if len(merged) >= top_k:
                        break
        depth += 1
    return merged


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
    reranker: Optional["Reranker"] = None,
) -> Answer:
    settings = settings or Settings()
    tracer = tracer or Tracer("ask", otel_enabled=settings.otel_enabled)
    # Decompose the query (single-clause => one verbatim sub-query). The LLM
    # decomposer is opt-in (planner_agentic + a generative LLM) with a guaranteed
    # deterministic fallback; the free path is always the deterministic decompose.
    if getattr(settings, "planner_agentic", False) and llm is not None and \
            getattr(llm, "is_generative", False):
        qp = llm_decompose(llm, query)
    else:
        qp = decompose(query)
    multi = len(qp.subqueries) > 1
    plan_dicts = [sq.to_dict() for sq in qp.subqueries]
    # The VISUAL leg (M1.1) turns on when the plan routes to visual OR the caller
    # explicitly asks for modality="visual"; it only fires if a visual query vector
    # is supplied (e.g. an image query), so a plain text ask is byte-identical.
    use_visual = (settings.visual_retrieval
                  and (qp.modality == "visual" or qp.strategy == "visual"
                       or modality == "visual"))
    # Consume the plan: the strategy chooses the retrieval mode (and, below, the
    # citation ordering) — it is NOT decorative. Only the contradiction route
    # turns on the graph leg today, walking CONTRADICTS/SUPPORTS edges to surface
    # the OTHER side of a disagreement (a moment that shares no query terms). We
    # deliberately do NOT follow ELABORATES: it is emitted intra-moment only, so
    # following it never reaches a new moment. hybrid/procedure/visual keep the
    # dense+lexical baseline — which is what the (factual) eval queries route to,
    # so the retrieval gates stay green.
    # SUPPORTS is included alongside CONTRADICTS on purpose: a contradiction answer
    # can then surface both the disagreement AND corroborating context. The
    # extractive synthesizer cites every surfaced moment neutrally, so adding
    # SUPPORTS only widens evidence. (Per-clause graph routing is derived inside the
    # multi branch from each sub-query's strategy, NOT from qp.strategy.)
    with tracer.span("retrieve") as _sp:
        if not multi:
            # SINGLE-CLAUSE: the literal today's path (retrieve over the full query),
            # so the output stays byte-identical.
            use_graph = qp.strategy == "contradiction"
            graph_rels = ["CONTRADICTS", "SUPPORTS"] if use_graph else None
            fused = retrieve(
                store, query, embedder=embedder, settings=settings, video_id=video_id,
                use_graph=use_graph, graph_rels=graph_rels, span=_sp,
                use_visual=use_visual, visual_query_vec=visual_query_vec,
            )
        else:
            # MULTI-PART (spec §5): retrieve + rerank EACH sub-query (the rerank sees
            # a focused clause), then merge round-robin so every clause is covered.
            legs = []
            for sq in qp.subqueries:
                sq_graph = sq.strategy == "contradiction"
                sq_rels = ["CONTRADICTS", "SUPPORTS"] if sq_graph else None
                sq_visual = settings.visual_retrieval and (
                    sq.modality == "visual" or sq.strategy == "visual")
                leg = retrieve(
                    store, sq.text, embedder=embedder, settings=settings, video_id=video_id,
                    use_graph=sq_graph, graph_rels=sq_rels,
                    use_visual=sq_visual, visual_query_vec=visual_query_vec,
                )
                legs.append(_apply_rerank(reranker, sq.text, leg, store))
            fused = _merge_round_robin(legs, settings.top_k)
        _sp.add_counter("results", len(fused))
    if not fused:
        return Answer(text=_LOW_EVIDENCE_MSG, citations=[], strategy=qp.strategy,
                      low_evidence=True, metrics=tracer.to_dict(), plan=plan_dicts)

    # M2.1 rerank stage (spec §5/§3): the single-clause path reranks the fused set
    # here (the multi-part path already reranked each sub-query above). The free
    # identity reranker is a no-op -> [n] indices/deep links re-derive contiguously.
    if reranker is not None and not multi:
        with tracer.span("rerank") as _rsp:
            fused = _apply_rerank(reranker, query, fused, store)
            _rsp.add_counter("candidates", len(fused))

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
            content = _citation_text(moment)
            snippet = _best_sentence(content, query)
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
                    ocr_unverified=_includes_unverified_visual(moment),
                    source_text=truncate(content, 600),
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
            except Exception as exc:  # noqa: BLE001 - graceful fallback, but visible
                import sys
                print(f"memovox: LLM answer synthesis failed ({type(exc).__name__}: "
                      f"{exc}); using the extractive synthesizer.", file=sys.stderr)
                text = _synthesize_extractive(citations)
        else:
            text = _synthesize_extractive(citations)

        # W5.1: a synthesized answer is only trustworthy if the cited evidence
        # actually covers what was asked. Gate on IDF-weighted query coverage so an
        # out-of-corpus question ("capital of Mongolia") is refused rather than
        # answered with the nearest-but-irrelevant moments. Withhold the citations
        # too — presenting them would itself be an unsupported claim.
        floor = getattr(settings, "answer_relevance_floor", 0.0)
        relevance = _relevance_coverage(
            store, query, citations,
            min_moments=getattr(settings, "answer_relevance_min_moments", 0),
        ) if floor > 0 else 1.0
        low_evidence = (not text.strip()) or relevance < floor
        if low_evidence:
            text = _LOW_EVIDENCE_MSG
            citations = []
        _sp.add_counter("relevance", round(relevance, 4))
        _sp.add_counter("citations", len(citations))
    # M2.3 clip stitching (spec §5/§8): a strictly ADDITIVE final step reading the
    # finalized citations — it widens cited spans into deep-linked watch windows and
    # never mutates text/citations, so the gates stay byte-identical.
    clips = stitch_clips(citations, videos=video_cache,
                         merge_gap_s=settings.clip_merge_gap_s)
    return Answer(text=text, citations=citations, strategy=qp.strategy,
                  low_evidence=low_evidence, metrics=tracer.to_dict(), plan=plan_dicts,
                  clips=clips)
