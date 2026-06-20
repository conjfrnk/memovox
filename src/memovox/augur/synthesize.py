"""Corpus-level synthesis — the "literature review" output mode (spec §5).

``synthesize(topic)`` is the cross-corpus counterpart to ``ask(query)``: instead
of answering one question from the top retrieved moments, it reads everything the
corpus says about a topic and reports

  * **consensus** — claims that AGREE across multiple videos (with a support count
    and a consensus score), and
  * **disagreements** — cross-video NLI contradictions,

keeping the two apart even when a contradiction is lexically near-identical to its
negation: a token-equivalent cluster whose members the NLI flags as contradictory
is demoted out of consensus and surfaced as a disagreement instead.

The free path is extractive and **every synthesis sentence carries a citation**
(spec §5); a generative LLM, when configured, composes prose from the same
structured consensus/contradiction evidence (and falls back to extractive on
error). Low-evidence topics are flagged, never confabulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..backends.base import LLMBackend, NLIBackend
from ..config import Settings
from ..loom.consensus import clusters_from_groups, partition_claims
from ..loom.consolidate import _content_tokens, find_contradictions
from ..loom.models import make_provenance
from ..loom.store import LoomStore
from ..util import truncate
from .types import Citation

_LOW_EVIDENCE_MSG = (
    "I don't have enough indexed evidence to synthesize that topic. "
    "Try ingesting more sources or broadening the topic."
)


@dataclass
class Synthesis:
    topic: str
    text: str
    citations: List[Citation] = field(default_factory=list)
    consensus_points: List[dict] = field(default_factory=list)
    contradictions: List[dict] = field(default_factory=list)
    low_evidence: bool = False

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "text": self.text,
            "low_evidence": self.low_evidence,
            "consensus_points": self.consensus_points,
            "contradictions": self.contradictions,
            "citations": [c.to_dict() for c in self.citations],
        }


def _topic_claims(store: LoomStore, topic_tokens: set) -> list:
    return [c for c in store.list_claims(status="committed")
            if _content_tokens(c.text) & topic_tokens]


def _build_citations(store: LoomStore, moment_ids: List[str]) -> List[Citation]:
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
        ) if video else None
        citations.append(Citation(
            index=i, video_id=moment.video_id, moment_id=moment.moment_id,
            t_start_s=moment.t_start_s, t_end_s=moment.t_end_s, modality=moment.modality,
            speaker=moment.speaker_id, title=video.title if video else None,
            deep_link=prov.deep_link if prov else None,
            snippet=truncate(moment.text_for_embedding(), 240),
        ))
    return citations


def _cluster_contradicts(cluster, nli: NLIBackend, threshold: float) -> bool:
    """True if any CROSS-video member pair of the cluster is an NLI contradiction.

    This is what keeps a lexically-equivalent-but-negated pair (e.g. "X holds" vs
    "X does not hold") from being reported as consensus — the token clustering
    groups them, but the NLI polarity check demotes them.
    """
    claims = cluster.claims
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i], claims[j]
            if a.video_id == b.video_id:
                continue
            res = nli.classify(a.text, b.text)
            if res.label == "contradiction" and res.contradict >= threshold:
                return True
    return False


def _cluster_entails(cluster, nli: NLIBackend, threshold: float) -> bool:
    """True iff some CROSS-video member pair is an NLI ENTAILMENT (either direction).

    Symmetric to :func:`_cluster_contradicts`: a cluster grouped only by embedding-cosine
    co-location ("breakfast IS the most important meal" vs a video DEBUNKING that thesis —
    both about breakfast, cosine-near, but NEUTRAL) must NOT be reported as agreement. The
    cosine fallback proposes candidates; the NLI must CONFIRM real agreement before it is
    consensus — the agreement-side analog of why contradictions are NLI-verified. Token-
    equivalent clusters (the free path) trivially pass (near-identical text -> entailment)."""
    claims = cluster.claims
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i], claims[j]
            if a.video_id == b.video_id:
                continue
            if (nli.classify(a.text, b.text).entail >= threshold
                    or nli.classify(b.text, a.text).entail >= threshold):
                return True
    return False


def _cite(text: str, index: int) -> str:
    snippet = (text or "").strip()
    if snippet and snippet[-1] not in ".!?":
        snippet += "."
    return f"{snippet} [{index}]"


_LLM_SYSTEM = (
    "Write a short, neutral literature-review synthesis of what the sources say "
    "about the topic. Use ONLY the numbered sources; cite every sentence with the "
    "matching [n]. State points of consensus and points of disagreement. Be concise."
)


def _synthesize_llm(llm: LLMBackend, topic: str, citations: List[Citation],
                    consensus_points: List[dict], contradictions: List[dict]) -> str:
    sources = "\n".join(f"[{c.index}] {c.snippet}" for c in citations)
    prompt = f"TOPIC: {topic}\n\nSOURCES:\n{sources}\n\nSYNTHESIS (with [n] citations):"
    return llm.complete(prompt, system=_LLM_SYSTEM, temperature=0.0).strip()


def synthesize(
    store: LoomStore,
    topic: str,
    *,
    nli: NLIBackend,
    llm: Optional[LLMBackend] = None,
    embedder: Optional[object] = None,
    settings: Optional[Settings] = None,
) -> Synthesis:
    """Synthesize what the corpus says about ``topic`` (spec §5)."""
    settings = settings or Settings()
    topic_tokens = _content_tokens(topic)
    if not topic_tokens:
        return Synthesis(topic=topic, text=_LOW_EVIDENCE_MSG, low_evidence=True)

    claims = _topic_claims(store, topic_tokens)
    if not claims:
        return Synthesis(topic=topic, text=_LOW_EVIDENCE_MSG, low_evidence=True)

    # RELEVANT POOL + CROSS-CORPUS FLOOR. _topic_claims OR-matches EVERY topic token,
    # including generic question words ("what"/"how") that _content_tokens keeps, so the
    # raw pool spans the whole corpus: an out-of-corpus topic ("population of Brazil")
    # then confabulates confident cross-video "consensus"/"contradictions" + hundreds of
    # off-topic citations (the round-3/round-4 synthesize leaks). Restrict the pool to
    # claims whose moment GENUINELY covers the topic (>= the relevance floor of the
    # topic's content words), then require >= 2 distinct covering videos — a synthesis is
    # cross-corpus by definition, so a topic no two sources genuinely cover has nothing to
    # synthesize. This focuses citations on the real topic and is what ask()'s top-k
    # retrieval gate achieves implicitly.
    from .answer import _coverage_tokens, _relevance_coverage
    cov_q = _coverage_tokens(topic)
    if cov_q:
        claims = [c for c in claims
                  if len(cov_q & _coverage_tokens(c.text)) / len(cov_q)
                  >= settings.answer_relevance_floor]
    if len({c.video_id for c in claims}) < 2:
        return Synthesis(topic=topic, text=_LOW_EVIDENCE_MSG, low_evidence=True)

    # Citation set: the distinct moments backing the (covering) topic claims, ordered
    # deterministically so [n] markers are stable.
    seen: set = set()
    moment_ids: List[str] = []
    for c in sorted(claims, key=lambda c: (c.video_id, c.t_start_s, c.claim_id)):
        if c.moment_id not in seen:
            seen.add(c.moment_id)
            moment_ids.append(c.moment_id)
    citations = _build_citations(store, moment_ids)
    index_of: Dict[str, int] = {c.moment_id: c.index for c in citations}

    # Topicality guard (mirror ask()): the covering tokens must also name a genuine corpus
    # topic (distinctive token with df >= min_df, not an incidental hapax).
    if _relevance_coverage(store, topic, citations,
                           min_moments=settings.answer_relevance_min_moments) \
            < settings.answer_relevance_floor:
        return Synthesis(topic=topic, text=_LOW_EVIDENCE_MSG, low_evidence=True)

    # Consensus: token-equivalence clusters, NLI-verified to exclude disagreements.
    # W5.6: when consensus_cosine is enabled AND an embedder is available, also group
    # paraphrases/synonyms by embedding cosine (embedded lazily so the default free
    # path pays nothing and stays byte-identical).
    vectors = None
    cosine = settings.consensus_cosine
    if cosine > 0.0 and embedder is not None:
        vectors = {c.claim_id: embedder.embed_one(c.text) for c in claims}
    groups, _ = partition_claims(claims, jaccard=settings.consensus_jaccard,
                                 cosine=cosine, vectors=vectors)
    clusters = clusters_from_groups(store, groups)
    consensus_points: List[dict] = []
    for cl in sorted(clusters, key=lambda c: (-c.support_count, -c.consensus)):
        if cl.support_count < 2:
            continue
        if _cluster_contradicts(cl, nli, settings.contradiction_threshold):
            continue
        # CONSENSUS must be NLI-CONFIRMED, not pure cosine co-location: a cluster the cosine
        # fallback grouped by topic similarity but the NLI finds no cross-video entailment in
        # is a topic neighborhood, not an agreement (a debunking source presented as endorsing
        # the thesis it debunks). Token-equivalent clusters pass trivially (near-identical text).
        if not _cluster_entails(cl, nli, settings.entailment_threshold):
            continue
        rep = max(cl.claims, key=lambda c: (c.salience, c.claim_id))
        consensus_points.append({
            "text": rep.text, "support_count": cl.support_count,
            "consensus": cl.consensus, "videos": cl.videos,
            "citation": index_of.get(rep.moment_id), "claim_id": rep.claim_id,
            "moment_id": rep.moment_id,
        })

    # Disagreements: cross-video contradictions within the topic (read-only).
    pairs = find_contradictions(
        store, nli=nli, topic=topic, threshold=settings.contradiction_threshold,
        write_edges=False,
    )
    contradictions = [p.to_dict() for p in pairs]

    # Compose the grounded, every-sentence-cited synthesis (extractive free path).
    # ``emitted`` tracks claim_ids (NOT moment_ids — they are different namespaces:
    # claim_id is "<moment_id>.c<n>"), so a claim already cited as a consensus point
    # is not re-emitted as a disagreement sentence.
    parts: List[str] = []
    emitted: set = set()
    for cp in consensus_points:
        idx = cp["citation"]
        if idx is not None:
            parts.append(_cite(cp["text"], idx))
            emitted.add(cp["claim_id"])
    for p in pairs:
        for claim in (p.claim_a, p.claim_b):
            idx = index_of.get(claim.moment_id)
            if idx is None or claim.claim_id in emitted:
                continue
            emitted.add(claim.claim_id)
            parts.append(_cite(claim.text, idx))

    # SALIENT FALLBACK: when the free/lexical path extracts no agreement/contradiction
    # STRUCTURE (token-Jaccard consensus + lexical NLI both empty) but the topic IS
    # genuinely covered (the OOC gate above already passed), emit the most salient
    # on-topic claims (distinct moments, cited) instead of a misleading "ingest more
    # sources". Cross-video agreement/contradiction detection needs the optional
    # [embed]/[nli] backends; this keeps synthesize useful on the free path.
    if not parts:
        seen_m: set = set()
        for c in sorted(claims, key=lambda c: (-c.salience, c.video_id, c.t_start_s, c.claim_id)):
            idx = index_of.get(c.moment_id)
            if idx is None or c.moment_id in seen_m:
                continue
            seen_m.add(c.moment_id)
            parts.append(_cite(c.text, idx))
            if len(parts) >= 4:
                break
    extractive = " ".join(parts)

    text = extractive
    if llm is not None and getattr(llm, "is_generative", False):
        from .answer import _llm_citations_valid
        try:
            text = _synthesize_llm(llm, topic, citations, consensus_points, contradictions)
        except Exception as exc:  # noqa: BLE001 - graceful fallback, but visible
            import sys
            print(f"memovox: LLM topic synthesis failed ({type(exc).__name__}: {exc}); "
                  "using the extractive synthesizer.", file=sys.stderr)
            text = extractive
        else:
            # GROUNDING GATE (mirror ask()): an uncited / dangling-marker generative
            # synthesis is discarded for the verified extractive one — every surfaced
            # sentence must carry a citation.
            if not _llm_citations_valid(text, citations):
                text = extractive

    low_evidence = not text.strip()
    return Synthesis(
        topic=topic, text=text or _LOW_EVIDENCE_MSG, citations=citations,
        consensus_points=consensus_points, contradictions=contradictions,
        low_evidence=low_evidence,
    )
