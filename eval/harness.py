"""Eval harness + golden-corpus metric runners (spec §10).

This is the measuring stick the rest of Phase 2 is gated on, so it is built to
be **deterministic and free-stack-pinned**: regardless of which optional ML
packages happen to be installed, the harness pins the free backends (hashing
embedder, lexical NLI, caption "ASR", no LLM/VLM/OCR) and forces HF/transformers
offline, so the numbers are reproducible.

Run it directly::

    python -m eval.harness                    # pretty-print the report
    python -m eval.harness --assert-thresholds  # exit non-zero if a gate fails

Metrics (all pure stdlib, all crash-safe on empty inputs):

  * retrieval — ``hit_rate`` @k, ``mrr``, ``ndcg`` @k
  * ``groundedness`` — fraction of answer sentences entailed by their citations
  * ``clustering_f1`` — pairwise F1, reused for entity resolution and speaker DER
  * ``contradiction_pr`` — precision/recall/F1 over cross-video CONTRADICTS pairs
  * ``synthesis`` — corpus-level synthesis groundedness + whether the seeded
    cross-corpus contradiction is surfaced (Phase 3, spec §5)

What clears its gate **today** (W0.3): ``retrieval`` and ``groundedness`` (they
depend only on existing retrieval + the extractive synthesizer). ``entity_f1``,
``der`` and ``contradiction`` are computed best-effort from whatever the store
currently contains — they are honest real numbers (0.0 when there is nothing to
score) and will move as later workstreams (entity/speaker resolution, contradiction
wiring) land. They are NOT stubbed with fake values.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import sqlite3
import warnings

# Pin the free stack and forbid network model fetches before any memovox import,
# so `python -m eval.harness` (which runs WITHOUT tests/__init__.py) is hermetic.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Make `src/` importable when run as `python -m eval.harness` from the repo root
# (mirrors what tests/__init__.py and the Makefile's PYTHONPATH do).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

GOLDEN_DIR = _REPO_ROOT / "eval" / "golden"

# The free backends, pinned explicitly so eval numbers are reproducible
# regardless of which optional ML packages are installed.
_FREE_BACKENDS = dict(
    embed_backend="hashing",
    nli_backend="lexical",
    asr_backend="captions",
    llm_backend="none",
    vlm_backend="none",
    ocr_backend="none",
    entity_backend="none",
    visual_embed_backend="signature",  # M1.1: free visual embedder pinned
    visual_retrieval=False,            # M1.1: VISUAL leg OFF for the gate
    rerank_backend="none",            # M2.1: identity reranker pinned (off==today)
    clip_merge_gap_s=2.5,             # M2.3: pin the stitch gap so the clip gate can't be env-perturbed
)


@dataclasses.dataclass(frozen=True)
class BackendConfig:
    """A named bundle of backend-slot overrides for the A/B benchmark (M3.4, §7).

    ``FREE_CONFIG`` pins every slot to its free value (the frozen snapshot the gate
    runs on). ``backend_kwargs()`` is the override dict for ``Memovox(**kwargs)``.
    Defaults are the FREE values, so an upgrade config overrides only what it lifts."""

    name: str
    asr_backend: str = "captions"
    embed_backend: str = "hashing"
    nli_backend: str = "lexical"
    llm_backend: str = "none"
    vlm_backend: str = "none"
    ocr_backend: str = "none"
    entity_backend: str = "none"
    voiceprint_backend: str = "none"      # free = deterministic no-op (never loads optional pyannote)
    visual_embed_backend: str = "signature"
    rerank_backend: str = "none"

    def backend_kwargs(self) -> dict:
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self) if f.name != "name"}


FREE_CONFIG = BackendConfig(name="free")

#: Slot -> backend_status() category, for availability probing.
_SLOT_CATEGORY = {
    "asr_backend": "asr", "embed_backend": "embed", "nli_backend": "nli",
    "llm_backend": "llm", "vlm_backend": "vlm", "ocr_backend": "ocr",
    "visual_embed_backend": "visual_embed", "rerank_backend": "rerank",
    "entity_backend": "entity_link", "voiceprint_backend": "voiceprint",
}

#: Upgrade configs that move a TEXT-corpus metric (embed/nli drive groundedness &
#: contradiction; rerank drives mrr/ndcg). Ranked when their deps are installed.
_UPGRADE_CANDIDATES = [
    BackendConfig(name="free+cross-encoder", rerank_backend="cross-encoder"),
    BackendConfig(name="st+deberta", embed_backend="sentence-transformers",
                  nli_backend="deberta-nli"),
]

#: Visual upgrade configs — NOT rankable by the text-corpus benchmark table (they
#: move only the ungated `multimodal` block, not the ranked text metrics).
_VISUAL_CANDIDATES = [
    BackendConfig(name="colpali+surya+qwen", visual_embed_backend="colpali",
                  ocr_backend="surya", vlm_backend="qwen2.5-vl"),
]


def _config_available(config: "BackendConfig", status: dict) -> bool:
    """True iff every slot the config lifts above FREE reports is_available."""
    free = FREE_CONFIG.backend_kwargs()
    for slot, value in config.backend_kwargs().items():
        if value != free[slot] and not status.get(_SLOT_CATEGORY[slot], {}).get(value, False):
            return False
    return True


def available_configs(golden_dir=GOLDEN_DIR):
    """The benchmark configs runnable here: ALWAYS [FREE], plus each text-rankable
    upgrade whose every lifted slot is installed (deterministic, name-sorted). On a
    bare/CI machine this auto-shrinks to exactly [FREE_CONFIG]."""
    from memovox.backends import backend_status

    status = backend_status()
    upgrades = [c for c in _UPGRADE_CANDIDATES if _config_available(c, status)]
    return [FREE_CONFIG] + sorted(upgrades, key=lambda c: c.name)


def unrankable_configs(golden_dir=GOLDEN_DIR):
    """Visual configs declared UNRANKABLE on the text-metric benchmark (reported
    with an explicit reason, never silently scored 0.0). They light up only once
    the benchmark table scores visual-specific metrics over the visual subset."""
    has_visual = (Path(golden_dir) / "visual.json").exists()
    reason = ("visual metrics are not in the text-corpus ranking table"
              if has_visual else "no visual subset present")
    return [(c.name, reason) for c in _VISUAL_CANDIDATES]

# Frozen eval-settings snapshot (M0.1 W8 / review discipline (b)): the default-OFF
# flags whose values the gates implicitly depend on. Pinned so a future default
# flip — or a leaked MEMOVOX_* env var — fails loudly instead of silently moving a
# gate number. (metrics are always-on with zero output change, so there is no
# metrics_enabled flag to pin.) When a later track adds a default-OFF flag, add it
# here in the same commit.
_DEFAULT_OFF_FLAGS = dict(
    visual_enabled=True,         # M-X: feature toggle whose flip would move a gate
    salience_floor=0.0,          # M-X: salience gate toggle (0.0 demotes nothing)
    budget_mode="soft",
    otel_enabled=False,
    vector_prefilter_fts=False,  # M0.2: opt-in FTS vector prefilter must stay OFF on the gate
    visual_retrieval=False,      # M1.1: VISUAL leg master switch OFF on the gate
    decay_enabled=False,         # M3.1: recency decay/demotion OFF on the gate
    asr_device="auto",           # M0.3: ASR device knobs pinned
    asr_compute_type="default",
    asr_allow_cpu=False,
    captions_as_prior=True,      # M0.3: §9 cost lever default pinned
    planner_agentic=False,       # M2.2: LLM query decomposer OFF on the gate
    local_only=False,            # M3.3: egress allowed on the gate (default posture)
)

# Settings fields deliberately NOT pinned (M-X W1): backend selectors (pinned by
# name via _FREE_BACKENDS instead) and pure numeric tuning knobs whose value does
# not toggle a feature on/off. A NEW flag must be pinned above or added here with a
# reason — the reflection completeness meta-test (tests/test_eval.py) enforces it.
_INTENTIONALLY_UNPINNED = frozenset({
    # backend selectors (pinned by name in _FREE_BACKENDS / FREE_CONFIG — M3.4;
    # voiceprint_backend lives only in FREE_CONFIG, the others in both)
    "asr_backend", "embed_backend", "nli_backend", "llm_backend", "vlm_backend",
    "ocr_backend", "entity_backend", "voiceprint_backend", "visual_embed_backend",
    # numeric tuning knobs (not feature toggles)
    "embed_dim", "frame_sample_fps", "frame_side", "frame_max", "scene_threshold",
    "keyframe_min_gain", "keyframe_per_scene_cap", "moment_max_sec", "moment_min_sec",
    "moment_gap_sec", "boundary_similarity", "entailment_threshold", "rrf_k", "top_k",
    "contradiction_threshold", "topic_similarity", "topic_min_size", "consensus_jaccard",
    "visual_workers",  # M1.1: pool size (1=serial); not a feature toggle
    "rerank_backend",  # M2.1: backend selector (pinned to 'none' in _FREE_BACKENDS)
    "clip_merge_gap_s",  # M2.3: clip stitch gap tuning knob
    "decay_halflife_days",  # M3.1: recency half-life tuning knob
    "answer_relevance_floor",       # W5.1: out-of-corpus refusal threshold (numeric knob)
    "answer_relevance_min_moments",  # W5.1: min corpus size before the gate activates
})

# Frozen FULL Settings snapshot (M1.2 W8) — beyond _DEFAULT_OFF_FLAGS, this pins
# every Settings default INCLUDING numeric tuning knobs (top_k, rrf_k, thresholds,
# keyframe knobs, …). A change to ANY default fails the snapshot test, forcing a
# conscious update + a fresh gold re-baseline — the determinism-erosion defense.
EVAL_SETTINGS_SNAPSHOT = {
    "answer_relevance_floor": 0.55, "answer_relevance_min_moments": 50,
    "asr_allow_cpu": False, "asr_backend": "auto", "asr_compute_type": "default",
    "asr_device": "auto", "boundary_similarity": 0.45, "budget_mode": "soft",
    "captions_as_prior": True, "clip_merge_gap_s": 2.5,
    "consensus_jaccard": 0.5, "contradiction_threshold": 0.55,
    "decay_enabled": False, "decay_halflife_days": 365.0,
    "embed_backend": "auto", "embed_dim": 256, "entailment_threshold": 0.5,
    "entity_backend": "auto", "frame_max": 1200, "frame_sample_fps": 1.0, "frame_side": 16,
    "keyframe_min_gain": 0.12, "keyframe_per_scene_cap": 8, "llm_backend": "auto",
    "moment_gap_sec": 2.5, "moment_max_sec": 90.0, "moment_min_sec": 8.0,
    "local_only": False,
    "nli_backend": "auto", "ocr_backend": "auto", "otel_enabled": False,
    "planner_agentic": False, "rerank_backend": "auto", "rrf_k": 60,
    "salience_floor": 0.0, "scene_threshold": 0.3, "top_k": 8, "topic_min_size": 1,
    "topic_similarity": 0.5, "vector_prefilter_fts": False, "visual_embed_backend": "signature",
    "visual_enabled": True, "visual_retrieval": False, "visual_workers": 1,
    "vlm_backend": "auto", "voiceprint_backend": "auto",
}

# Default retrieval cutoff for hit_rate / nDCG.
DEFAULT_K = 5


# --------------------------------------------------------------------------- #
# pure-stdlib metric functions
#
# Retrieval metrics take a list of per-query tuples ``(retrieved_ids, relevant)``
# where ``retrieved_ids`` is the ranked list of result ids and ``relevant`` is a
# set of gold-relevant ids. They never crash on empty inputs.
# --------------------------------------------------------------------------- #

PerQuery = Tuple[Sequence[str], Set[str]]


def hit_rate(per_query: Iterable[PerQuery], *, k: int = DEFAULT_K) -> float:
    """Fraction of queries with >=1 relevant id in the top-k retrieved."""
    queries = list(per_query)
    if not queries:
        return 0.0
    hits = 0
    for retrieved, relevant in queries:
        if relevant and any(rid in relevant for rid in list(retrieved)[:k]):
            hits += 1
    return hits / len(queries)


def mrr(per_query: Iterable[PerQuery]) -> float:
    """Mean reciprocal rank of the first relevant retrieved id (0 if none)."""
    queries = list(per_query)
    if not queries:
        return 0.0
    total = 0.0
    for retrieved, relevant in queries:
        rr = 0.0
        for rank, rid in enumerate(retrieved, start=1):
            if rid in relevant:
                rr = 1.0 / rank
                break
        total += rr
    return total / len(queries)


def ndcg(per_query: Iterable[PerQuery], *, k: int = DEFAULT_K) -> float:
    """Mean nDCG@k with binary relevance gains."""
    queries = list(per_query)
    if not queries:
        return 0.0
    total = 0.0
    for retrieved, relevant in queries:
        if not relevant:
            continue  # contributes 0
        dcg = 0.0
        for i, rid in enumerate(list(retrieved)[:k]):
            if rid in relevant:
                dcg += 1.0 / math.log2(i + 2)  # rank i (0-based) -> position i+1
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        total += (dcg / idcg) if idcg else 0.0
    return total / len(queries)


def subquery_recall(per_subquery: Iterable[PerQuery]) -> float:
    """Fraction of sub-queries whose relevant moment is in the composed answer's
    citations (M2.2). 0.0 on empty input. The agentic-planner coverage metric."""
    subs = list(per_subquery)
    if not subs:
        return 0.0
    hits = sum(1 for retrieved, relevant in subs
               if relevant and any(rid in relevant for rid in retrieved))
    return hits / len(subs)


def groundedness(answer_sentences: Sequence[str], cited_spans, nli, *,
                 threshold: float = 0.5) -> float:
    """Fraction of answer sentences entailed by the cited spans.

    ``cited_spans`` may be a single premise string or an iterable of spans
    (joined). A sentence is grounded if the NLI labels it ENTAILMENT or its
    entailment score clears ``threshold``.
    """
    sentences = [s for s in (answer_sentences or []) if s and s.strip()]
    if not sentences:
        return 0.0
    if isinstance(cited_spans, str):
        premise = cited_spans
    else:
        premise = "\n".join(s for s in (cited_spans or []) if s)
    if not premise.strip():
        return 0.0
    grounded = 0
    for sentence in sentences:
        res = nli.classify(premise, sentence)
        if res.label == "entailment" or res.entail >= threshold:
            grounded += 1
    return grounded / len(sentences)


def _pair_set(clusters: Iterable[Set[str]]) -> Set[frozenset]:
    """All unordered same-cluster pairs across the given clusters."""
    pairs: Set[frozenset] = set()
    for cluster in clusters:
        members = sorted(cluster)
        for a, b in combinations(members, 2):
            pairs.add(frozenset((a, b)))
    return pairs


def clustering_f1(pred_clusters: Iterable[Set[str]],
                  gold_clusters: Iterable[Set[str]]) -> Tuple[float, float, float]:
    """Pairwise precision / recall / F1 of "same cluster" predictions.

    Over all unordered item pairs, a true positive is a pair the prediction
    places in the same cluster that the gold also places together. Reused for
    both entity resolution and speaker DER.
    """
    gold_pairs = _pair_set(gold_clusters)
    pred_pairs = _pair_set(pred_clusters)
    if not gold_pairs and not pred_pairs:
        return (0.0, 0.0, 0.0)
    tp = len(pred_pairs & gold_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(gold_pairs) if gold_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return (precision, recall, f1)


def _norm_pairs(pairs: Iterable[Tuple[str, str]]) -> Set[frozenset]:
    out: Set[frozenset] = set()
    for a, b in pairs:
        if a == b:
            continue
        out.add(frozenset((a, b)))
    return out


def contradiction_pr(found_pairs: Iterable[Tuple[str, str]],
                     gold_pairs: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    """Precision / recall / F1 over unordered (video, video) contradiction pairs."""
    found = _norm_pairs(found_pairs)
    gold = _norm_pairs(gold_pairs)
    if not found and not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(found & gold)
    precision = tp / len(found) if found else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# golden-corpus loading + id translation
# --------------------------------------------------------------------------- #


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class _Ingested:
    """Bundle of an ingested golden corpus + the logical<->store id maps."""

    def __init__(self, mv, logical_to_store: Dict[str, str]) -> None:
        self.mv = mv
        self.logical_to_store = logical_to_store
        self.store_to_logical = {v: k for k, v in logical_to_store.items()}


# The visual fixture lives in eval/golden/ (M1.1) but is NOT part of the SCORED
# corpus — it feeds only the ungated `multimodal` block, so it must never enter the
# retrieval/entity/speaker/contradiction/parity/span scorers.
_NON_CORPUS_STEMS = {"talk_vis"}


def _corpus_vtts(golden_dir: Path = GOLDEN_DIR) -> List[Path]:
    """Golden VTTs that belong to the SCORED corpus (excludes the visual fixture)."""
    return [g for g in sorted(golden_dir.glob("*.en.vtt"))
            if g.name.split(".")[0] not in _NON_CORPUS_STEMS]


#: The non-backend free-path flags _FREE_BACKENDS pins (kept alongside a config's
#: backend slots so an injected BackendConfig changes ONLY backend choice — M3.4).
_FREE_NON_BACKEND = {k: v for k, v in _FREE_BACKENDS.items() if not k.endswith("_backend")}


def _ingest_golden(golden_dir: Path, store_dir: str,
                   config: "BackendConfig" = None) -> _Ingested:
    from memovox import Memovox

    config = config or FREE_CONFIG
    mv = Memovox(store=store_dir, **_FREE_NON_BACKEND, **config.backend_kwargs())
    logical_to_store: Dict[str, str] = {}
    for vtt in sorted(golden_dir.glob("*.en.vtt")):
        logical_id = vtt.name.split(".")[0]  # filename stem before ".en.vtt"
        if logical_id in _NON_CORPUS_STEMS:
            continue
        report = mv.ingest(str(vtt))
        logical_to_store[logical_id] = report.video_id
    # Run consolidation so cross-video CONTRADICTS/SUPPORTS edges + topic_id exist
    # BEFORE retrieval scoring — this is what lets the §5 graph leg fire end-to-end
    # (and feeds topic_f1). Safe on the golden corpus: dedup is a no-op (no exact
    # duplicates), so claim counts are unchanged.
    mv.consolidate()
    return _Ingested(mv, logical_to_store)


# --------------------------------------------------------------------------- #
# per-metric collectors over the ingested store
# --------------------------------------------------------------------------- #


def _relevant_moment_ids(store, store_video_ids: Iterable[str],
                         substrings: Sequence[str]) -> Set[str]:
    """Store moment ids whose text contains ANY of the substrings (case-insensitive)."""
    wanted = [s.lower() for s in substrings if s]
    if not wanted:
        return set()
    hits: Set[str] = set()
    for vid in store_video_ids:
        for m in store.moments_for_video(vid):
            text = m.text_for_embedding().lower()
            if any(sub in text for sub in wanted):
                hits.add(m.moment_id)
    return hits


_CITE_MARKER = re.compile(r"\[(\d+)\]")


def _snippet_for_citation(citations, index: int, store) -> str:
    """Resolve citation ``index`` to its cited span (snippet, else moment text)."""
    for c in citations:
        if c.index == index:
            snippet = (c.snippet or "").strip()
            if not snippet:
                m = store.get_moment(c.moment_id)
                snippet = m.text_for_embedding() if m else ""
            return snippet
    return ""


def _answer_groundedness(ans, nli, store, *, threshold: float) -> float:
    """Fraction of an answer's sentences entailed by *their own* cited spans.

    The extractive synthesizer tags each sentence with the ``[n]`` marker of the
    citation it was drawn from. Grounding each sentence against the moment(s) it
    actually cites is the precise reading of "premise = the cited moments' text"
    (per spec §10) and avoids polluting the premise with unrelated cited spans
    that may carry an opposing-polarity claim from elsewhere in the corpus.

    Conservative rule: a sentence with NO ``[n]`` citation marker counts as NOT
    grounded — an uncited sentence has no provenance to be entailed by. The
    extractive synthesizer always emits ``[n]``, so this doesn't change today's
    score, but it keeps the metric honest for a future generative-LLM answer
    (where matching uncited sentences against the union of all citations would
    leniently hide hallucinations).
    """
    from memovox.util import split_sentences

    citations = ans.citations
    sentences = split_sentences(ans.text)
    sentences = [s for s in sentences if s and s.strip()]
    if not sentences:
        return 0.0

    grounded = 0
    for sentence in sentences:
        indices = [int(m) for m in _CITE_MARKER.findall(sentence)]
        if not indices:
            continue  # uncited sentence -> not grounded
        spans = [_snippet_for_citation(citations, i, store) for i in indices]
        spans = [s for s in spans if s]
        if not spans:
            continue  # cited a missing/empty span -> not grounded
        premise = "\n".join(spans)
        # Strip the [n] citation markers from the hypothesis so the NLI sees
        # only the claim text, not the bracketed reference token.
        hypothesis = _CITE_MARKER.sub("", sentence).strip()
        score = groundedness([hypothesis], premise, nli, threshold=threshold)
        if score >= 1.0:
            grounded += 1
    return grounded / len(sentences)


def _retrieval_and_groundedness(ing: _Ingested, qa: List[dict], nli, *,
                                k: int, entail_threshold: float) -> Tuple[List[PerQuery], float]:
    from memovox.loom.store import LoomStore

    per_query: List[PerQuery] = []
    grounded_scores: List[float] = []
    store_video_ids = list(ing.logical_to_store.values())

    with LoomStore(ing.mv.config) as store:
        for item in qa:
            relevant = _relevant_moment_ids(store, store_video_ids,
                                            item.get("relevant_moment_substrings", []))
            ans = ing.mv.ask(item["q"])
            retrieved_ids = [c.moment_id for c in ans.citations]
            per_query.append((retrieved_ids, relevant))
            grounded_scores.append(
                _answer_groundedness(ans, nli, store, threshold=entail_threshold)
            )

    overall_groundedness = (
        sum(grounded_scores) / len(grounded_scores) if grounded_scores else 0.0
    )
    return per_query, overall_groundedness


def _entity_clusters(ing: _Ingested, gold_entities: dict):
    """Gold + predicted entity clusters over a SHARED atom universe.

    Each atom is a per-video mention key ``"<logical_video_id>:<surface_form>"``.
    Keying mentions per video — parallel to speaker identities — is what makes
    this metric actually move: the gold groups each canonical entity's mentions
    *across* talks, so the shared entity ``Chinchilla`` (in both talks) forms a
    real cross-video same-cluster pair that cross-corpus entity resolution
    (W2.3) must recover, while ``Transformer``/``Llama`` (one talk each) stay
    singletons.

    For the pairwise F1 to be meaningful, GOLD and PRED must score the SAME atom
    set. We therefore (a) build the gold clusters from ``entities.json`` and
    (b) derive the prediction BY READING THE PERSISTED GRAPH — never by
    re-running resolution. This is what makes the metric a real regression guard:
    if :func:`resolve_entities` is broken or a no-op (or the cascade-delete bug
    re-appears and orphans a mention), the lookups below FAIL and the affected
    atoms collapse to singletons, so ``entity_f1`` drops.

    For each gold atom ``"<logical>:<surface>"`` we compute the expected
    deterministic id ``"ent:"+slugify(surface)`` (the offline NullLinker the free
    stack pins) and then VERIFY it against persistence: the atom maps to that
    ``entity_id`` only if ``get_entity`` finds the node AND ``entity_mentions``
    lists a claim whose ``video_id`` is the store id for that logical video.
    Otherwise the atom gets a UNIQUE singleton label so it cannot share a cluster.
    Atoms outside the gold universe never enter either side.
    """
    from collections import defaultdict

    from memovox.loom.store import LoomStore
    from memovox.util import slugify

    # Gold: group per-video mention keys by canonical surface form.
    mentions = gold_entities.get("mentions", {})  # {logical_id: [surface, ...]}
    by_canonical: Dict[str, Set[str]] = defaultdict(set)
    gold_atoms: List[Tuple[str, str, str]] = []  # (atom, logical_id, surface)
    for logical_id, surfaces in mentions.items():
        for surface in surfaces:
            atom = f"{logical_id}:{surface}"
            by_canonical[surface].add(atom)
            gold_atoms.append((atom, logical_id, surface))
    gold_clusters = [members for members in by_canonical.values()]

    # Predicted: label each gold atom by the entity the pipeline PERSISTED for it.
    # Read-only — we never call resolve_entities/extract_mentions here. Catch only
    # sqlite3.OperationalError (a genuinely-absent schema -> 0.0); real errors
    # surface so a resolution bug can't falsely pass.
    pred_groups: Dict[str, Set[str]] = defaultdict(set)
    with LoomStore(ing.mv.config) as store:
        try:
            store.conn.execute("SELECT entity_id FROM entities LIMIT 1").fetchone()
        except sqlite3.OperationalError as exc:
            warnings.warn(
                f"entity_f1: entities table unavailable ({exc}); scoring 0.0",
                stacklevel=2,
            )
            return [], gold_clusters
        for atom, logical_id, surface in gold_atoms:
            expected_id = f"ent:{slugify(surface)}"
            store_vid = ing.logical_to_store.get(logical_id)
            label = f"__unresolved__:{atom}"  # singleton unless persistence agrees
            if store_vid is not None and store.get_entity(expected_id) is not None:
                claim_ids = store.entity_mentions(expected_id)
                claims = store.get_claims(claim_ids)
                if any(c.video_id == store_vid for c in claims):
                    label = expected_id
            pred_groups[label].add(atom)
    pred_clusters = [members for members in pred_groups.values() if members]
    return pred_clusters, gold_clusters


def _speaker_clusters(ing: _Ingested, gold_speakers: dict):
    """Gold + predicted speaker identity clusters (members are logical keys
    ``"<logical_video_id>:<raw_label>"``).

    Gold: group the ``identities`` map by canonical identity id. Predicted: group
    the store's per-video speakers by the canonical identity the pipeline
    PERSISTED for each (``speakers.canonical_id``, exposed via
    :meth:`LoomStore.canonical_speaker`). Cross-video speaker resolution (W4.1)
    unifies the same named speaker across talks onto one ``spk:<slug>`` identity,
    so the two Dr. Lee per-video speakers group together and ``der`` becomes a
    real cross-video same-cluster signal.

    Like ``entity_f1``, this READS PERSISTENCE rather than re-deriving resolution:
    if :func:`resolve_speakers` is broken or a no-op, every per-video speaker is
    self-canonical, the cross-talk Dr. Lee pair collapses to singletons, and
    ``der`` DROPS — so the metric is a genuine regression guard, not a tautology.
    """
    from collections import defaultdict

    from memovox.loom.store import LoomStore

    # Gold clusters keyed by canonical identity id.
    by_identity: Dict[str, Set[str]] = defaultdict(set)
    for key, identity in gold_speakers.get("identities", {}).items():
        by_identity[identity].add(key)
    gold_clusters = [members for members in by_identity.values()]

    # Predicted: read store speakers, translate to logical "<logical>:<label>"
    # keys, and group by the PERSISTED canonical identity. Catch ONLY the
    # "no such table/column" absence (sqlite3.OperationalError) and warn; any
    # other error propagates so a real resolution bug surfaces instead of
    # silently scoring 0.0.
    pred_groups: Dict[str, Set[str]] = defaultdict(set)
    with LoomStore(ing.mv.config) as store:
        try:
            rows = store.conn.execute(
                "SELECT speaker_id, label, resolved_name FROM speakers"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            warnings.warn(
                f"der: speakers table unavailable ({exc}); scoring 0.0",
                stacklevel=2,
            )
            rows = []
        for r in rows:
            speaker_id = r["speaker_id"]
            # Skip the canonical identity rows themselves (spk:*) — only the
            # per-video atoms are scored against the gold per-video keys.
            if speaker_id.startswith("spk:"):
                continue
            label = r["label"]
            # speaker_id is "<store_video_id>:<raw>" and store video ids contain
            # a colon ("vid:hash"), so split on the LAST ":" to recover the
            # video id, then translate store id -> logical.
            store_vid, _, raw = speaker_id.rpartition(":")
            logical = ing.store_to_logical.get(store_vid, store_vid)
            member_key = f"{logical}:{label or raw}"
            # Group by the PERSISTED canonical identity (falls back to the
            # per-video speaker_id when unresolved -> own singleton).
            canonical_key = store.canonical_speaker(speaker_id)
            pred_groups[canonical_key].add(member_key)
    pred_clusters = [members for members in pred_groups.values()]
    return pred_clusters, gold_clusters


def _synthesis_metrics(ing: _Ingested, nli, *, topic: str, threshold: float) -> dict:
    """Corpus-level synthesis quality (Phase 3, spec §5).

    Runs ``synthesize(topic)`` over the golden corpus and measures:

      * ``groundedness`` — fraction of synthesis sentences entailed by their own
        cited span (reuses :func:`_answer_groundedness`; the Synthesis carries the
        same ``text`` + ``[n]`` citations contract as an Answer);
      * ``contradiction_surfaced`` — whether the seeded cross-talk disagreement is
        reported (the marquee cross-corpus signal);
      * ``consensus_points`` — count of agreement clusters surfaced.
    """
    from memovox.loom.store import LoomStore

    syn = ing.mv.synthesize(topic)
    with LoomStore(ing.mv.config) as store:
        g = _answer_groundedness(syn, nli, store, threshold=threshold)
    return {
        "groundedness": round(g, 6),
        "contradiction_surfaced": bool(syn.contradictions),
        "consensus_points": len(syn.consensus_points),
    }


def _contradiction_pairs(ing: _Ingested, gold_contradictions: List[dict]):
    """Found + gold cross-video contradiction pairs as logical-id pairs.

    Reads the CONTRADICTS edges the free consolidation path wrote during
    ``_ingest_golden`` (``mv.consolidate()``) and translates Claim src/dst ->
    store video_id -> logical id. The edges read catches only
    ``sqlite3.OperationalError`` (the "no such table" absence) and warns, so a
    legitimately-missing graph degrades to 0.0 visibly rather than crashing.

    (We do NOT re-run ``mv.contradictions()`` here — consolidation already wrote
    the edges in ``_ingest_golden``; re-running would double the NLI work.)
    """
    from memovox.loom.store import LoomStore

    found: List[Tuple[str, str]] = []
    with LoomStore(ing.mv.config) as store:
        try:
            edges = store.edges(rel="CONTRADICTS")
        except sqlite3.OperationalError as exc:
            warnings.warn(
                f"contradiction: edges table unavailable ({exc}); scoring 0.0",
                stacklevel=2,
            )
            edges = []
        for e in edges:
            a = store.get_claim(e["src"]) if e.get("src") else None
            b = store.get_claim(e["dst"]) if e.get("dst") else None
            if not a or not b:
                continue
            la = ing.store_to_logical.get(a.video_id)
            lb = ing.store_to_logical.get(b.video_id)
            if la and lb and la != lb:
                found.append((la, lb))

    gold = [(c["video_a"], c["video_b"]) for c in gold_contradictions
            if c.get("video_a") and c.get("video_b")]
    return found, gold


# --------------------------------------------------------------------------- #
# run_eval
# --------------------------------------------------------------------------- #


def run_eval(golden_dir=GOLDEN_DIR, *, store: Optional[_Ingested] = None,
             k: int = DEFAULT_K, config: "BackendConfig" = None) -> dict:
    """Run the full eval over the golden corpus and return the report dict.

    When ``store`` is ``None`` (the CLI path), build a temp store, ingest the
    golden corpus with the free stack pinned, compute the report, and tear the
    temp store down. A caller may instead pass a pre-ingested ``_Ingested``
    bundle (with its logical<->store id maps) to reuse it.

    Report keys::

        {
          "retrieval": {"hit_rate", "mrr", "ndcg", "k"},
          "groundedness": float,
          "entity_f1": float,
          "der": float,
          "contradiction": {"precision", "recall", "f1"},
        }
    """
    golden_dir = Path(golden_dir)
    qa = _load_json(golden_dir / "qa.json")
    gold_entities = _load_json(golden_dir / "entities.json")
    gold_speakers = _load_json(golden_dir / "speakers.json")
    gold_contradictions = _load_json(golden_dir / "contradictions.json")

    config = config or FREE_CONFIG
    if store is not None:
        return _compute_report(store, qa, gold_entities, gold_speakers,
                               gold_contradictions, k=k, config=config)

    with tempfile.TemporaryDirectory(prefix="memovox-eval-") as tmp:
        ing = _ingest_golden(golden_dir, tmp, config)
        return _compute_report(ing, qa, gold_entities, gold_speakers,
                               gold_contradictions, k=k, config=config)


# Free-path retrieval parity (M0.2 W2). A fixed query set whose vector+lexical
# top-k is recorded (as path-independent LOGICAL moment ids) in
# eval/golden/parity.json; any reordering by a later refactor (W1) or the
# cosine->dot optimization (W3) trips this tripwire. RRF keys off rank, so rank
# stability is the real invariant — we compare the full ordered id list.
_PARITY_K = 5


def _parity_queries() -> List[str]:
    return [item["q"] for item in _load_json(GOLDEN_DIR / "qa.json")]


def _logicalize(ing: _Ingested, moment_id: str) -> str:
    vid, sep, rest = moment_id.partition("#")
    return f"{ing.store_to_logical.get(vid, vid)}{sep}{rest}"


def _compute_parity_results(ing: _Ingested) -> dict:
    from memovox.backends import get_embedder
    from memovox.loom.store import LoomStore

    emb = get_embedder("hashing", config=ing.mv.config)
    out: dict = {}
    with LoomStore(ing.mv.config) as store:
        for q in _parity_queries():
            qv = emb.embed_one(q)
            out[q] = {
                "vector": [_logicalize(ing, m) for m, _ in store.vector_search(qv, _PARITY_K)],
                "lexical": [_logicalize(ing, m) for m, _ in store.lexical_search(q, _PARITY_K)],
            }
    return out


def parity(live: dict, recorded: dict) -> float:
    """Fraction of queries whose live (vector, lexical) top-k exactly matches the
    recorded golden. Order-sensitive — a single reordering drops the score."""
    if not recorded:
        return 1.0
    matches = sum(1 for q, rec in recorded.items() if live.get(q) == rec)
    return matches / len(recorded)


def _parity_block(ing: _Ingested) -> dict:
    path = GOLDEN_DIR / "parity.json"
    if not path.exists():
        return {"score": 1.0, "queries": 0, "recorded": False, "mismatches": []}
    recorded = _load_json(path)
    live = _compute_parity_results(ing)
    mismatches = [q for q in recorded if live.get(q) != recorded[q]]
    return {
        "score": round(parity(live, recorded), 6),
        "queries": len(recorded),
        "recorded": True,
        "mismatches": mismatches,
    }


def _record_parity(golden_dir: Path) -> Path:
    """Record the CURRENT free-path top-k as the parity golden (reproducible)."""
    with tempfile.TemporaryDirectory() as tmp:
        ing = _ingest_golden(golden_dir, str(Path(tmp) / "store"))
        results = _compute_parity_results(ing)
    out = GOLDEN_DIR / "parity.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def _span_snapshot(ing: _Ingested) -> dict:
    """Logical claim id -> [t_start_s, t_end_s] for every committed golden claim."""
    from memovox.loom.store import LoomStore

    out: dict = {}
    with LoomStore(ing.mv.config) as store:
        for store_vid in ing.logical_to_store.values():
            for c in store.claims_for_video(store_vid, status=None):
                out[_logicalize(ing, c.claim_id)] = [round(c.t_start_s, 3), round(c.t_end_s, 3)]
    return out


def _span_unchanged_block(ing: _Ingested) -> dict:
    """M0.3 W5 hard guard: the free VTT/captions span output must be byte-identical
    to the recorded baseline (the golden VTTs carry no word tags, so word-window
    tightening runs its identity branch — any silent widening/narrowing fails CI)."""
    live = _span_snapshot(ing)
    path = GOLDEN_DIR / "span_baseline.json"
    if not path.exists():
        return {"score": 1.0, "claims": len(live), "recorded": False, "drifted": []}
    baseline = _load_json(path)
    drifted = sorted(
        {cid for cid in baseline if live.get(cid) != baseline[cid]}
        | {cid for cid in live if cid not in baseline}
    )
    score = 1.0 if live == baseline else (
        sum(1 for cid in baseline if live.get(cid) == baseline[cid]) / max(len(baseline), 1)
    )
    return {"score": round(score, 6), "claims": len(baseline), "recorded": True,
            "drifted": drifted}


def span_iou(pred, gold) -> float:
    """Interval IoU of two ``(t_start, t_end)`` spans (the unified span/citation
    metric — subsumes the ASR track's span_iou and the eval track's
    citation_accuracy). 0.0 if either span is empty/None."""
    if not pred or not gold:
        return 0.0
    p0, p1 = pred
    g0, g1 = gold
    inter = max(0.0, min(p1, g1) - max(p0, g0))
    union = max(p1, g1) - min(p0, g0)
    return inter / union if union > 0 else 0.0


def _span_accuracy(ing: _Ingested, qa: list) -> dict:
    """UNIFIED span/citation-accuracy (M1.2, UNGATED): mean interval IoU of each
    cited gold-relevant moment's displayed span vs its gold span (a QA item's
    explicit ``gold_span`` if present, else the moment's window — word-precise once
    M0.3 tightening engages). Plus the M0.3 word-tightening signal."""
    from memovox.loom.store import LoomStore

    store_vids = list(ing.logical_to_store.values())
    ious = []
    with LoomStore(ing.mv.config) as store:
        moment_span = {}
        for vid in store_vids:
            for m in store.moments_for_video(vid):
                moment_span[m.moment_id] = (m.t_start_s, m.t_end_s)
        for item in qa:
            gold_ids = _relevant_moment_ids(store, store_vids,
                                            item.get("relevant_moment_substrings", []))
            if not gold_ids:
                continue
            ans = ing.mv.ask(item["q"])
            for c in ans.citations:
                if c.moment_id in gold_ids:
                    gold_span = item.get("gold_span") or moment_span.get(c.moment_id)
                    ious.append(span_iou((c.t_start_s, c.t_end_s), tuple(gold_span)))
                    break
    mean_iou = round(sum(ious) / len(ious), 6) if ious else None
    block = {"mean_iou": mean_iou, "items_scored": len(ious)}
    block.update(_word_tightening_signal())
    return block


def _word_tightening_signal() -> dict:
    """M0.3 word-precision signal: fraction of a word-bearing fixture's claims whose
    span was tightened strictly inside its source cue. ``None`` if absent."""
    fixture = _REPO_ROOT / "eval" / "fixtures" / "words_clip.json"
    if not fixture.exists():
        return {"tightened_fraction": None, "claims": 0}
    from memovox import pipeline
    from memovox.config import Config, Settings
    from memovox.loom.store import LoomStore

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(store=str(Path(tmp) / "store"), settings=Settings(**_FREE_BACKENDS)).ensure()
        rep = pipeline.ingest(cfg, str(fixture), source_url="https://x/words")
        with LoomStore(cfg) as store:
            claims = store.claims_for_video(rep.video_id)
            moments = {m.moment_id: m for m in store.moments_for_video(rep.video_id)}
            tightened = 0
            for c in claims:
                m = moments.get(c.moment_id)
                if m and (c.t_start_s > m.t_start_s or c.t_end_s < m.t_end_s):
                    tightened += 1
    return {"tightened_fraction": round(tightened / len(claims), 6) if claims else None,
            "claims": len(claims)}


def _record_span_baseline(golden_dir: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        ing = _ingest_golden(golden_dir, str(Path(tmp) / "store"))
        snap = _span_snapshot(ing)
    out = GOLDEN_DIR / "span_baseline.json"
    out.write_text(json.dumps(snap, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                   encoding="utf-8")
    return out


def _multimodal_metrics() -> dict:
    """UNGATED (M1.1): transcript-only vs tri-modal hit_rate on an on-screen-only
    item. With the VISUAL leg ON, a moment whose answer lives only in its visual
    signature is retrieved; with it OFF it is missed. Gated in M1.2 at >=3 items."""
    fixture = GOLDEN_DIR / "visual.json"
    vtt = GOLDEN_DIR / "talk_vis.en.vtt"
    if not fixture.exists() or not vtt.exists():
        return {"transcript_only": None, "tri_modal": None, "delta": None}

    from memovox import pipeline
    from memovox.augur.retrieve import retrieve
    from memovox.backends import get_embedder
    from memovox.config import Config, Settings
    from memovox.loom.store import LoomStore
    from memovox.tessera import VisualEvent, VisualResult

    spec = _load_json(fixture)
    vr = VisualResult(available=True, n_frames=1, n_scenes=1,
                      events=[VisualEvent(**spec["visual_event"])])
    top_k = int(spec.get("top_k", 5))
    qa = spec["qa"]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(store=str(Path(tmp) / "store"), settings=Settings(
            **_FREE_BACKENDS, top_k=top_k)).ensure()
        rep = pipeline.ingest(cfg, str(vtt), source_url=spec["source_url"], visual_result=vr)
        emb = get_embedder("hashing", config=cfg)
        with LoomStore(cfg) as store:
            # the on-screen-only TARGET is the moment carrying a visual vector.
            target = {r["moment_id"] for r in store.conn.execute(
                "SELECT v.moment_id AS moment_id FROM visual_vectors v "
                "JOIN moments m ON m.moment_id = v.moment_id WHERE m.video_id = ?",
                (rep.video_id,)).fetchall()}
            transcript_only = [m for m, _ in retrieve(store, qa["q"], embedder=emb,
                                                      settings=cfg.settings)]
            tri = [m for m, _ in retrieve(store, qa["q"], embedder=emb, settings=cfg.settings,
                                          use_visual=True, visual_query_vec=qa["visual_query"])]
    hit_t = 1.0 if (target & set(transcript_only)) else 0.0
    hit_tri = 1.0 if (target & set(tri)) else 0.0
    return {"transcript_only": hit_t, "tri_modal": hit_tri, "delta": round(hit_tri - hit_t, 6)}


def _incremental_equivalence() -> float:
    """1.0 iff incremental consolidation (batched, watermark) produces the IDENTICAL
    CONTRADICTS/SUPPORTS graph as a single full pass on the golden corpus (M0.2)."""
    from memovox import pipeline
    from memovox.backends import get_nli
    from memovox.config import Config, Settings
    from memovox.loom.consolidate import consolidate
    from memovox.loom.store import LoomStore

    golden = _corpus_vtts()
    if len(golden) < 2:
        return 1.0

    def _edges(store) -> set:
        s = set()
        for rel in ("CONTRADICTS", "SUPPORTS"):
            for e in store.edges(rel=rel):
                s.add((e["src"], e["rel"], e["dst"], e["video_id"]))
        return s

    with tempfile.TemporaryDirectory() as tmp:
        full = Config(store=str(Path(tmp) / "full"), settings=Settings(**_FREE_BACKENDS)).ensure()
        for g in golden:
            pipeline.ingest(full, str(g), source_url=f"https://x/{g.stem}")
        with LoomStore(full) as store:
            consolidate(store, nli=get_nli("lexical", config=full), since_watermark=0)
            full_edges = _edges(store)

        inc = Config(store=str(Path(tmp) / "inc"), settings=Settings(**_FREE_BACKENDS)).ensure()
        for g in golden:
            pipeline.ingest(inc, str(g), source_url=f"https://x/{g.stem}")
            with LoomStore(inc) as store:
                consolidate(store, nli=get_nli("lexical", config=inc))
        with LoomStore(inc) as store:
            inc_edges = _edges(store)

    return 1.0 if inc_edges == full_edges else 0.0


def _observability_metrics(ing: _Ingested) -> dict:
    """Collect the corpus-size-INDEPENDENT structural facts the M0.1 spine emits.

    Re-ingests one golden video under the pinned free stack with an explicit
    Tracer (so every stage span is inspectable), runs an ask + a forced-small
    cap, and returns booleans/counts only — never wall-clock magnitudes (those
    are machine-dependent and must never be thresholded). The block is UNGATED
    (discipline (a)); only these structural invariants are asserted in tests.
    """
    from memovox import Memovox, augur, pipeline
    from memovox.backends import get_embedder, get_nli
    from memovox.loom.consolidate import find_contradictions
    from memovox.loom.store import LoomStore
    from memovox.observe import Span, Tracer

    golden = _corpus_vtts()
    expected = ("asr", "visual", "moments", "embed", "claims", "resolve", "digest")
    if not golden:
        return {"stages_present": [], "all_status_ok": False, "wall_ms_nonneg": False,
                "counters_reconcile": False, "ask_stages": [], "forced_cap_dropped": 0,
                "ok": False}

    with tempfile.TemporaryDirectory() as tmp:
        mv = Memovox(store=str(Path(tmp) / "store"), **_FREE_BACKENDS)
        tracer = Tracer("ingest")
        pipeline.ingest(mv.config, str(golden[0]), settings=mv.settings, tracer=tracer)
        spans = {s.stage: s for s in tracer.spans}
        all_ok = all(s.status == "ok" for s in tracer.spans)
        wall_nonneg = all(s.wall_ms >= 0.0 for s in tracer.spans)
        claims = spans.get("claims")
        reconciles = bool(claims) and (
            claims.counters.get("committed", 0) + claims.counters.get("unsupported", 0)
            == claims.counters.get("claims", 0)
        )

        ask_tracer = Tracer("ask")
        forced = None
        with LoomStore(mv.config) as store:
            emb = get_embedder("hashing", config=mv.config)
            augur.ask(store, "scaling laws", embedder=emb, settings=mv.settings,
                      tracer=ask_tracer)
            # force the consolidation cap small so a drop is guaranteed to surface
            sp = Span(stage="contradictions")
            find_contradictions(store, nli=get_nli("lexical", config=mv.config),
                                max_claims=1, write_edges=False, span=sp)
            forced = next((c for c in sp.caps if c["name"] == "max_claims"), None)

    ask_stages = sorted({s.stage for s in ask_tracer.spans})
    forced_dropped = int((forced or {}).get("dropped", 0))
    ok = bool(
        all(stage in spans for stage in expected)
        and all_ok and wall_nonneg and reconciles
        and "retrieve" in ask_stages and "synthesize" in ask_stages
        and forced_dropped > 0
    )
    return {
        "stages_present": sorted(spans),
        "all_status_ok": all_ok,
        "wall_ms_nonneg": wall_nonneg,
        "counters_reconcile": reconciles,
        "ask_stages": ask_stages,
        "forced_cap_dropped": forced_dropped,
        "ok": ok,
    }


def _topic_clusters(ing: _Ingested, gold_topics: dict):
    """Gold + predicted topic clusters (members are logical moment ids), reading
    the PERSISTED ``Moment.topic_id`` (never re-running induce_topics) — a real
    regression guard for topic induction, mirroring _entity_clusters."""
    from collections import defaultdict

    from memovox.loom.store import LoomStore

    gold_clusters = [set(c) for c in gold_topics.get("clusters", [])]
    atoms = [a for c in gold_clusters for a in c]
    pred_groups: Dict[str, Set[str]] = defaultdict(set)
    with LoomStore(ing.mv.config) as store:
        for atom in atoms:
            logical, _, suffix = atom.partition("#")
            store_vid = ing.logical_to_store.get(logical)
            label = f"__no_topic__:{atom}"  # singleton unless a topic_id is persisted
            if store_vid:
                m = store.get_moment(f"{store_vid}#{suffix}")
                if m and m.topic_id:
                    label = m.topic_id
            pred_groups[label].add(atom)
    return [v for v in pred_groups.values()], gold_clusters


def _keyframe_efficiency() -> dict:
    """UNGATED (M1.1/M1.2): adaptive info-gain select_keyframes vs a uniform stride
    on a synthetic static+dense scene — adaptive keeps FEWER frames (ratio < 1)."""
    from memovox.tessera.frames import FrameSig
    from memovox.tessera.keyframes import select_keyframes
    from memovox.tessera.scenes import Scene

    # 16 identical (static) frames + 4 alternating (slide-dense) frames.
    frames = [FrameSig(t=float(i), vec=[0.5, 0.5, 0.5, 0.5]) for i in range(16)]
    frames += [FrameSig(t=float(16 + i), vec=[float(i % 2)] * 4) for i in range(4)]
    scenes = [Scene(index=0, start_idx=0, end_idx=19, t_start=0.0, t_end=19.0)]
    adaptive = len(select_keyframes(frames, scenes, min_gain=0.1, per_scene_cap=100))
    uniform = max(1, len(frames) // 2)  # fixed stride-2 baseline
    return {"adaptive_frames": adaptive, "uniform_frames": uniform,
            "ratio": round(adaptive / uniform, 6)}


def _claim_granularity(ing: _Ingested) -> dict:
    """UNGATED (§12 fold-in): claims-per-moment + mean salience cross-tab (the lever
    for the M-X.3 extraction-granularity knob). Read-only; crash-safe on empty."""
    from memovox.loom.store import LoomStore

    claims = moments = 0
    sal: List[float] = []
    with LoomStore(ing.mv.config) as store:
        for vid in ing.logical_to_store.values():
            moments += len(store.moments_for_video(vid))
            for c in store.claims_for_video(vid):
                claims += 1
                sal.append(c.salience)
    return {
        "claims_per_moment": round(claims / moments, 4) if moments else 0.0,
        "mean_salience": round(sum(sal) / len(sal), 4) if sal else 0.0,
        "moments": moments, "claims": claims,
    }


def clip_coverage(found_clips, gold_clip) -> float:
    """Best-match interval IoU between a gold clip span and the stitched clips
    (M2.3). ``found_clips`` are (t_start, t_end) spans; 0.0 if none match."""
    if not found_clips or not gold_clip:
        return 0.0
    return max((span_iou(c, tuple(gold_clip)) for c in found_clips), default=0.0)


def _corpus_signature(store) -> dict:
    """A COMPREHENSIVE persisted-graph fingerprint (M3.2) — equal fingerprints ⇒ the
    batch and incremental ingest paths produced the identical graph. Covers committed
    claims, entities, canonical speakers, AND every discourse/provenance edge type
    (not just CONTRADICTS), so a divergence in MENTIONS/ELABORATES/CORRECTS/SAME_AS
    between the two paths makes the equivalence gate FAIL rather than false-pass."""
    claims = sorted(c.claim_id for c in store.list_claims(status="committed"))
    entities = sorted(r["entity_id"] for r in store.conn.execute("SELECT entity_id FROM entities"))
    speakers = sorted(
        (r["canonical_id"] or r["speaker_id"])
        for r in store.conn.execute("SELECT speaker_id, canonical_id FROM speakers")
    )
    edges = {}
    for rel in ("CONTRADICTS", "SUPPORTS", "MENTIONS", "ELABORATES", "CORRECTS", "SAME_AS"):
        edges[rel] = sorted((e["src"], e["dst"], e.get("video_id")) for e in store.edges(rel=rel))
    return {"claims": claims, "entities": entities, "speakers": speakers, "edges": edges}


def _build_corpus(golden_dir: Path, store_dir: str, *, deferred: bool):
    """Ingest the golden corpus either as a batch (resolve per video, today's path)
    or incrementally (resolve_corpus=False per video + ONE resolve_corpus_pass), then
    one consolidation. Returns (signature, idempotent_reingest). M3.2."""
    from memovox import pipeline
    from memovox.loom.store import LoomStore
    from memovox.sdk import Memovox

    mv = Memovox(store=store_dir, **_FREE_BACKENDS)
    vtts = sorted(v for v in golden_dir.glob("*.en.vtt")
                  if v.name.split(".")[0] not in _NON_CORPUS_STEMS)
    for vtt in vtts:
        mv.ingest(str(vtt), resolve_corpus=not deferred)
    if deferred:
        with LoomStore(mv.config) as store:
            pipeline.resolve_corpus_pass(mv.config, store)
    mv.consolidate()
    with LoomStore(mv.config) as store:
        sig = _corpus_signature(store)
    before = len(sig["claims"])
    for vtt in vtts:  # idempotent re-ingest: a 2nd pass commits no new claims
        mv.ingest(str(vtt), resolve_corpus=not deferred)
    with LoomStore(mv.config) as store:
        after = len(_corpus_signature(store)["claims"])
    return sig, (before == after)


def _serving_metrics(golden_dir: Path) -> dict:
    """GATED (M3.3): a background-enqueued, worker-drained consolidation produces the
    SAME counts as an inline Memovox.consolidate() on the identically-ingested
    corpus — proving the async job path doesn't drift from the inline one."""
    import json as _json

    from memovox.sdk import Memovox
    from memovox.serving.jobs import JobStore, JobWorker

    def _ingest(store_dir):
        mv = Memovox(store=store_dir, **_FREE_BACKENDS)
        for vtt in sorted(golden_dir.glob("*.en.vtt")):
            if vtt.name.split(".")[0] not in _NON_CORPUS_STEMS:
                mv.ingest(str(vtt))
        return mv

    with tempfile.TemporaryDirectory(prefix="mv-inline-") as a, \
            tempfile.TemporaryDirectory(prefix="mv-bg-") as b:
        inline = _ingest(a).consolidate()
        mv_b = _ingest(b)
        jobs = JobStore(mv_b.config)
        jid = jobs.enqueue("consolidate", {})
        JobWorker(mv_b, once=True).drain()
        background = _json.loads(jobs.get_job(jid)["result_json"] or "{}")
        jobs.close()
    # Compare the COUNTS, not the volatile per-stage timing trace (wall_ms).
    drop = lambda d: {k: v for k, v in d.items() if k != "metrics"}
    return {"equivalent": drop(inline) == drop(background)}


def _incremental_metrics(golden_dir: Path) -> dict:
    """UNGATED→GATED (M3.2): batch-ingest and incremental-ingest (deferred resolve)
    produce the SAME persisted graph (⇒ same gated report), and a re-ingest of seen
    videos commits nothing (idempotent re-sync)."""
    with tempfile.TemporaryDirectory(prefix="mv-batch-") as b, \
            tempfile.TemporaryDirectory(prefix="mv-incr-") as i:
        batch_sig, batch_idem = _build_corpus(golden_dir, b, deferred=False)
        incr_sig, incr_idem = _build_corpus(golden_dir, i, deferred=True)
    return {"equivalent": batch_sig == incr_sig,
            "idempotent_resync": bool(batch_idem and incr_idem)}


def _decay_metrics() -> dict:
    """UNGATED (M3.1): decay behaviors on a SELF-CONTAINED dated fixture (kept out
    of the scored golden corpus so it cannot perturb the default gates). Two
    structural sub-metrics under decay_enabled=True:
      - recent_first_ordering: the newer source's moment outranks the older one;
      - superseded_excluded: a fully-superseded moment drops out of results.
    Deterministic; the unit tests in tests/test_decay.py are the behavioral guard."""
    from memovox.augur.retrieve import retrieve
    from memovox.backends import get_embedder
    from memovox.config import Config, Settings
    from memovox.loom import LoomStore, Moment, Video
    from memovox.loom.models import Claim

    q = "scaling laws and model performance over the years"
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(store=str(Path(tmp) / "s")).ensure()
        emb = get_embedder("hashing", config=cfg)
        with LoomStore(cfg) as store:
            for vid, date, txt in [
                ("yt:old", "2019-01-01", "scaling laws and model performance the older account"),
                ("yt:mid", "2022-01-01", "scaling laws and model performance the middle account"),
                ("yt:new", "2026-01-01", "scaling laws and model performance the recent account"),
            ]:
                store.upsert_video(Video(vid, f"https://youtu.be/{vid}", vid, published_at=date))
                m = Moment(f"{vid}#m0000", vid, 0.0, 10.0, txt, "spk", index=0)
                store.add_moment(m, emb.embed_one(m.text_for_embedding()))
            # a fully-superseded moment (all claims superseded)
            store.upsert_video(Video("yt:sup", "https://youtu.be/sup", "sup", published_at="2026-06-01"))
            ms = Moment("yt:sup#m0000", "yt:sup", 0.0, 10.0,
                        "scaling laws and model performance now outdated", "spk", index=0)
            store.add_moment(ms, emb.embed_one(ms.text_for_embedding()))
            store.add_claim(Claim("yt:sup#m0000.c0", "yt:sup#m0000", "yt:sup", "old", subject="x"))
            store.add_claim(Claim("yt:sup#m0000.c1", "yt:sup#m0000", "yt:sup", "new", subject="x"))
            store.supersede_claim("yt:sup#m0000.c0", "yt:sup#m0000.c1")
            store.conn.execute("UPDATE claims SET status='superseded' WHERE claim_id='yt:sup#m0000.c1'")
            store.conn.commit()

            on = [m for m, _ in retrieve(store, q, embedder=emb, settings=Settings(decay_enabled=True))]
            off = [m for m, _ in retrieve(store, q, embedder=emb, settings=Settings(decay_enabled=False))]
    dated = [m for m in on if m in ("yt:old#m0000", "yt:mid#m0000", "yt:new#m0000")]
    recent_first = dated == ["yt:new#m0000", "yt:mid#m0000", "yt:old#m0000"]
    superseded_excluded = ("yt:sup#m0000" in off) and ("yt:sup#m0000" not in on)
    return {"recent_first_ordering": recent_first, "superseded_excluded": superseded_excluded,
            "items": 3}


def _clip_metrics(ing: _Ingested) -> dict:
    """UNGATED until W7 (M2.3): mean best-match clip coverage over clips.json, plus
    the two structural invariants — clips are non-overlapping per video, and
    stitching is idempotent (stitch(stitch(x)) == stitch(x))."""
    from memovox.augur import stitch_clips
    from memovox.augur.types import Citation

    gold = _load_json(GOLDEN_DIR / "clips.json").get("items", []) if (GOLDEN_DIR / "clips.json").exists() else []
    coverages = []
    non_overlap = True
    idempotent = True
    for item in gold:
        store_vid = ing.logical_to_store.get(item["video"])
        ans = ing.mv.ask(item["q"])
        by_video = {}
        for c in ans.clips:
            by_video.setdefault(c.video_id, []).append((c.t_start_s, c.t_end_s))
        spans = by_video.get(store_vid, [])
        coverages.append(clip_coverage(spans, item["gold_clip"]))
        # invariant: per-video non-overlap
        for vid_spans in by_video.values():
            ordered = sorted(vid_spans)
            for (a0, a1), (b0, b1) in zip(ordered, ordered[1:]):
                if b0 < a1:
                    non_overlap = False
        # invariant: idempotence of the SPAN MATH — re-stitching reproduces the same
        # (t_start, t_end, video_id). videos={} intentionally drops deep_link/title
        # (not under test) and citation_indices are re-enumerated; only spans are
        # compared. Do NOT add deep_link/citation_indices to the comparison without
        # passing the real videos map + original indices.
        restitch = stitch_clips(
            [Citation(index=i, video_id=c.video_id, moment_id=f"{c.video_id}#r{i}",
                      t_start_s=c.t_start_s, t_end_s=c.t_end_s, title=c.title)
             for i, c in enumerate(ans.clips)],
            videos={}, merge_gap_s=ing.mv.settings.clip_merge_gap_s)
        if [(c.t_start_s, c.t_end_s, c.video_id) for c in restitch] != \
                [(c.t_start_s, c.t_end_s, c.video_id) for c in ans.clips]:
            idempotent = False
    return {
        "coverage": round(sum(coverages) / len(coverages), 6) if coverages else None,
        "items": len(coverages),
        "non_overlap": non_overlap,
        "idempotent": idempotent,
    }


def _plan_metrics(ing: _Ingested, qa) -> dict:
    """UNGATED (M2.2): for each multi-part golden item, the fraction of its
    sub-queries whose gold moment appears in the SINGLE composed answer's citations,
    averaged over multi-part items. Proves the agentic planner covers every clause."""
    from memovox.loom.store import LoomStore

    store_vids = list(ing.logical_to_store.values())
    item_scores = []
    with LoomStore(ing.mv.config) as store:
        for item in qa:
            subs = item.get("subqueries")
            if not subs:
                continue
            cited = {c.moment_id for c in ing.mv.ask(item["q"]).citations}
            per_sub = []
            for sq in subs:
                gold = _relevant_moment_ids(store, store_vids,
                                            sq.get("relevant_moment_substrings", []))
                per_sub.append((list(cited), gold))
            item_scores.append(subquery_recall(per_sub))
    return {"subquery_recall": round(sum(item_scores) / len(item_scores), 6) if item_scores else None,
            "multipart_items": len(item_scores)}


def _rerank_metrics(ing: _Ingested, qa, *, k: int) -> dict:
    """UNGATED (M2.1): mrr/ndcg of the free (identity) reranked retrieval vs the
    no-rerank baseline. With the identity default they are EQUAL (off==today) — the
    equivalence is the regression guard. A cross-encoder may improve, never degrade."""
    from memovox import augur
    from memovox.backends import get_embedder, get_reranker
    from memovox.loom.store import LoomStore

    emb = get_embedder("hashing", config=ing.mv.config)
    reranker = get_reranker(ing.mv.settings.rerank_backend, config=ing.mv.config)
    store_vids = list(ing.logical_to_store.values())
    per_no, per_re = [], []
    with LoomStore(ing.mv.config) as store:
        for item in qa:
            gold = _relevant_moment_ids(store, store_vids,
                                        item.get("relevant_moment_substrings", []))
            ids_no = [c.moment_id for c in augur.ask(
                store, item["q"], embedder=emb, settings=ing.mv.settings, reranker=None).citations]
            ids_re = [c.moment_id for c in augur.ask(
                store, item["q"], embedder=emb, settings=ing.mv.settings, reranker=reranker).citations]
            per_no.append((ids_no, gold))
            per_re.append((ids_re, gold))
    return {
        "mrr": round(mrr(per_re), 6), "ndcg": round(ndcg(per_re, k=k), 6),
        "no_rerank_mrr": round(mrr(per_no), 6), "no_rerank_ndcg": round(ndcg(per_no, k=k), 6),
    }


def _compute_report(ing: _Ingested, qa, gold_entities, gold_speakers,
                    gold_contradictions, *, k: int, config: "BackendConfig" = None) -> dict:
    from memovox.backends import get_nli
    from memovox.config import Settings

    config = config or FREE_CONFIG
    settings = ing.mv.settings if hasattr(ing.mv, "settings") else Settings()
    entail_threshold = getattr(settings, "entailment_threshold", 0.5)
    # M3.4: the report-time scorer is the INJECTED config's NLI, not a literal.
    nli = get_nli(config.nli_backend, config=ing.mv.config)

    per_query, overall_groundedness = _retrieval_and_groundedness(
        ing, qa, nli, k=k, entail_threshold=entail_threshold
    )

    pred_ent, gold_ent = _entity_clusters(ing, gold_entities)
    _, _, entity_f1 = clustering_f1(pred_ent, gold_ent)

    pred_spk, gold_spk = _speaker_clusters(ing, gold_speakers)
    _, _, der = clustering_f1(pred_spk, gold_spk)

    found_pairs, gold_pairs = _contradiction_pairs(ing, gold_contradictions)
    contradiction = contradiction_pr(found_pairs, gold_pairs)

    # Synthesis runs LAST: it reads the graph the steps above persisted and is
    # itself read-only (synthesize never writes), so it cannot perturb the
    # metrics computed before it.
    synthesis = _synthesis_metrics(ing, nli, topic="scaling laws",
                                   threshold=entail_threshold)

    # Observability is collected LAST and is read-only w.r.t. the scored store
    # (its ingest runs in an isolated temp store; its cap probe uses
    # write_edges=False), so it cannot perturb any metric computed above.
    observability = _observability_metrics(ing)
    parity_block = _parity_block(ing)
    incremental_equiv = _incremental_equivalence()
    span_unchanged = _span_unchanged_block(ing)
    span_accuracy = _span_accuracy(ing, qa)
    multimodal = _multimodal_metrics()

    gold_topics = _load_json(GOLDEN_DIR / "topics.json") if (GOLDEN_DIR / "topics.json").exists() else {}
    _, _, topic_f1 = clustering_f1(*_topic_clusters(ing, gold_topics)) if gold_topics else (0.0, 0.0, 0.0)
    keyframe_efficiency = _keyframe_efficiency()
    claim_granularity = _claim_granularity(ing)
    rerank = _rerank_metrics(ing, qa, k=k)
    plan = _plan_metrics(ing, qa)
    clip = _clip_metrics(ing)
    decay = _decay_metrics()
    incremental = _incremental_metrics(GOLDEN_DIR)
    serving = _serving_metrics(GOLDEN_DIR)

    return {
        "retrieval": {
            "hit_rate": round(hit_rate(per_query, k=k), 6),
            "mrr": round(mrr(per_query), 6),
            "ndcg": round(ndcg(per_query, k=k), 6),
            "k": k,
        },
        "groundedness": round(overall_groundedness, 6),
        "entity_f1": round(entity_f1, 6),
        "der": round(der, 6),
        "contradiction": {
            "precision": round(contradiction["precision"], 6),
            "recall": round(contradiction["recall"], 6),
            "f1": round(contradiction["f1"], 6),
        },
        "synthesis": synthesis,
        "observability": observability,  # UNGATED (discipline (a)); structural only
        "parity": parity_block,          # exact-equivalence invariant (gated)
        "incremental_equivalence": incremental_equiv,  # exact-equivalence invariant (gated)
        "span_unchanged": span_unchanged,  # M0.3 free-path span byte-identity (gated)
        "span_accuracy": span_accuracy,    # M0.3 word-precision signal (UNGATED; M1.2 gates)
        "multimodal": multimodal,          # M1.1 transcript-only vs tri-modal lift (UNGATED)
        "topic_f1": round(topic_f1, 6),    # M1.2 topic induction guard (UNGATED; thin corpus)
        "keyframe_efficiency": keyframe_efficiency,  # M1.2 adaptive vs uniform (UNGATED)
        "claim_granularity": claim_granularity,      # M1.2 §12 granularity curve (UNGATED)
        "rerank": rerank,                            # M2.1 rerank mrr/ndcg vs no-rerank (UNGATED)
        "plan": plan,                                # M2.2 agentic subquery_recall (UNGATED)
        "clip": clip,                                # M2.3 clip coverage + invariants
        "decay": decay,                              # M3.1 recency ordering + supersede demotion (GATED: exact)
        "incremental": incremental,                  # M3.2 batch==incremental + idempotent re-sync
        "serving": serving,                          # M3.3 background==inline consolidate (GATED: exact)
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

# Gates that are meaningful TODAY (W0.3): retrieval, groundedness, and the
# (already-working) free contradiction path. entity_f1/der are deliberately
# UNGATED — they're legitimately 0.0 until cross-corpus entity (W2.3) and
# speaker (W4.1) resolution land, so gating them now would fail CI spuriously.
_HIT_RATE_GATE = 0.6
_GROUNDEDNESS_GATE = 0.8
_CONTRADICTION_F1_GATE = 0.5
# Synthesis groundedness is robust (the extractive synthesizer cites every
# sentence from its own span), so it is gated. topic_f1 is deliberately NOT a
# golden gate: topic-induction quality over a 2-talk corpus is too small to be a
# stable signal — it is covered by tests/test_topics.py instead.
_SYNTHESIS_GROUNDEDNESS_GATE = 0.8
# M1.2 W9: entity_f1/der promoted to gates after talk_c verification — both read
# 1.0 on the 3-talk corpus (stable across runs), so a conservative 0.5 floor
# (matching contradiction.f1) is non-flaky and catches a resolution regression.
_ENTITY_F1_GATE = 0.5
_DER_GATE = 0.5
# M2.3 W7: stitched-clip coverage gated now that >= 3 stable golden clips exist
# (5 items; deterministic mean ~0.37). A non-regression floor — stitching may
# tighten coverage, never drop it below 0.3.
_CLIP_COVERAGE_GATE = 0.3
# M-hardening: citation/span accuracy (span_accuracy.mean_iou) is the provenance
# guarantee — promoted from UNGATED to a CI gate. qa.json has 7 items (>= 3, so
# gate-eligible); the deterministic free-path value is 1.0 today. This is a
# NON-REGRESSION FLOOR: citation windows may tighten (M0.3) but must not drift
# wholesale off their gold spans.
_SPAN_IOU_GATE = 0.75


def _print_report(report: dict) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _assert_default_off_flags() -> None:
    """Fail loudly if a pinned default-OFF flag drifted (review discipline (b))."""
    from memovox.config import Settings

    s = Settings()
    drift = {k: getattr(s, k) for k, v in _DEFAULT_OFF_FLAGS.items() if getattr(s, k) != v}
    if drift:
        raise SystemExit(
            f"eval-settings snapshot drift: default-OFF flags changed {drift}; "
            "update _DEFAULT_OFF_FLAGS deliberately and re-baseline the gates."
        )


# Thin-fixture gating discipline (M-X W2). A metric may sit in _check_thresholds
# only if it is an exact-equivalence INVARIANT (correctness, gates at any corpus
# size) OR a STATISTICAL metric backed by >=3 golden items (else a hard gate on a
# 2-3 item fixture is noise, not signal). Every gated key is declared here; the
# meta-test (tests/test_eval.py) asserts _check_thresholds gates exactly this set,
# so a future premature gate forces a deliberate choice.
_MIN_FIXTURES_TO_GATE = 3
_GATE_DECLARATIONS = {
    "retrieval.hit_rate": {"kind": "statistical", "fixture": "qa.json"},
    "groundedness": {"kind": "statistical", "fixture": "qa.json"},
    "span_accuracy.mean_iou": {"kind": "statistical", "fixture": "qa.json"},
    # contradiction/synthesis ride the single seeded cross-corpus disagreement —
    # grandfathered thin (pre-Phase-4 baseline); talk_c (M1.2) brings it to >=3.
    "contradiction.f1": {"kind": "statistical", "fixture": "contradictions.json",
                         "grandfathered_thin": True},
    "synthesis.groundedness": {"kind": "statistical", "fixture": "contradictions.json",
                               "grandfathered_thin": True},
    # M1.2 W9: gated at 1.0 on the 3-talk corpus; entities/speakers gold are still
    # thin, so grandfathered with a conservative floor (a resolution regression to
    # 0.0 still trips the gate, which is the point).
    "entity_f1": {"kind": "statistical", "fixture": "entities.json",
                  "grandfathered_thin": True},
    "der": {"kind": "statistical", "fixture": "speakers.json", "grandfathered_thin": True},
    # M2.3 W7: genuinely eligible — clips.json has 5 stable items (>= 3); gated at 0.3.
    "clip.coverage": {"kind": "statistical", "fixture": "clips.json"},
    # M3.1: deterministic structural invariants on a synthetic dated fixture (exact).
    "decay.recent_first_ordering": {"kind": "exact"},
    "decay.superseded_excluded": {"kind": "exact"},
    # M3.2: ingest-deferral equivalence + idempotent re-sync (exact invariants).
    "incremental.equivalent": {"kind": "exact"},
    "incremental.idempotent_resync": {"kind": "exact"},
    # M3.3: background==inline consolidate (exact invariant).
    "serving.equivalent": {"kind": "exact"},
    "parity": {"kind": "exact"},
    "incremental_equivalence": {"kind": "exact"},
    "span_unchanged": {"kind": "exact"},
}


def _fixture_count(name: str) -> int:
    path = GOLDEN_DIR / name
    if not path.exists():
        return 0
    data = _load_json(path)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return len(data["items"])  # {"items": [...]} fixtures (e.g. clips.json)
    return 0


def _gate_eligible(decl: dict) -> bool:
    """Whether a metric may be hard-gated under the thin-fixture discipline."""
    if decl.get("kind") == "exact":
        return True  # exact-equivalence invariant — corpus-size independent
    if decl.get("grandfathered_thin"):
        return True  # pre-existing baseline, accepted as thin
    return _fixture_count(decl.get("fixture", "")) >= _MIN_FIXTURES_TO_GATE


def _check_thresholds(report: dict) -> List[str]:
    failures: List[str] = []
    hr = report["retrieval"]["hit_rate"]
    gr = report["groundedness"]
    cf1 = report["contradiction"]["f1"]
    sg = report.get("synthesis", {}).get("groundedness", 0.0)
    if hr < _HIT_RATE_GATE:
        failures.append(f"retrieval.hit_rate {hr:.3f} < {_HIT_RATE_GATE}")
    if gr < _GROUNDEDNESS_GATE:
        failures.append(f"groundedness {gr:.3f} < {_GROUNDEDNESS_GATE}")
    # M-hardening: citation/span accuracy (the provenance guarantee). None when no
    # gold-relevant citation was scored — skip; else enforce the non-regression floor.
    siou = (report.get("span_accuracy") or {}).get("mean_iou")
    if siou is not None and siou < _SPAN_IOU_GATE:
        failures.append(f"span_accuracy.mean_iou {siou:.3f} < {_SPAN_IOU_GATE}")
    if cf1 < _CONTRADICTION_F1_GATE:
        failures.append(f"contradiction.f1 {cf1:.3f} < {_CONTRADICTION_F1_GATE}")
    if sg < _SYNTHESIS_GROUNDEDNESS_GATE:
        failures.append(f"synthesis.groundedness {sg:.3f} < {_SYNTHESIS_GROUNDEDNESS_GATE}")
    # M1.2 W9: entity_f1/der gated after talk_c verification (1.0 on 3 talks).
    ef1 = report.get("entity_f1", 1.0)
    if ef1 < _ENTITY_F1_GATE:
        failures.append(f"entity_f1 {ef1:.3f} < {_ENTITY_F1_GATE}")
    der = report.get("der", 1.0)
    if der < _DER_GATE:
        failures.append(f"der {der:.3f} < {_DER_GATE}")
    # M2.3 W7: stitched-clip coverage (only when the block is present).
    clip_cov = (report.get("clip") or {}).get("coverage")
    if clip_cov is not None and clip_cov < _CLIP_COVERAGE_GATE:
        failures.append(f"clip.coverage {clip_cov:.3f} < {_CLIP_COVERAGE_GATE}")
    # M3.1: decay structural invariants (exact — deterministic on the dated fixture).
    decay = report.get("decay") or {}
    if "recent_first_ordering" in decay and not decay["recent_first_ordering"]:
        failures.append("decay.recent_first_ordering is False")
    if "superseded_excluded" in decay and not decay["superseded_excluded"]:
        failures.append("decay.superseded_excluded is False")
    # M3.2: incremental-ingest equivalence + idempotent re-sync (exact invariants).
    incr = report.get("incremental") or {}
    if "equivalent" in incr and not incr["equivalent"]:
        failures.append("incremental.equivalent is False (batch != incremental)")
    if "idempotent_resync" in incr and not incr["idempotent_resync"]:
        failures.append("incremental.idempotent_resync is False")
    # M3.3: background-drained consolidate == inline consolidate (exact invariant).
    serving = report.get("serving") or {}
    if "equivalent" in serving and not serving["equivalent"]:
        failures.append("serving.equivalent is False (background != inline consolidate)")
    # M0.2 exact-equivalence invariants — gated at 1.0 (correctness, not statistics).
    pscore = report.get("parity", {}).get("score", 1.0)
    if pscore < 1.0:
        failures.append(f"parity {pscore:.3f} < 1.0 (free-path retrieval moved)")
    inc = report.get("incremental_equivalence", 1.0)
    if inc < 1.0:
        failures.append(f"incremental_equivalence {inc:.3f} < 1.0 (incremental != full)")
    span = report.get("span_unchanged", {}).get("score", 1.0)
    if span < 1.0:
        failures.append(f"span_unchanged {span:.3f} < 1.0 (free-path claim spans drifted)")
    return failures


#: The metrics the benchmark ranks across configs (name -> extractor).
_RANK_METRICS = [
    ("hit_rate", lambda r: r["retrieval"]["hit_rate"]),
    ("mrr", lambda r: r["retrieval"]["mrr"]),
    ("ndcg", lambda r: r["retrieval"]["ndcg"]),
    ("groundedness", lambda r: r["groundedness"]),
    ("contradiction.f1", lambda r: r["contradiction"]["f1"]),
    ("synthesis.groundedness", lambda r: r["synthesis"]["groundedness"]),
]


def run_benchmark(golden_dir=GOLDEN_DIR, configs=None, k: int = DEFAULT_K):
    """Run the SAME run_eval() metric path once per available config (each in its own
    fresh temp store), returning an ordered ``[(config_name, report), ...]`` (FREE
    first). The FREE row's report is metric-identical to bare run_eval(). M3.4."""
    configs = configs if configs is not None else available_configs(golden_dir)
    return [(c.name, run_eval(golden_dir, config=c, k=k)) for c in configs]


def _benchmark_rows(results) -> dict:
    return {name: {m: getter(rep) for m, getter in _RANK_METRICS} for name, rep in results}


def _benchmark_json(results) -> str:
    return json.dumps({"configs": [name for name, _ in results],
                       "metrics": _benchmark_rows(results)},
                      sort_keys=True, indent=2, ensure_ascii=False)


def _print_benchmark_table(results) -> None:
    rows = _benchmark_rows(results)
    names = [name for name, _ in results]
    # Per-metric best: highest value, deterministic tie-break by the FIRST config in
    # the (free-first, name-sorted) order.
    best = {m: max(names, key=lambda n: rows[n][m]) for m, _ in _RANK_METRICS}
    width = max((len(n) for n in names), default=4) + 1
    header = "config".ljust(width) + "".join(f"{m:>22}" for m, _ in _RANK_METRICS)
    print(header)
    print("-" * len(header))
    for name in names:
        line = name.ljust(width)
        for m, _ in _RANK_METRICS:
            mark = "*" if best[m] == name else " "
            line += f"{rows[name][m]:>21.4f}{mark}"
        print(line)
    for cfg_name, reason in unrankable_configs():
        print(f"  (unrankable: {cfg_name} — {reason})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m eval.harness",
        description="Run the memovox eval harness over the golden corpus.",
    )
    parser.add_argument(
        "--golden-dir", default=str(GOLDEN_DIR),
        help="Directory of the golden corpus (default: eval/golden).",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Retrieval cutoff for hit_rate/nDCG (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--assert-thresholds", action="store_true",
        help="Exit non-zero if the retrieval.hit_rate, groundedness, "
             "contradiction.f1, or synthesis.groundedness gate fails.",
    )
    parser.add_argument(
        "--record-parity", action="store_true",
        help="(maintenance) re-record eval/golden/parity.json from the CURRENT "
             "free-path top-k, then exit. Use only after a deliberate, reviewed change.",
    )
    parser.add_argument(
        "--record-spans", action="store_true",
        help="(maintenance) re-record eval/golden/span_baseline.json from the CURRENT "
             "free-path claim spans, then exit. Use only after a reviewed change.",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="A/B-rank the available BackendConfigs (auto-shrinks to FREE on a bare "
             "machine). Prints a deterministic ranking table; add --json for JSON.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="With --benchmark: emit the ranking table as machine-readable JSON.",
    )
    parser.add_argument(
        "--assert-no-regression", action="store_true",
        help="With --benchmark: exit non-zero if the FREE row fails the existing "
             "thresholds (the benchmark-mode equivalent of --assert-thresholds).",
    )
    args = parser.parse_args(argv)

    if args.record_parity:
        path = _record_parity(Path(args.golden_dir))
        print(f"recorded parity golden -> {path}")
        return 0

    if args.record_spans:
        path = _record_span_baseline(Path(args.golden_dir))
        print(f"recorded span baseline -> {path}")
        return 0

    _assert_default_off_flags()

    if args.benchmark:
        results = run_benchmark(args.golden_dir, k=args.k)
        if args.json:
            print(_benchmark_json(results))
        else:
            _print_benchmark_table(results)
        if args.assert_no_regression:
            free = dict(results).get("free")
            if free is None:  # available_configs always prepends FREE; defensive
                print("\nNo FREE row to gate.", file=sys.stderr)
                return 1
            failures = _check_thresholds(free)
            if failures:
                print("\nFREE-ROW REGRESSION:", file=sys.stderr)
                for f in failures:
                    print(f"  - {f}", file=sys.stderr)
                return 1
            print("\nFREE row clears all thresholds.", file=sys.stderr)
        return 0

    report = run_eval(args.golden_dir, k=args.k)
    _print_report(report)

    if args.assert_thresholds:
        failures = _check_thresholds(report)
        if failures:
            print("\nGATE FAILED:", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print("\nAll gates passed.", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
