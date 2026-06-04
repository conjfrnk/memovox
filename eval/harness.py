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
)

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
    asr_device="auto",           # M0.3: ASR device knobs pinned
    asr_compute_type="default",
    asr_allow_cpu=False,
    captions_as_prior=True,      # M0.3: §9 cost lever default pinned
)

# Settings fields deliberately NOT pinned (M-X W1): backend selectors (pinned by
# name via _FREE_BACKENDS instead) and pure numeric tuning knobs whose value does
# not toggle a feature on/off. A NEW flag must be pinned above or added here with a
# reason — the reflection completeness meta-test (tests/test_eval.py) enforces it.
_INTENTIONALLY_UNPINNED = frozenset({
    # backend selectors (pinned by name in _FREE_BACKENDS)
    "asr_backend", "embed_backend", "nli_backend", "llm_backend", "vlm_backend",
    "ocr_backend", "entity_backend", "voiceprint_backend",
    # numeric tuning knobs (not feature toggles)
    "embed_dim", "frame_sample_fps", "frame_side", "frame_max", "scene_threshold",
    "keyframe_min_gain", "keyframe_per_scene_cap", "moment_max_sec", "moment_min_sec",
    "moment_gap_sec", "boundary_similarity", "entailment_threshold", "rrf_k", "top_k",
    "contradiction_threshold", "topic_similarity", "topic_min_size", "consensus_jaccard",
})

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


def _ingest_golden(golden_dir: Path, store_dir: str) -> _Ingested:
    from memovox import Memovox

    mv = Memovox(store=store_dir, **_FREE_BACKENDS)
    logical_to_store: Dict[str, str] = {}
    for vtt in sorted(golden_dir.glob("*.en.vtt")):
        logical_id = vtt.name.split(".")[0]  # filename stem before ".en.vtt"
        report = mv.ingest(str(vtt))
        logical_to_store[logical_id] = report.video_id
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

    Runs the free consolidation path (writes CONTRADICTS edges) then reads them
    back, translating Claim src/dst -> store video_id -> logical id.

    Consolidation works today, so we do NOT swallow ``mv.contradictions()`` — a
    failure there is a real regression and must propagate. The edges read catches
    only ``sqlite3.OperationalError`` (the "no such table" absence) and warns, so
    a legitimately-missing graph degrades to 0.0 visibly rather than crashing.
    """
    from memovox.loom.store import LoomStore

    # Run consolidation so cross-corpus CONTRADICTS edges exist. Let any error
    # propagate: this is a working metric today and silent failure would falsely
    # pass the contradiction gate.
    ing.mv.contradictions()

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
             k: int = DEFAULT_K) -> dict:
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

    if store is not None:
        return _compute_report(store, qa, gold_entities, gold_speakers,
                               gold_contradictions, k=k)

    with tempfile.TemporaryDirectory(prefix="memovox-eval-") as tmp:
        ing = _ingest_golden(golden_dir, tmp)
        return _compute_report(ing, qa, gold_entities, gold_speakers,
                               gold_contradictions, k=k)


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


def _span_accuracy() -> dict:
    """UNGATED (M1.2 owns gating): on a word-bearing free-path fixture, the fraction
    of committed claims whose span was tightened strictly inside its source cue —
    the word-precision signal. ``None`` if the fixture is absent."""
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


def _incremental_equivalence() -> float:
    """1.0 iff incremental consolidation (batched, watermark) produces the IDENTICAL
    CONTRADICTS/SUPPORTS graph as a single full pass on the golden corpus (M0.2)."""
    from memovox import pipeline
    from memovox.backends import get_nli
    from memovox.config import Config, Settings
    from memovox.loom.consolidate import consolidate
    from memovox.loom.store import LoomStore

    golden = sorted(GOLDEN_DIR.glob("*.en.vtt"))
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

    golden = sorted(GOLDEN_DIR.glob("*.en.vtt"))
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


def _compute_report(ing: _Ingested, qa, gold_entities, gold_speakers,
                    gold_contradictions, *, k: int) -> dict:
    from memovox.backends import get_nli
    from memovox.config import Settings

    settings = ing.mv.settings if hasattr(ing.mv, "settings") else Settings()
    entail_threshold = getattr(settings, "entailment_threshold", 0.5)
    nli = get_nli("lexical", config=ing.mv.config)

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
    span_accuracy = _span_accuracy()

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
    # contradiction/synthesis ride the single seeded cross-corpus disagreement —
    # grandfathered thin (pre-Phase-4 baseline); talk_c (M1.2) brings it to >=3.
    "contradiction.f1": {"kind": "statistical", "fixture": "contradictions.json",
                         "grandfathered_thin": True},
    "synthesis.groundedness": {"kind": "statistical", "fixture": "contradictions.json",
                               "grandfathered_thin": True},
    "parity": {"kind": "exact"},
    "incremental_equivalence": {"kind": "exact"},
    "span_unchanged": {"kind": "exact"},
}


def _fixture_count(name: str) -> int:
    path = GOLDEN_DIR / name
    if not path.exists():
        return 0
    data = _load_json(path)
    return len(data) if isinstance(data, list) else 0


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
    if cf1 < _CONTRADICTION_F1_GATE:
        failures.append(f"contradiction.f1 {cf1:.3f} < {_CONTRADICTION_F1_GATE}")
    if sg < _SYNTHESIS_GROUNDEDNESS_GATE:
        failures.append(f"synthesis.groundedness {sg:.3f} < {_SYNTHESIS_GROUNDEDNESS_GATE}")
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
