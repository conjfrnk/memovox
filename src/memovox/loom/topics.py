"""Topic induction + ABOUT edges (Phase 3, spec §4.7 / §6).

Cluster Moments into emergent Topics and maintain a topic map of the whole
library: a deterministic ``topic:<slug>`` node per cluster, each member Moment's
``topic_id``, and a provenanced ``(Moment)-[:ABOUT]->(Topic)`` edge.

Free + deterministic by construction: it clusters over the **persisted text
vectors** (the hashing embedder on the free path) with a greedy single pass in
sorted ``moment_id`` order — the same shape as :func:`cluster_by_voiceprint`
(W4.2). No re-embedding, no model, and re-running over an unchanged corpus is a
graph no-op (deterministic ids; ABOUT edges are ``UNIQUE``-guarded;
``upsert_topic`` is INSERT OR REPLACE).
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Tuple

from ..config import Settings
from ..util import short_hash, slugify
from ..vectormath import cosine
from .consolidate import _content_tokens
from .models import Topic


def _topic_label(texts: Sequence[str], *, top: int = 3) -> Optional[str]:
    """A human label from a cluster's most frequent content tokens.

    Stopword-stripped (via :func:`_content_tokens`); ranked by ``(-count, token)``
    so the label is a deterministic function of the clustered text. Returns
    ``None`` when the cluster carries no content tokens (e.g. all stopwords).
    """
    counts: Counter = Counter()
    for text in texts:
        counts.update(_content_tokens(text))
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return " ".join(tok for tok, _ in ranked)


#: Cap on cluster representatives scanned per moment. The greedy pass compares each moment
#: against every existing cluster rep, so a topically-DIVERSE corpus (reps grows ~O(n)) makes
#: induce_topics O(n_moments x n_clusters) — and the moment count is attacker-controlled by a
#: single uncapped ingest, so a 60k-moment transcript could pin the (offline) consolidate worker
#: for hours. Scanning only the first this-many reps bounds per-moment work to O(cap x dim).
#: It is OUTPUT-IDENTICAL for any corpus whose total distinct-topic count stays under the cap
#: (reps never exceeds it, so the break never fires) — i.e. every realistic library; only a
#: pathologically diverse corpus past the cap clusters approximately (still deterministic).
#: Topic induction is a non-critical browsing aid (citations/grounding don't depend on it), so
#: a generous-but-bounding ceiling is the right trade: 512 distinct topics far exceeds the eval
#: corpus and a realistic library, while bounding even an 8000-moment all-distinct worst case to
#: ~6s (vs ~3 min unbounded) and a 60k-moment adversarial ingest to seconds (vs hours).
_MAX_TOPIC_REPS = 512


def _greedy_clusters(
    pairs: Sequence[Tuple[str, Sequence[float]]], *, threshold: float
) -> List[List[str]]:
    """Greedy single-pass cosine clustering of (id, vector) pairs.

    Each id joins the FIRST existing cluster whose representative (first/lowest-id
    member) vector is ``>= threshold``, else opens its own. Dimension mismatches
    are a non-match (never fed to cosine, which would compare only the shared
    prefix). Iteration follows the input order, which the caller fixes (sorted by
    id), so the partition is deterministic. The rep scan is bounded by
    ``_MAX_TOPIC_REPS`` to keep the pass near-linear (see that constant).
    """
    clusters: List[List[str]] = []
    reps: List[Sequence[float]] = []
    for mid, vec in pairs:
        placed = False
        for i, rep in enumerate(reps):
            if i >= _MAX_TOPIC_REPS:
                break  # bound per-moment comparisons (transparent below the cap)
            if len(vec) != len(rep):
                continue
            if cosine(vec, rep) >= threshold:
                clusters[i].append(mid)
                placed = True
                break
        if not placed:
            clusters.append([mid])
            reps.append(vec)
    return clusters


def induce_topics(store, *, settings: Optional[Settings] = None) -> List[Topic]:
    """Cluster the corpus's Moments into Topics and persist the topic map.

    Returns the induced topics. For each cluster of at least
    ``settings.topic_min_size`` Moments: upsert a ``Topic`` (with its
    ``moment_count``), stamp each member's ``topic_id``, and emit a provenanced
    ``(Moment)-[:ABOUT]->(Topic)`` edge. Clusters below the floor are left
    untouched (their Moments keep whatever ``topic_id`` they already had).
    """
    settings = settings or Settings()
    pairs = store.moment_vectors()
    clusters = _greedy_clusters(pairs, threshold=settings.topic_similarity)

    topics: List[Topic] = []
    for members in clusters:
        if len(members) < settings.topic_min_size:
            continue
        moments = store.get_moments(members)
        label = _topic_label([m.text_for_embedding() for m in moments])
        if label:
            topic_id = "topic:" + slugify(label, default="topic")
        else:
            # No content tokens to name it — fall back to a stable id over the
            # member set so the topic is still inspectable and idempotent.
            topic_id = "topic:" + short_hash("|".join(sorted(members)))
            label = topic_id
        topic = Topic(topic_id=topic_id, label=label, moment_count=len(moments))
        store.upsert_topic(topic)
        topics.append(topic)
        for m in moments:
            store.clear_about_edges(m.moment_id)  # drop stale ABOUT edges to old topics
            store.set_moment_topic(m.moment_id, topic_id)
            store.add_edge(
                m.moment_id, "ABOUT", topic_id,
                src_type="Moment", dst_type="Topic", video_id=m.video_id,
                t_start_s=m.t_start_s, t_end_s=m.t_end_s, modality=m.modality,
            )
    return topics
