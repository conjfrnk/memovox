"""Per-source subscription sync cursor (M3.2).

A persisted set of already-seen video ids per source, stored in the ``meta`` table
(``sync_state:<source_key>`` → JSON sorted id list) so a re-``sync`` skips seen ids
BEFORE any download. Thin deterministic helpers over ``store.set_meta``/``get_meta``
— no schema change. The cursor is advisory: ids are marked seen only on ingest
success, so a failed ingest is retried on the next sync.
"""

from __future__ import annotations

import json
from typing import Set

from .util import short_hash


def source_key(url: str) -> str:
    """Stable cursor key for a source URL (normalized, hashed)."""
    return short_hash((url or "").strip().lower())


def _meta_key(url: str) -> str:
    return f"sync_state:{source_key(url)}"


def seen_ids(store, url: str) -> Set[str]:
    """The set of video ids already ingested for this source (empty if unknown)."""
    raw = store.get_meta(_meta_key(url))
    if not raw:
        return set()
    try:
        return set(json.loads(raw))
    except (ValueError, TypeError):
        return set()


def mark_seen(store, url: str, video_id: str) -> None:
    """Record ``video_id`` as seen for this source (additive, idempotent)."""
    ids = seen_ids(store, url)
    if video_id in ids:
        return
    ids.add(video_id)
    store.set_meta(_meta_key(url), json.dumps(sorted(ids)))  # sorted -> stable on disk


def clear(store, url: str) -> None:
    """Forget a source's cursor (a re-subscribe / --force re-ingests its catalog)."""
    store.set_meta(_meta_key(url), json.dumps([]))
