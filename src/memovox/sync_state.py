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
    except (ValueError, TypeError) as exc:
        import sys
        print(f"memovox: corrupt sync cursor for {url} ({exc}); treating as empty "
              "(the source will be re-checked).", file=sys.stderr)
        return set()


def mark_seen(store, url: str, video_id: str) -> None:
    """Record ``video_id`` as seen for this source (additive, idempotent, ATOMIC).

    Delegates to the store's write-locked read-modify-write so two concurrent cursor
    writers (a direct ``mv.sync()`` racing a worker ``sync`` job) cannot lose an id —
    the old get/modify/set here was a non-atomic whole-value overwrite (last writer
    wins, dropping the other's id -> a needless re-ingest on the next sync)."""
    store.append_meta_json_id(_meta_key(url), video_id)


def clear(store, url: str) -> None:
    """Forget a source's cursor (a re-subscribe / --force re-ingests its catalog)."""
    store.set_meta(_meta_key(url), json.dumps([]))
