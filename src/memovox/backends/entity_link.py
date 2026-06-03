"""Entity-linker backends — canonicalize a surface mention to a graph entity.

W2.3 consumes this to turn the raw mentions from W2.1 into canonical graph
entities. The default :class:`NullLinker` is free, offline, and fully
deterministic: it derives a stable ``ent:<slug>`` id from the surface form.
:class:`WikidataLinker` is an optional online upgrade that *adds* a
``wikidata_qid`` (and may set the canonical label) but — crucially for
reproducible eval — never changes the slug-derived ``entity_id``, so the graph
topology does not depend on connectivity. Any network/parse error degrades
gracefully to the exact :class:`NullLinker` output.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..util import slugify
from .base import Backend


@dataclass
class Canonical:
    """A canonicalized entity: a stable slug id, a display name, an optional QID."""

    entity_id: str
    name: str
    wikidata_qid: Optional[str] = None


class EntityLinker(Backend):
    """Map a surface mention to a canonical graph entity."""

    @abstractmethod
    def canonicalize(self, surface: str) -> Canonical:
        raise NotImplementedError


def _slug_canonical(surface: str) -> Canonical:
    """The deterministic, dependency-free canonical form (no QID)."""
    return Canonical(entity_id=f"ent:{slugify(surface)}", name=surface, wikidata_qid=None)


class NullLinker(EntityLinker):
    """Slug-based linker — the always-available, deterministic fallback."""

    name = "none"

    def canonicalize(self, surface: str) -> Canonical:
        return _slug_canonical(surface)


class WikidataLinker(EntityLinker):
    """Attach Wikidata QIDs via the public ``wbsearchentities`` API (optional).

    The ``entity_id`` is *always* the slug form (identical to :class:`NullLinker`)
    so the graph topology is reproducible regardless of connectivity; a
    successful lookup only adds ``wikidata_qid`` and the canonical label. Any
    network, timeout, parse, or empty-result condition falls back to the slug
    form, so callers never have to handle Wikidata-specific failures.
    """

    name = "wikidata"
    API = "https://www.wikidata.org/w/api.php"
    #: Wikimedia UA policy can 403 the default ``Python-urllib`` agent.
    USER_AGENT = "memovox/0.1 (https://github.com/conjfrnk/memovox; entity linking)"

    @classmethod
    def is_available(cls) -> bool:
        # Respect offline mode (hermetic tests/CI set HF_HUB_OFFLINE=1) — no probe.
        if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("MEMOVOX_OFFLINE") == "1":
            return False
        try:
            socket.create_connection(("www.wikidata.org", 443), timeout=1.5).close()
            return True
        except OSError:
            return False

    def canonicalize(self, surface: str) -> Canonical:
        fallback = _slug_canonical(surface)
        query = urllib.parse.urlencode(
            {
                "action": "wbsearchentities",
                "search": surface,
                "language": "en",
                "format": "json",
                "limit": 1,
            }
        )
        url = f"{self.API}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=3) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            socket.timeout,
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            # Any network/timeout/parse error degrades to the slug form;
            # a genuine programming bug (AttributeError/TypeError) still surfaces.
            return fallback
        hits = payload.get("search") or []
        if not hits:
            return fallback
        hit = hits[0]
        qid = hit.get("id")
        if not qid:
            return fallback
        label = hit.get("label") or surface
        # entity_id stays slug-derived (NOT from QID/label) for reproducibility.
        return Canonical(entity_id=fallback.entity_id, name=label, wikidata_qid=qid)
