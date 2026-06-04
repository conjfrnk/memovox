"""Storage-backend registry — ``get_*("auto"|"sqlite", …)`` with a free default.

Mirrors ``memovox.backends.__init__`` for the storage slots. ``"auto"`` resolves
to the always-available SQLite implementation; optional ANN / Tantivy / Kùzu
backends register here behind ``is_available()`` (opt-in, never in the CI gate).
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from ...errors import BackendUnavailable
from .ann import LanceVectorIndex, QdrantVectorIndex
from .base import GraphStore, LexicalIndex, StorageBackend, VectorIndex
from .sqlite import SqliteGraphStore, SqliteLexicalIndex, SqliteVectorIndex

__all__ = [
    "StorageBackend",
    "VectorIndex",
    "LexicalIndex",
    "GraphStore",
    "get_vector_index",
    "get_lexical_index",
    "get_graph_store",
]

_VECTOR_INDICES = {
    "sqlite": SqliteVectorIndex,
    "lance": LanceVectorIndex,      # opt-in (M0.2 W7); BackendUnavailable if absent
    "qdrant": QdrantVectorIndex,    # opt-in (M0.2 W7); BackendUnavailable if absent
}
_LEXICAL_INDICES = {"sqlite": SqliteLexicalIndex}
_GRAPH_STORES = {"sqlite": SqliteGraphStore}


def get_vector_index(name: str = "auto", *, conn: Optional[sqlite3.Connection] = None,
                     **options) -> VectorIndex:
    if name == "auto":
        name = "sqlite"
    cls = _VECTOR_INDICES.get(name)
    if cls is None or not cls.is_available():
        raise BackendUnavailable(
            f"Vector index {name!r} unavailable. Options: {list(_VECTOR_INDICES)} or 'auto'."
        )
    return cls(conn, **options)


def get_lexical_index(name: str = "auto", *, conn: Optional[sqlite3.Connection] = None,
                      fts: bool = False, **options) -> LexicalIndex:
    if name == "auto":
        name = "sqlite"
    cls = _LEXICAL_INDICES.get(name)
    if cls is None or not cls.is_available():
        raise BackendUnavailable(
            f"Lexical index {name!r} unavailable. Options: {list(_LEXICAL_INDICES)} or 'auto'."
        )
    return cls(conn, fts, **options)


def get_graph_store(name: str = "auto", *, conn: Optional[sqlite3.Connection] = None,
                    **options) -> GraphStore:
    if name == "auto":
        name = "sqlite"
    cls = _GRAPH_STORES.get(name)
    if cls is None or not cls.is_available():
        raise BackendUnavailable(
            f"Graph store {name!r} unavailable. Options: {list(_GRAPH_STORES)} or 'auto'."
        )
    return cls(conn, **options)
