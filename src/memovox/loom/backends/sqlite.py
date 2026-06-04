"""SQLite-backed storage indices — the always-available free defaults.

The SQL bodies here are lifted **verbatim** from ``LoomStore`` (the no-op refactor
in M0.2 W1) so the four-index store keeps a single embedded, inspectable DB. Each
index holds a reference to the shared ``sqlite3.Connection`` owned by ``LoomStore``;
writes that are part of a larger transaction (``add``) do NOT commit, while the
graph ``add_edge`` commits exactly as the legacy method did.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from ...util import tokenize
from ...vectormath import dot, norm, normalize, pack_floats, unpack_floats
from .base import GraphStore, LexicalIndex, VectorIndex


def _fts_query(query: str) -> str:
    tokens = [t for t in tokenize(query) if t]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"*' for t in tokens)


def _row_to_edge(r: sqlite3.Row) -> Dict:
    return {
        "src": r["src"], "rel": r["rel"], "dst": r["dst"],
        "src_type": r["src_type"], "dst_type": r["dst_type"], "video_id": r["video_id"],
        "t_start_s": r["t_start_s"], "t_end_s": r["t_end_s"], "modality": r["modality"],
        "confidence": r["confidence"], "props": json.loads(r["props"] or "{}"),
    }


class SqliteVectorIndex(VectorIndex):
    name = "sqlite"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, moment_id: str, embedding: Sequence[float]) -> None:
        # Store unit-normalized so retrieval can score by dot product (== cosine
        # for unit vectors) without recomputing norm() per row. Zero vectors are
        # stored unchanged (normalize is a no-op on them).
        vec = normalize(embedding)
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors (moment_id, dim, vec) VALUES (?, ?, ?)",
            (moment_id, len(vec), pack_floats(vec)),
        )

    def search(
        self,
        query_vec: Sequence[float],
        top_k: int = 20,
        *,
        video_id: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        if not query_vec or norm(query_vec) == 0.0:
            return []
        q = normalize(query_vec)  # cosine == dot once both sides are unit vectors
        # Opt-in FTS candidate prefilter: when query_text is provided, score only
        # the lexical-match candidate set (None == FTS unavailable -> score all).
        restrict = self._fts_candidate_ids(query_text) if query_text else None
        if video_id:
            rows = self.conn.execute(
                "SELECT v.moment_id AS moment_id, v.vec AS vec FROM vectors v "
                "JOIN moments m ON m.moment_id = v.moment_id WHERE m.video_id = ?",
                (video_id,),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT moment_id, vec FROM vectors").fetchall()
        qlen = len(q)
        scored: List[Tuple[str, float]] = []
        for r in rows:
            if restrict is not None and r["moment_id"] not in restrict:
                continue
            vec = unpack_floats(r["vec"])
            if len(vec) != qlen:
                continue
            scored.append((r["moment_id"], dot(q, vec)))  # no per-row norm() recompute
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _fts_candidate_ids(self, query_text: str) -> Optional[set]:
        """Moment ids matching ``query_text`` via FTS5, or None if FTS unavailable."""
        try:
            rows = self.conn.execute(
                "SELECT moment_id FROM moments_fts WHERE moments_fts MATCH ?",
                (_fts_query(query_text),),
            ).fetchall()
        except sqlite3.OperationalError:
            return None
        return {r["moment_id"] for r in rows}


class SqliteLexicalIndex(LexicalIndex):
    name = "sqlite"

    def __init__(self, conn: sqlite3.Connection, fts: bool) -> None:
        self.conn = conn
        self.fts = fts

    def add(self, moment_id: str, text: str) -> None:
        if self.fts:
            self.conn.execute("DELETE FROM moments_fts WHERE moment_id = ?", (moment_id,))
            self.conn.execute(
                "INSERT INTO moments_fts (moment_id, text) VALUES (?, ?)",
                (moment_id, text),
            )

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        query = (query or "").strip()
        if not query:
            return []
        if self.fts:
            try:
                rows = self.conn.execute(
                    "SELECT moment_id, rank FROM moments_fts WHERE moments_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (_fts_query(query), top_k),
                ).fetchall()
                # FTS5 rank: lower is better -> convert to descending score.
                return [(r["moment_id"], -float(r["rank"])) for r in rows]
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        rows = self.conn.execute(
            "SELECT moment_id FROM moments WHERE transcript LIKE ? OR ocr_text LIKE ? LIMIT ?",
            (like, like, top_k),
        ).fetchall()
        return [(r["moment_id"], 1.0) for r in rows]


class SqliteGraphStore(GraphStore):
    name = "sqlite"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add_edge(
        self, src: str, rel: str, dst: str, *,
        src_type: str = "", dst_type: str = "", video_id: Optional[str] = None,
        t_start_s: float = 0.0, t_end_s: float = 0.0, modality: str = "speech",
        confidence: float = 1.0, props: Optional[dict] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO edges
            (src, rel, dst, src_type, dst_type, video_id, t_start_s, t_end_s,
             modality, confidence, props)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (src, rel, dst, src_type, dst_type, video_id, t_start_s, t_end_s,
             modality, confidence, json.dumps(props or {})),
        )
        self.conn.commit()

    def neighbors(self, node: str, *, rel: Optional[str] = None,
                  direction: str = "out") -> List[dict]:
        col = "src" if direction == "out" else "dst"
        sql = f"SELECT * FROM edges WHERE {col} = ?"
        params: List[object] = [node]
        if rel:
            sql += " AND rel = ?"
            params.append(rel)
        return [_row_to_edge(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def edges(self, *, rel: Optional[str] = None) -> List[dict]:
        sql = "SELECT * FROM edges"
        params: List[object] = []
        if rel:
            sql += " WHERE rel = ?"
            params.append(rel)
        return [_row_to_edge(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]
