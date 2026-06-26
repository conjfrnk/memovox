"""SQLite-backed storage indices — the always-available free defaults.

The SQL bodies here are lifted **verbatim** from ``LoomStore`` (the no-op refactor
in M0.2 W1) so the four-index store keeps a single embedded, inspectable DB. Each
index holds a reference to the shared ``sqlite3.Connection`` owned by ``LoomStore``;
writes that are part of a larger transaction (``add``) do NOT commit, while the
graph ``add_edge`` commits exactly as the legacy method did.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from ...errors import VectorSpaceError
from ...util import tokenize
from ...vectormath import cosine, dot, norm, normalize, pack_floats, unpack_floats
from .base import GraphStore, LexicalIndex, VectorIndex


#: Cap the distinct prefix-OR terms in the FTS5 MATCH expression. A real question is tens of
#: tokens; a 300k-token paste builds a MATCH whose evaluation is ~O(n^2) in term count — a
#: single-request CPU DoS NOT bounded by the request body-size cap (cost scales with token
#: COUNT, not bytes). Dedupe + slice keeps the expression small with negligible recall impact.
_FTS_MAX_TERMS = 200


def _fts_query(query: str) -> str:
    tokens = list(dict.fromkeys(t for t in tokenize(query) if t))[:_FTS_MAX_TERMS]
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
    """A single embedding space backed by one BLOB table.

    The text index (``vectors``, space ``text``) stores unit-normalized vectors and
    scores by dot product. The visual index (``visual_vectors``, space
    ``visual_sig``) stores RAW signatures and scores by cosine (scale-invariant) —
    visual signatures are a different space and must NOT be normalized. A search
    requesting a different ``space`` than the index serves raises VectorSpaceError.
    """

    name = "sqlite"

    def __init__(self, conn: sqlite3.Connection, *, table: str = "vectors",
                 space: str = "text", normalize_vectors: bool = True) -> None:
        self.conn = conn
        self.table = table
        self.space = space
        self.normalize_vectors = normalize_vectors

    def add(self, moment_id: str, embedding: Sequence[float]) -> None:
        vec = normalize(embedding) if self.normalize_vectors else list(embedding)
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self.table} (moment_id, dim, vec, space) "
            "VALUES (?, ?, ?, ?)",
            (moment_id, len(vec), pack_floats(vec), self.space),
        )

    def search(
        self,
        query_vec: Sequence[float],
        top_k: int = 20,
        *,
        video_id: Optional[str] = None,
        query_text: Optional[str] = None,
        space: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        if space is not None and space != self.space:
            raise VectorSpaceError(
                f"index serves space {self.space!r} but a {space!r}-space query was made"
            )
        # Reject an empty, zero, OR non-finite query vector. A NaN/Inf query bypasses a bare
        # ``norm(q) == 0.0`` check (NaN==0.0 / Inf==0.0 are both False), then normalize()
        # yields all-NaN and every score is NaN — which makes the (-score, id) sort below
        # non-transitive and the top-k non-deterministic. An optional embed backend can emit
        # NaN on a degenerate (near-zero) row, and visual_query_vec is external input.
        qn = norm(query_vec)
        if not query_vec or not math.isfinite(qn) or qn == 0.0:
            return []
        restrict = self._fts_candidate_ids(query_text) if query_text else None
        if video_id:
            rows = self.conn.execute(
                f"SELECT v.moment_id AS moment_id, v.vec AS vec FROM {self.table} v "
                "JOIN moments m ON m.moment_id = v.moment_id "
                "WHERE m.video_id = ? AND v.space = ?",
                (video_id, self.space),
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"SELECT moment_id, vec FROM {self.table} WHERE space = ?", (self.space,)
            ).fetchall()
        if self.normalize_vectors:
            q = normalize(query_vec)  # cosine == dot once both sides are unit vectors
            score = lambda vec: dot(q, vec)  # noqa: E731
        else:
            score = lambda vec: cosine(query_vec, vec)  # raw visual space  # noqa: E731
        qlen = len(query_vec)
        scored: List[Tuple[str, float]] = []
        for r in rows:
            if restrict is not None and r["moment_id"] not in restrict:
                continue
            vec = unpack_floats(r["vec"])
            if len(vec) != qlen:
                continue
            s = score(vec)
            if not math.isfinite(s):
                continue  # a single NaN/Inf stored vector must not poison the per-leg sort
            scored.append((r["moment_id"], s))
        # Stable tiebreak by moment_id: SQLite returns rows in no guaranteed order
        # (no ORDER BY), so equal scores would otherwise rank non-deterministically.
        scored.sort(key=lambda x: (-x[1], x[0]))
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

    def count_edges(self, *, rel: Optional[str] = None) -> int:
        if rel:
            row = self.conn.execute("SELECT COUNT(*) FROM edges WHERE rel = ?", (rel,)).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        return int(row[0])
