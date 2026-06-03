"""Loom — the knowledge store: one SQLite DB holding all four indices.

A single embedded database carries (spec §6):
  * relational metadata + provenance (videos, moments, claims, entities, ...)
  * the **lexical** index (SQLite FTS5, with a LIKE fallback)
  * the **vector** index (float32 BLOBs + brute-force cosine — no numpy needed)
  * the **temporal knowledge graph** (timestamped, provenanced edge table)

Keeping everything in one inspectable file honors "human-readable substrate":
you can ``sqlite3 ~/.memovox/memovox.db`` and read your knowledge directly.
Optional Qdrant/LanceDB/Kùzu backends can replace individual legs later.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import Config
from ..util import now_iso, tokenize
from ..vectormath import cosine, norm, pack_floats, unpack_floats
from .models import (
    STATUS_COMMITTED,
    STATUS_SUPERSEDED,
    Claim,
    Entity,
    Moment,
    Speaker,
    Topic,
    Video,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    source_url TEXT, title TEXT, channel TEXT, published_at TEXT,
    duration_s REAL, lang TEXT, content_hash TEXT,
    ingested_at TEXT, pipeline_version TEXT
);

CREATE TABLE IF NOT EXISTS speakers (
    speaker_id TEXT PRIMARY KEY, label TEXT, resolved_name TEXT, voiceprint_ref TEXT,
    canonical_id TEXT
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id TEXT PRIMARY KEY, label TEXT, moment_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY, canonical_name TEXT, type TEXT,
    wikidata_qid TEXT, aliases TEXT
);

CREATE TABLE IF NOT EXISTS moments (
    moment_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    idx INTEGER DEFAULT 0,
    t_start_s REAL, t_end_s REAL,
    transcript TEXT, speaker_id TEXT,
    visual_caption TEXT, ocr_text TEXT, topic_id TEXT
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    moment_id TEXT NOT NULL REFERENCES moments(moment_id) ON DELETE CASCADE,
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    text TEXT, subject TEXT, predicate TEXT, object TEXT,
    claim_type TEXT, salience REAL, entailment_score REAL,
    status TEXT, superseded_by TEXT,
    t_start_s REAL, t_end_s REAL, speaker_id TEXT, qualifiers TEXT
);

CREATE TABLE IF NOT EXISTS mentions (
    claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    PRIMARY KEY (claim_id, entity_id)
);

CREATE TABLE IF NOT EXISTS vectors (
    moment_id TEXT PRIMARY KEY REFERENCES moments(moment_id) ON DELETE CASCADE,
    dim INTEGER, vec BLOB
);

CREATE TABLE IF NOT EXISTS visual_vectors (
    moment_id TEXT PRIMARY KEY REFERENCES moments(moment_id) ON DELETE CASCADE,
    dim INTEGER, vec BLOB
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT, rel TEXT, dst TEXT, src_type TEXT, dst_type TEXT,
    video_id TEXT, t_start_s REAL, t_end_s REAL, modality TEXT,
    confidence REAL, props TEXT,
    UNIQUE (src, rel, dst, video_id)
);

CREATE INDEX IF NOT EXISTS idx_moments_video ON moments(video_id);
CREATE INDEX IF NOT EXISTS idx_claims_video ON claims(video_id);
CREATE INDEX IF NOT EXISTS idx_claims_moment ON claims(moment_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(rel);
"""


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp._mv_probe USING fts5(x)")
        conn.execute("DROP TABLE temp._mv_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _fts_query(query: str) -> str:
    tokens = [t for t in tokenize(query) if t]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"*' for t in tokens)


class LoomStore:
    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure()
        self.conn = sqlite3.connect(str(config.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.fts = _fts5_available(self.conn)
        self._migrate()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LoomStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _migrate(self) -> None:
        self.conn.executescript(_SCHEMA)
        # Idempotent column add for stores created before W4.1: executescript's
        # ``CREATE TABLE IF NOT EXISTS`` won't add a column to an existing table.
        try:
            self.conn.execute("ALTER TABLE speakers ADD COLUMN canonical_id TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists (fresh store / already migrated)
        if self.fts:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS moments_fts "
                "USING fts5(moment_id UNINDEXED, text)"
            )
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    # -- meta --------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    # -- videos ------------------------------------------------------------

    def get_video(self, video_id: str) -> Optional[Video]:
        row = self.conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return _row_to_video(row) if row else None

    def list_videos(self) -> List[Video]:
        rows = self.conn.execute("SELECT * FROM videos ORDER BY ingested_at DESC").fetchall()
        return [_row_to_video(r) for r in rows]

    def is_unchanged(self, video: Video) -> bool:
        existing = self.get_video(video.video_id)
        return bool(
            existing
            and existing.content_hash == video.content_hash
            and existing.pipeline_version == video.pipeline_version
        )

    def upsert_video(self, video: Video) -> None:
        if not video.ingested_at:
            video.ingested_at = now_iso()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO videos
            (video_id, source_url, title, channel, published_at, duration_s,
             lang, content_hash, ingested_at, pipeline_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video.video_id, video.source_url, video.title, video.channel,
                video.published_at, video.duration_s, video.lang, video.content_hash,
                video.ingested_at, video.pipeline_version,
            ),
        )
        self.conn.commit()

    def delete_video(self, video_id: str) -> bool:
        ids = [r["moment_id"] for r in self.conn.execute(
            "SELECT moment_id FROM moments WHERE video_id = ?", (video_id,)
        ).fetchall()]
        if self.fts and ids:
            self.conn.executemany("DELETE FROM moments_fts WHERE moment_id = ?", [(i,) for i in ids])
        self.conn.execute("DELETE FROM edges WHERE video_id = ?", (video_id,))
        cur = self.conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # -- moments -----------------------------------------------------------

    def add_moment(
        self,
        moment: Moment,
        embedding: Optional[Sequence[float]] = None,
        *,
        visual_embedding: Optional[Sequence[float]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO moments
            (moment_id, video_id, idx, t_start_s, t_end_s, transcript,
             speaker_id, visual_caption, ocr_text, topic_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                moment.moment_id, moment.video_id, moment.index, moment.t_start_s,
                moment.t_end_s, moment.transcript, moment.speaker_id,
                moment.visual_caption, moment.ocr_text, moment.topic_id,
            ),
        )
        if self.fts:
            self.conn.execute("DELETE FROM moments_fts WHERE moment_id = ?", (moment.moment_id,))
            self.conn.execute(
                "INSERT INTO moments_fts (moment_id, text) VALUES (?, ?)",
                (moment.moment_id, moment.text_for_embedding()),
            )
        if embedding is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO vectors (moment_id, dim, vec) VALUES (?, ?, ?)",
                (moment.moment_id, len(embedding), pack_floats(embedding)),
            )
        if visual_embedding is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO visual_vectors (moment_id, dim, vec) VALUES (?, ?, ?)",
                (moment.moment_id, len(visual_embedding), pack_floats(visual_embedding)),
            )
        self.conn.commit()

    def get_visual_vector(self, moment_id: str) -> Optional[List[float]]:
        row = self.conn.execute(
            "SELECT vec FROM visual_vectors WHERE moment_id = ?", (moment_id,)
        ).fetchone()
        return unpack_floats(row["vec"]) if row else None

    def get_moment(self, moment_id: str) -> Optional[Moment]:
        row = self.conn.execute("SELECT * FROM moments WHERE moment_id = ?", (moment_id,)).fetchone()
        return _row_to_moment(row) if row else None

    def get_moments(self, moment_ids: Sequence[str]) -> List[Moment]:
        if not moment_ids:
            return []
        placeholders = ",".join("?" for _ in moment_ids)
        rows = self.conn.execute(
            f"SELECT * FROM moments WHERE moment_id IN ({placeholders})", tuple(moment_ids)
        ).fetchall()
        by_id = {r["moment_id"]: _row_to_moment(r) for r in rows}
        return [by_id[m] for m in moment_ids if m in by_id]

    def moments_for_video(self, video_id: str) -> List[Moment]:
        rows = self.conn.execute(
            "SELECT * FROM moments WHERE video_id = ? ORDER BY idx", (video_id,)
        ).fetchall()
        return [_row_to_moment(r) for r in rows]

    def set_moment_topic(self, moment_id: str, topic_id: Optional[str]) -> None:
        """Assign (or clear) a Moment's topic (topic induction, spec §4.7)."""
        self.conn.execute(
            "UPDATE moments SET topic_id = ? WHERE moment_id = ?", (topic_id, moment_id)
        )
        self.conn.commit()

    def moments_for_topic(self, topic_id: str) -> List[Moment]:
        rows = self.conn.execute(
            "SELECT * FROM moments WHERE topic_id = ? ORDER BY video_id, idx", (topic_id,)
        ).fetchall()
        return [_row_to_moment(r) for r in rows]

    def moment_vectors(self) -> List[Tuple[str, List[float]]]:
        """All (moment_id, text vector) pairs, ordered by moment_id (deterministic).

        The read side of topic induction / clustering — reuses the persisted dense
        vectors so no re-embedding (or model) is needed on the free path.
        """
        rows = self.conn.execute(
            "SELECT moment_id, vec FROM vectors ORDER BY moment_id"
        ).fetchall()
        return [(r["moment_id"], unpack_floats(r["vec"])) for r in rows]

    # -- claims ------------------------------------------------------------

    def add_claim(self, claim: Claim) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO claims
            (claim_id, moment_id, video_id, text, subject, predicate, object,
             claim_type, salience, entailment_score, status, superseded_by,
             t_start_s, t_end_s, speaker_id, qualifiers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.claim_id, claim.moment_id, claim.video_id, claim.text,
                claim.subject, claim.predicate, claim.object, claim.claim_type,
                claim.salience, claim.entailment_score, claim.status, claim.superseded_by,
                claim.t_start_s, claim.t_end_s, claim.speaker_id,
                json.dumps(claim.qualifiers or {}),
            ),
        )
        self.conn.commit()

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        row = self.conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        return _row_to_claim(row) if row else None

    def get_claims(self, claim_ids: Sequence[str]) -> List[Claim]:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        rows = self.conn.execute(
            f"SELECT * FROM claims WHERE claim_id IN ({placeholders})", tuple(claim_ids)
        ).fetchall()
        by_id = {r["claim_id"]: _row_to_claim(r) for r in rows}
        return [by_id[c] for c in claim_ids if c in by_id]

    def claims_for_video(self, video_id: str, *, status: Optional[str] = STATUS_COMMITTED) -> List[Claim]:
        sql = "SELECT * FROM claims WHERE video_id = ?"
        params: List[object] = [video_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY t_start_s"
        return [_row_to_claim(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def claims_for_moment(self, moment_id: str, *, status: Optional[str] = STATUS_COMMITTED) -> List[Claim]:
        """Committed claims for a moment by default; pass ``status=None`` for all
        (including unsupported/superseded)."""
        sql = "SELECT * FROM claims WHERE moment_id = ?"
        params: List[object] = [moment_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY claim_id"
        return [_row_to_claim(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def supersede_claim(self, old_id: str, new_id: str) -> None:
        """Mark ``old_id`` as superseded by ``new_id`` (versioned, never deleted).

        The old claim is retained (still fetchable via get_claim) but its status
        becomes 'superseded', so it drops out of default committed queries.
        """
        self.conn.execute(
            "UPDATE claims SET status = ?, superseded_by = ? WHERE claim_id = ?",
            (STATUS_SUPERSEDED, new_id, old_id),
        )
        self.conn.commit()

    def list_claims(self, *, status: Optional[str] = STATUS_COMMITTED, claim_type: Optional[str] = None) -> List[Claim]:
        sql = "SELECT * FROM claims WHERE 1=1"
        params: List[object] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if claim_type:
            sql += " AND claim_type = ?"
            params.append(claim_type)
        return [_row_to_claim(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    # -- speakers / entities / topics -------------------------------------

    def upsert_speaker(self, speaker: Speaker) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO speakers "
            "(speaker_id, label, resolved_name, voiceprint_ref, canonical_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (speaker.speaker_id, speaker.label, speaker.resolved_name,
             speaker.voiceprint_ref, speaker.canonical_id),
        )
        self.conn.commit()

    def get_speaker(self, speaker_id: str) -> Optional[Speaker]:
        row = self.conn.execute("SELECT * FROM speakers WHERE speaker_id = ?", (speaker_id,)).fetchone()
        return _row_to_speaker(row) if row else None

    def list_speakers(self) -> List[Speaker]:
        rows = self.conn.execute("SELECT * FROM speakers ORDER BY speaker_id").fetchall()
        return [_row_to_speaker(r) for r in rows]

    def canonical_speaker(self, speaker_id: str) -> str:
        """The cross-video canonical id for a speaker.

        Returns the persisted ``canonical_id`` if cross-video speaker resolution
        (W4.1) has unified this speaker onto a ``spk:<slug>`` identity; otherwise
        the ``speaker_id`` itself (an unresolved / self-canonical speaker — e.g.
        an anonymous diarization label, which is never merged across videos).
        """
        row = self.conn.execute(
            "SELECT canonical_id FROM speakers WHERE speaker_id = ?", (speaker_id,)
        ).fetchone()
        if row and row["canonical_id"]:
            return row["canonical_id"]
        return speaker_id

    def upsert_entity(self, entity: Entity) -> None:
        # UPSERT (not INSERT OR REPLACE): re-inserting an entity that already
        # exists must UPDATE the row IN PLACE. INSERT OR REPLACE would DELETE the
        # old row first, and the mentions table's ``entity_id ... ON DELETE
        # CASCADE`` FK would then wipe every mention already linked to it — so the
        # second video to mention a shared entity would orphan the first video's
        # link. Merge aliases so the surface-form list accumulates across videos.
        existing = self.get_entity(entity.entity_id)
        if existing is not None:
            merged = list(existing.aliases)
            for alias in entity.aliases or []:
                if alias not in merged:
                    merged.append(alias)
            canonical = existing.canonical_name or entity.canonical_name
            etype = entity.type or existing.type
            qid = entity.wikidata_qid or existing.wikidata_qid
            self.conn.execute(
                "UPDATE entities SET canonical_name = ?, type = ?, wikidata_qid = ?, "
                "aliases = ? WHERE entity_id = ?",
                (canonical, etype, qid, json.dumps(merged), entity.entity_id),
            )
        else:
            self.conn.execute(
                "INSERT INTO entities (entity_id, canonical_name, type, wikidata_qid, aliases) "
                "VALUES (?, ?, ?, ?, ?)",
                (entity.entity_id, entity.canonical_name, entity.type, entity.wikidata_qid,
                 json.dumps(entity.aliases or [])),
            )
        self.conn.commit()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        row = self.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        return _row_to_entity(row) if row else None

    def list_entities(self) -> List[Entity]:
        rows = self.conn.execute("SELECT * FROM entities ORDER BY entity_id").fetchall()
        return [_row_to_entity(r) for r in rows]

    def link_mention(self, claim_id: str, entity_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO mentions (claim_id, entity_id) VALUES (?, ?)",
            (claim_id, entity_id),
        )
        self.conn.commit()

    def upsert_topic(self, topic: Topic) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO topics (topic_id, label, moment_count) VALUES (?, ?, ?)",
            (topic.topic_id, topic.label, topic.moment_count),
        )
        self.conn.commit()

    def get_topic(self, topic_id: str) -> Optional[Topic]:
        row = self.conn.execute(
            "SELECT * FROM topics WHERE topic_id = ?", (topic_id,)
        ).fetchone()
        return _row_to_topic(row) if row else None

    def list_topics(self) -> List[Topic]:
        rows = self.conn.execute("SELECT * FROM topics ORDER BY topic_id").fetchall()
        return [_row_to_topic(r) for r in rows]

    # -- graph -------------------------------------------------------------

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

    def neighbors(self, node: str, *, rel: Optional[str] = None, direction: str = "out") -> List[dict]:
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

    # -- search ------------------------------------------------------------

    def lexical_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
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

    def vector_search(
        self, query_vec: Sequence[float], top_k: int = 20, *, video_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        if not query_vec or norm(query_vec) == 0.0:
            return []
        if video_id:
            rows = self.conn.execute(
                "SELECT v.moment_id AS moment_id, v.vec AS vec FROM vectors v "
                "JOIN moments m ON m.moment_id = v.moment_id WHERE m.video_id = ?",
                (video_id,),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT moment_id, vec FROM vectors").fetchall()
        qlen = len(query_vec)
        scored: List[Tuple[str, float]] = []
        for r in rows:
            vec = unpack_floats(r["vec"])
            if len(vec) != qlen:
                continue
            scored.append((r["moment_id"], cosine(query_vec, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        c = self.conn

        def count(table: str) -> int:
            return int(c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        committed = int(c.execute("SELECT COUNT(*) FROM claims WHERE status='committed'").fetchone()[0])
        unsupported = int(c.execute("SELECT COUNT(*) FROM claims WHERE status='unsupported'").fetchone()[0])
        return {
            "videos": count("videos"),
            "moments": count("moments"),
            "claims": count("claims"),
            "claims_committed": committed,
            "claims_unsupported": unsupported,
            "entities": count("entities"),
            "speakers": count("speakers"),
            "edges": count("edges"),
            "vectors": count("vectors"),
            "visual_vectors": count("visual_vectors"),
            "fts5": self.fts,
            "embed_meta": self.get_meta("embed_backend"),
            "store": str(self.config.store),
        }

    def entity_mentions(self, entity_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT claim_id FROM mentions WHERE entity_id = ?", (entity_id,)
        ).fetchall()
        return [r["claim_id"] for r in rows]


# --------------------------------------------------------------------------- #
# row -> model converters
# --------------------------------------------------------------------------- #


def _row_to_video(r: sqlite3.Row) -> Video:
    return Video(
        video_id=r["video_id"], source_url=r["source_url"], title=r["title"],
        channel=r["channel"], published_at=r["published_at"], duration_s=r["duration_s"],
        lang=r["lang"], content_hash=r["content_hash"], ingested_at=r["ingested_at"],
        pipeline_version=r["pipeline_version"],
    )


def _row_to_moment(r: sqlite3.Row) -> Moment:
    return Moment(
        moment_id=r["moment_id"], video_id=r["video_id"], t_start_s=r["t_start_s"],
        t_end_s=r["t_end_s"], transcript=r["transcript"], speaker_id=r["speaker_id"],
        visual_caption=r["visual_caption"], ocr_text=r["ocr_text"], topic_id=r["topic_id"],
        index=r["idx"] or 0,
    )


def _row_to_claim(r: sqlite3.Row) -> Claim:
    return Claim(
        claim_id=r["claim_id"], moment_id=r["moment_id"], video_id=r["video_id"],
        text=r["text"], subject=r["subject"], predicate=r["predicate"], object=r["object"],
        claim_type=r["claim_type"], salience=r["salience"], entailment_score=r["entailment_score"],
        status=r["status"], superseded_by=r["superseded_by"], t_start_s=r["t_start_s"],
        t_end_s=r["t_end_s"], speaker_id=r["speaker_id"],
        qualifiers=json.loads(r["qualifiers"] or "{}"),
    )


def _row_to_speaker(r: sqlite3.Row) -> Speaker:
    return Speaker(
        speaker_id=r["speaker_id"],
        label=r["label"],
        voiceprint_ref=r["voiceprint_ref"],
        resolved_name=r["resolved_name"],
        canonical_id=r["canonical_id"],
    )


def _row_to_topic(r: sqlite3.Row) -> Topic:
    return Topic(
        topic_id=r["topic_id"],
        label=r["label"],
        moment_count=r["moment_count"] or 0,
    )


def _row_to_entity(r: sqlite3.Row) -> Entity:
    return Entity(
        entity_id=r["entity_id"],
        canonical_name=r["canonical_name"],
        type=r["type"] or "concept",
        wikidata_qid=r["wikidata_qid"],
        aliases=json.loads(r["aliases"] or "[]"),
    )


def _row_to_edge(r: sqlite3.Row) -> Dict:
    return {
        "src": r["src"], "rel": r["rel"], "dst": r["dst"],
        "src_type": r["src_type"], "dst_type": r["dst_type"], "video_id": r["video_id"],
        "t_start_s": r["t_start_s"], "t_end_s": r["t_end_s"], "modality": r["modality"],
        "confidence": r["confidence"], "props": json.loads(r["props"] or "{}"),
    }
