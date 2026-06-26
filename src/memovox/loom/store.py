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
from ..errors import SchemaVersionError
from ..observe import Tracer
from ..util import now_iso
from ..vectormath import normalize, pack_floats, unpack_floats
from .backends import get_graph_store, get_lexical_index, get_vector_index
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

SCHEMA_VERSION = 3

#: Per-video "the ingest pipeline ran to completion" marker (meta key prefix). Set as the
#: LAST step of a successful ingest; checked alongside is_unchanged so a video left
#: half-written by a crashed/killed ingest is NOT masked as "unchanged" forever. No schema
#: change — it lives in the existing meta table.
_INGEST_COMPLETE_PREFIX = "ingest_complete:"

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
    dim INTEGER, vec BLOB, space TEXT DEFAULT 'text'
);

CREATE TABLE IF NOT EXISTS visual_vectors (
    moment_id TEXT PRIMARY KEY REFERENCES moments(moment_id) ON DELETE CASCADE,
    dim INTEGER, vec BLOB, space TEXT DEFAULT 'visual_sig'
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT, rel TEXT, dst TEXT, src_type TEXT, dst_type TEXT,
    video_id TEXT, t_start_s REAL, t_end_s REAL, modality TEXT,
    confidence REAL, props TEXT,
    UNIQUE (src, rel, dst, video_id)
);

CREATE TABLE IF NOT EXISTS stage_metrics (
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    stage TEXT, wall_ms REAL, counters TEXT, caps TEXT, recorded_at TEXT
);

CREATE TABLE IF NOT EXISTS metrics_ledger (
    metric TEXT PRIMARY KEY, value REAL
);

CREATE INDEX IF NOT EXISTS idx_stage_metrics_video ON stage_metrics(video_id);
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
        # The four-index store composes pluggable backends (M0.2); the SQLite
        # defaults below share this connection and are the always-available free
        # path. Public LoomStore methods delegate to them.
        self.vector_index = get_vector_index("auto", conn=self.conn)
        self.lexical_index = get_lexical_index("auto", conn=self.conn, fts=self.fts)
        self.graph_store = get_graph_store("auto", conn=self.conn)
        # Visual signature index (M1.1): a SEPARATE space; RAW (un-normalized)
        # signatures scored by cosine. Reuses the SQLite VectorIndex on a different
        # table so there is one visual vector index, not a second ad-hoc cosine.
        self.visual_index = get_vector_index(
            "auto", conn=self.conn, table="visual_vectors", space="visual_sig",
            normalize_vectors=False,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LoomStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _migrate(self) -> None:
        # Refuse to open a store written by a NEWER memovox: silently down-stamping
        # user_version + running older code against a future schema risks corruption.
        current = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if current > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"store schema version {current} is newer than this memovox "
                f"(supports {SCHEMA_VERSION}); upgrade memovox or use a separate store."
            )
        self.conn.executescript(_SCHEMA)
        # Idempotent column add for stores created before W4.1: executescript's
        # ``CREATE TABLE IF NOT EXISTS`` won't add a column to an existing table.
        try:
            self.conn.execute("ALTER TABLE speakers ADD COLUMN canonical_id TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists (fresh store / already migrated)
        # M1.1: space-tag vector tables (idempotent; backfill old rows).
        for tbl, default_space in (("vectors", "text"), ("visual_vectors", "visual_sig")):
            try:
                self.conn.execute(f"ALTER TABLE {tbl} ADD COLUMN space TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            self.conn.execute(
                f"UPDATE {tbl} SET space = ? WHERE space IS NULL", (default_space,)
            )
        if self.fts:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS moments_fts "
                "USING fts5(moment_id UNINDEXED, text)"
            )
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()
        self._backfill_normalized_vectors()
        self._backfill_fts()
        self._backfill_ingest_complete()

    def _backfill_normalized_vectors(self) -> None:
        """One-time (M0.2 W3): unit-normalize any pre-existing main vectors so the
        dot-product retrieval path is exact. Guarded by a meta flag (idempotent);
        visual_vectors live in a different space and are deliberately untouched.
        """
        if self.get_meta("vectors_normalized") == "1":
            return
        rows = self.conn.execute("SELECT moment_id, vec FROM vectors").fetchall()
        for r in rows:
            packed = pack_floats(normalize(unpack_floats(r["vec"])))
            if packed != r["vec"]:  # compare packed bytes, not float lists (float32 round-trip)
                self.conn.execute("UPDATE vectors SET vec = ? WHERE moment_id = ?",
                                  (packed, r["moment_id"]))
        self.set_meta("vectors_normalized", "1")  # commits

    def _backfill_fts(self) -> None:
        """One-time: populate ``moments_fts`` from the existing ``moments`` rows.

        ``_migrate`` creates ``moments_fts`` ``IF NOT EXISTS`` but only ``add_moment`` ever
        writes it. memovox deliberately supports a SQLite build WITHOUT fts5 (moments are
        then indexed only via the LIKE fallback, ``moments_fts`` absent). If such a store is
        later opened on an fts5-CAPABLE build (system python vs uv/conda, or a wheel
        upgrade), migration creates an EMPTY ``moments_fts`` and flips ``self.fts`` True —
        so ``lexical_search`` / ``doc_freq`` would run MATCH against an empty index and
        silently miss every pre-existing moment. Backfill closes that index-vs-truth gap.
        Self-healing via a count invariant (NOT a one-shot flag): a sticky flag left a store
        written on a no-fts5 build AFTER an earlier fts5 open permanently unindexed (the flag
        was already 1, so the reopen skipped the backfill). A COUNT(moments) vs
        COUNT(moments_fts) comparison short-circuits the common in-sync open cheaply and
        reconciles whenever they diverge — in either direction of the supported transition."""
        if not self.fts:
            return
        n_moments = self.conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0]
        n_fts = self.conn.execute("SELECT COUNT(*) FROM moments_fts").fetchone()[0]
        if n_moments == n_fts:
            return  # index already in parity — the fast path on every open
        rows = self.conn.execute(
            "SELECT * FROM moments WHERE moment_id NOT IN (SELECT moment_id FROM moments_fts)"
        ).fetchall()
        for r in rows:
            m = _row_to_moment(r)
            self.conn.execute("INSERT INTO moments_fts (moment_id, text) VALUES (?, ?)",
                              (m.moment_id, m.text_for_embedding()))
        self.conn.commit()

    def _backfill_ingest_complete(self) -> None:
        """Pre-marker videos are assumed COMPLETE — they were committed by older code that
        had no partial-ingest marker, so without this the first re-ingest of every legacy
        video would needlessly rebuild it (is_ingest_complete would read False). One-time,
        flag-guarded; new videos get the marker the normal way (set at end of ingest)."""
        if self.get_meta("ingest_complete_backfilled") == "1":
            return
        for r in self.conn.execute("SELECT video_id FROM videos").fetchall():
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, '1')",
                (_INGEST_COMPLETE_PREFIX + r["video_id"],))
        self.set_meta("ingest_complete_backfilled", "1")  # commits

    # -- meta --------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def mark_ingest_complete(self, video_id: str) -> None:
        """Record that ``video_id``'s ingest ran to completion (the LAST ingest step)."""
        self.set_meta(_INGEST_COMPLETE_PREFIX + video_id, "1")

    def is_ingest_complete(self, video_id: str) -> bool:
        """True iff ``video_id`` was fully ingested (vs left partial by a crashed ingest)."""
        return self.get_meta(_INGEST_COMPLETE_PREFIX + video_id) == "1"

    def append_meta_json_id(self, key: str, value_id: str) -> None:
        """Atomically add ``value_id`` to a JSON sorted-id-list meta value.

        The read-modify-write runs under a single write-locked (``BEGIN IMMEDIATE``)
        transaction so two concurrent appenders cannot clobber each other — the lost-update
        failure mode of the old get_meta()+set_meta() whole-value overwrite (a dropped
        seen-id silently triggers a needless, expensive re-ingest on the next sync).
        Idempotent: an id already present is a no-op. Format unchanged (sorted JSON list)."""
        # The connection uses sqlite3's legacy isolation mode, so an uncommitted write would
        # leave a transaction open and make the explicit BEGIN IMMEDIATE raise "cannot start a
        # transaction within a transaction". Flush any pending implicit txn first so this
        # helper is robust regardless of the caller's connection state.
        if self.conn.in_transaction:
            self.conn.commit()
        self.conn.execute("BEGIN IMMEDIATE")  # acquire the write lock BEFORE the read
        try:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            ids: set = set()
            if row and row["value"]:
                try:
                    ids = set(json.loads(row["value"]))
                except (ValueError, TypeError):
                    ids = set()  # corrupt cursor -> rebuild from this id
            if value_id not in ids:
                ids.add(value_id)
                self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                                  (key, json.dumps(sorted(ids))))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- observability metrics (M0.1) --------------------------------------

    def record_stage_metrics(self, video_id: str, tracer: Tracer) -> None:
        """Persist one row per traced stage for ``video_id`` (idempotent: replaces).

        ``wall_ms`` is volatile (machine-dependent) and never gated; counters/caps
        are deterministic. Old rows for the video are cleared first so re-ingest
        leaves exactly one current trace.
        """
        self.conn.execute("DELETE FROM stage_metrics WHERE video_id = ?", (video_id,))
        recorded = now_iso()
        for sp in tracer.spans:
            self.conn.execute(
                "INSERT INTO stage_metrics (video_id, stage, wall_ms, counters, caps, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (video_id, sp.stage, sp.wall_ms,
                 json.dumps(sp.counters, sort_keys=True),
                 json.dumps(sp.caps, sort_keys=True), recorded),
            )
        self.conn.commit()

    def stage_metrics(self, video_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT stage, wall_ms, counters, caps, recorded_at FROM stage_metrics "
            "WHERE video_id = ? ORDER BY rowid", (video_id,)
        ).fetchall()
        return [
            {"stage": r["stage"], "wall_ms": r["wall_ms"],
             "counters": json.loads(r["counters"]), "caps": json.loads(r["caps"]),
             "recorded_at": r["recorded_at"]}
            for r in rows
        ]

    def bump_ledger(self, updates: Dict[str, float]) -> None:
        """Accumulate cumulative, monotonic lifetime counters (conflict-safe upsert)."""
        for metric, delta in updates.items():
            self.conn.execute(
                "INSERT INTO metrics_ledger (metric, value) VALUES (?, ?) "
                "ON CONFLICT(metric) DO UPDATE SET value = value + excluded.value",
                (metric, float(delta)),
            )
        self.conn.commit()

    def metrics_ledger(self) -> Dict[str, float]:
        rows = self.conn.execute(
            "SELECT metric, value FROM metrics_ledger ORDER BY metric"
        ).fetchall()
        return {r["metric"]: r["value"] for r in rows}

    def _ledger_contribution(self, video_id: str) -> Dict[str, float]:
        """The video's own contribution to the cumulative ledger, reconstructed
        from its persisted stage metrics — the exact inverse of the bump in
        ``pipeline.ingest``. Returns ``{}`` when the video has no stage metrics
        (it was never recorded, so it never bumped the ledger): nothing to undo."""
        metrics = self.stage_metrics(video_id)
        if not metrics:
            return {}
        by_stage = {m["stage"]: m["counters"] for m in metrics}
        moments = by_stage.get("moments", {})
        claims = by_stage.get("claims", {})
        visual = by_stage.get("visual", {})
        return {
            "videos": 1.0,
            "moments": float(moments.get("moments", 0)),
            "claims_committed": float(claims.get("committed", 0)),
            "claims_unsupported": float(claims.get("unsupported", 0)),
            "visual_events": float(visual.get("events", 0)),
            "frames": float(visual.get("frames", 0)),
        }

    def _decrement_ledger(self, updates: Dict[str, float]) -> None:
        """Subtract a (forgotten) video's contribution from the cumulative ledger,
        clamped at 0 so redaction can never drive a lifetime counter negative."""
        for metric, delta in updates.items():
            self.conn.execute(
                "UPDATE metrics_ledger SET value = MAX(0, value - ?) WHERE metric = ?",
                (float(delta), metric),
            )

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
        """Redaction primitive (§2/§12): atomically delete a video and ALL its derived
        data. The FK cascade (moments→claims→mentions/vectors) handles the per-video
        rows; the ``edges`` table has NO foreign key (plain ``video_id`` column), so
        edges are removed explicitly: (a) ALL edges stamped with this video_id —
        including SAME_AS speaker edges whose endpoints are neither a moment nor a
        claim; (b) cross-video edges that POINT AT the deleted moments/claims but are
        stamped with ANOTHER video's id. Then GC entity/topic nodes left member-less
        AND their now-dangling inbound MENTIONS/ABOUT edges, drop the FTS rows, and
        reset the consolidation watermark (deleted rowids can be reused, which would
        otherwise make an incremental re-scan skip genuinely new claims). One
        transaction; nothing dangles."""
        moment_ids = [r["moment_id"] for r in self.conn.execute(
            "SELECT moment_id FROM moments WHERE video_id = ?", (video_id,)).fetchall()]
        claim_ids = [r["claim_id"] for r in self.conn.execute(
            "SELECT claim_id FROM claims WHERE video_id = ?", (video_id,)).fetchall()]
        # Capture the ledger contribution BEFORE the cascade drops stage_metrics, so
        # forget (redaction) leaves the cumulative ledger consistent with reality
        # rather than still counting the deleted video's moments/claims.
        ledger_delta = self._ledger_contribution(video_id)
        # ONE atomic transaction (``with self.conn``): if any statement fails mid-way —
        # e.g. the edge IN-list exceeding the legacy 999 bound-variable limit, now also
        # chunked below — the whole delete ROLLS BACK rather than leaving the video gone
        # but its cross-video edges dangling (a non-atomic half-redaction).
        with self.conn:
            if self.fts and moment_ids:
                self.conn.executemany("DELETE FROM moments_fts WHERE moment_id = ?",
                                      [(i,) for i in moment_ids])
            cur = self.conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
            if cur.rowcount > 0:
                # (a) every edge stamped with this video (edges have no FK cascade) —
                # catches SAME_AS speaker edges (both endpoints are Speaker nodes).
                self.conn.execute("DELETE FROM edges WHERE video_id = ?", (video_id,))
                # (b) cross-video edges (stamped with another video's id) pointing AT the
                # now-deleted nodes — chunked, two single-column passes (see helper).
                self._delete_edges_referencing(moment_ids + claim_ids)
                # (c) per-video speaker rows ("<video_id>:<label>") have NO FK to videos,
                # so the moment/claim cascade never reaches them. Delete them explicitly
                # (substr-prefix match, not LIKE — a YouTube id can contain '_', a LIKE
                # wildcard) so a redaction leaves behind no orphaned — possibly NAMED —
                # speaker row (PII surviving the delete).
                prefix = f"{video_id}:"
                self.conn.execute("DELETE FROM speakers WHERE substr(speaker_id, 1, ?) = ?",
                                  (len(prefix), prefix))
                # ...then GC any canonical "spk:*" identity no surviving per-video speaker
                # still points at (a cross-video name fed only by the deleted video).
                self.conn.execute(
                    "DELETE FROM speakers WHERE speaker_id LIKE 'spk:%' "
                    "AND speaker_id NOT IN (SELECT canonical_id FROM speakers "
                    "WHERE canonical_id IS NOT NULL AND speaker_id NOT LIKE 'spk:%')")
                # GC corpus nodes this redaction left member-less.
                self.conn.execute(
                    "DELETE FROM entities WHERE entity_id NOT IN (SELECT entity_id FROM mentions)")
                self.conn.execute(
                    "DELETE FROM topics WHERE topic_id NOT IN "
                    "(SELECT topic_id FROM moments WHERE topic_id IS NOT NULL)")
                # ...and inbound edges from SURVIVING videos that now point at a GC'd node.
                self.conn.execute(
                    "DELETE FROM edges WHERE rel = 'MENTIONS' AND dst NOT IN "
                    "(SELECT entity_id FROM entities)")
                self.conn.execute(
                    "DELETE FROM edges WHERE rel = 'ABOUT' AND dst NOT IN "
                    "(SELECT topic_id FROM topics)")
                # rowid reuse safety: force a full re-scan on the next consolidation.
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('consolidation_watermark', '0')")
                # drop the per-video completion marker (redaction completeness + so a
                # re-ingest of this id is never masked as "complete" by a stale marker).
                self.conn.execute("DELETE FROM meta WHERE key = ?",
                                  (_INGEST_COMPLETE_PREFIX + video_id,))
                # undo the video's contribution to the cumulative metrics ledger.
                self._decrement_ledger(ledger_delta)
        return cur.rowcount > 0

    def _delete_edges_referencing(self, node_ids: Sequence[str], *, batch: int = 400) -> None:
        """Delete every edge whose ``src`` OR ``dst`` is in ``node_ids``, in chunks small
        enough to stay under the SQLite bound-variable limit (legacy default 999). Two
        single-column passes keep the per-statement placeholder count at len(chunk), so a
        video with thousands of moments+claims is redacted without "too many SQL variables".
        Caller runs this inside the ``delete_video`` transaction."""
        for i in range(0, len(node_ids), batch):
            chunk = node_ids[i:i + batch]
            if not chunk:
                continue
            ph = ",".join("?" * len(chunk))
            self.conn.execute(f"DELETE FROM edges WHERE src IN ({ph})", chunk)
            self.conn.execute(f"DELETE FROM edges WHERE dst IN ({ph})", chunk)

    # -- moments -----------------------------------------------------------

    def add_moment(
        self,
        moment: Moment,
        embedding: Optional[Sequence[float]] = None,
        *,
        visual_embedding: Optional[Sequence[float]] = None,
    ) -> None:
        # UPSERT (update-in-place on conflict) rather than INSERT OR REPLACE: with
        # foreign_keys=ON, REPLACE first DELETEs the conflicting row, which CASCADE-wipes
        # this moment's vectors/visual_vectors AND every claim attached to it (and their
        # mentions). Re-adding a moment must update the parent row, not silently destroy
        # its derived rows — the same reason upsert_entity uses read-then-UPDATE.
        self.conn.execute(
            """
            INSERT INTO moments
            (moment_id, video_id, idx, t_start_s, t_end_s, transcript,
             speaker_id, visual_caption, ocr_text, topic_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(moment_id) DO UPDATE SET
              video_id=excluded.video_id, idx=excluded.idx, t_start_s=excluded.t_start_s,
              t_end_s=excluded.t_end_s, transcript=excluded.transcript,
              speaker_id=excluded.speaker_id, visual_caption=excluded.visual_caption,
              ocr_text=excluded.ocr_text, topic_id=excluded.topic_id
            """,
            (
                moment.moment_id, moment.video_id, moment.index, moment.t_start_s,
                moment.t_end_s, moment.transcript, moment.speaker_id,
                moment.visual_caption, moment.ocr_text, moment.topic_id,
            ),
        )
        self.lexical_index.add(moment.moment_id, moment.text_for_embedding())
        if embedding is not None:
            self.vector_index.add(moment.moment_id, embedding)
        if visual_embedding is not None:
            self.visual_index.add(moment.moment_id, visual_embedding)  # raw, space='visual_sig'
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

    def clear_about_edges(self, moment_id: str) -> None:
        """Drop a moment's ABOUT->Topic edges so re-induction (which may re-cluster a
        moment onto a different topic) leaves at most ONE current ABOUT edge instead
        of accumulating stale edges to superseded topics. Idempotent."""
        self.conn.execute("DELETE FROM edges WHERE src = ? AND rel = 'ABOUT'", (moment_id,))
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
        # UPSERT, not INSERT OR REPLACE: REPLACE would CASCADE-delete this claim's mentions
        # rows (FK ON DELETE CASCADE) on every re-add. Update-in-place preserves them.
        self.conn.execute(
            """
            INSERT INTO claims
            (claim_id, moment_id, video_id, text, subject, predicate, object,
             claim_type, salience, entailment_score, status, superseded_by,
             t_start_s, t_end_s, speaker_id, qualifiers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
              moment_id=excluded.moment_id, video_id=excluded.video_id, text=excluded.text,
              subject=excluded.subject, predicate=excluded.predicate, object=excluded.object,
              claim_type=excluded.claim_type, salience=excluded.salience,
              entailment_score=excluded.entailment_score, status=excluded.status,
              superseded_by=excluded.superseded_by, t_start_s=excluded.t_start_s,
              t_end_s=excluded.t_end_s, speaker_id=excluded.speaker_id,
              qualifiers=excluded.qualifiers
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

    def claim_history(self, claim_id: str) -> List[Claim]:
        """All versions in a claim's supersede lineage, oldest→newest (M3.1, §2).

        Given ANY id in the chain, walk predecessors (claims whose ``superseded_by``
        points here) back to the head, then follow ``superseded_by`` forward to the
        end. Nothing is deleted — superseded versions are returned alongside the
        live one. Empty if the claim doesn't exist.

        Assumes a LINEAR lineage (the ``supersede_claim`` 1:1 contract): the
        backward walk takes one predecessor per step, so a branching
        ``superseded_by`` (only reachable via corrupt/out-of-band writes) yields
        one branch, not the union. The cycle guards keep it terminating regardless."""
        if not self.get_claim(claim_id):
            return []
        head, seen = claim_id, set()
        while head not in seen:
            seen.add(head)
            row = self.conn.execute(
                "SELECT claim_id FROM claims WHERE superseded_by = ?", (head,)
            ).fetchone()
            if not row:
                break
            head = row["claim_id"]
        chain: List[Claim] = []
        cur, walked = self.get_claim(head), set()
        while cur and cur.claim_id not in walked:
            walked.add(cur.claim_id)
            chain.append(cur)
            cur = self.get_claim(cur.superseded_by) if cur.superseded_by else None
        return chain

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
        # Explicit, stable insertion order (rowid) — the deterministic ordering the
        # consolidation cap + watermark (M0.2) rely on; identical to the prior
        # implicit table-scan order, so no result moves.
        sql += " ORDER BY rowid"
        return [_row_to_claim(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def max_claim_rowid(self) -> int:
        """High-water cursor for incremental consolidation (M0.2): the max claims
        rowid (monotonic with insertion; replace re-inserts at higher rowids)."""
        row = self.conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM claims").fetchone()
        return int(row[0])

    def committed_claim_ids_since(self, rowid: int) -> set:
        """Committed claim ids inserted after ``rowid`` (the NEW claims to scan)."""
        rows = self.conn.execute(
            "SELECT claim_id FROM claims WHERE rowid > ? AND status = ?",
            (rowid, STATUS_COMMITTED),
        ).fetchall()
        return {r["claim_id"] for r in rows}

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
        self.graph_store.add_edge(
            src, rel, dst, src_type=src_type, dst_type=dst_type, video_id=video_id,
            t_start_s=t_start_s, t_end_s=t_end_s, modality=modality,
            confidence=confidence, props=props,
        )

    def neighbors(self, node: str, *, rel: Optional[str] = None, direction: str = "out") -> List[dict]:
        return self.graph_store.neighbors(node, rel=rel, direction=direction)

    def edges(self, *, rel: Optional[str] = None) -> List[dict]:
        return self.graph_store.edges(rel=rel)

    def count_edges(self, *, rel: Optional[str] = None) -> int:
        return self.graph_store.count_edges(rel=rel)

    # -- search ------------------------------------------------------------

    def lexical_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        return self.lexical_index.search(query, top_k)

    def doc_freq(self, term: str) -> int:
        """Corpus document frequency: how many moments contain ``term`` (for IDF
        weighting of answer relevance, W5.1). FTS5 MATCH when available, else a
        LIKE scan over transcripts. The term is treated as a literal phrase."""
        term = (term or "").strip()
        if not term:
            return 0
        if self.fts:
            try:
                row = self.conn.execute(
                    "SELECT COUNT(*) FROM moments_fts WHERE moments_fts MATCH ?",
                    (f'"{term}"',),
                ).fetchone()
                return int(row[0]) if row else 0
            except sqlite3.OperationalError:
                pass  # fall through to the LIKE scan
        # Escape LIKE wildcards so a term containing % or _ is matched literally
        # (else doc_freq('%') would match every row).
        safe = term.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        row = self.conn.execute(
            "SELECT COUNT(*) FROM moments WHERE LOWER(transcript) LIKE ? ESCAPE '\\'",
            (f"%{safe}%",),
        ).fetchone()
        return int(row[0]) if row else 0

    def vector_search(
        self, query_vec: Sequence[float], top_k: int = 20, *,
        video_id: Optional[str] = None, query_text: Optional[str] = None,
        space: str = "text",
    ) -> List[Tuple[str, float]]:
        # The FTS candidate prefilter (M0.2 W4) is opt-in: pass query_text to the
        # index ONLY when the flag is on, so the default free path scores all
        # vectors (byte-identical to today).
        if not self.config.settings.vector_prefilter_fts:
            query_text = None
        return self.vector_index.search(query_vec, top_k, video_id=video_id,
                                        query_text=query_text, space=space)

    def visual_search(
        self, query_vec: Sequence[float], top_k: int = 20, *,
        video_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Rank moments by their RAW visual signature (M1.1, space='visual_sig')."""
        return self.visual_index.search(query_vec, top_k, video_id=video_id,
                                        space="visual_sig")

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


