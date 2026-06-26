"""Stdlib SQLite-backed async job runner (M3.3).

A ``jobs`` table in the same ``memovox.db`` drives ``queued → running →
succeeded|failed`` with attempt counting, exponential-backoff retry, and idempotent
``(kind, args_hash)`` de-dup (re-enqueueing a non-terminal job returns the existing
id). The :class:`JobWorker` is a ``threading.Thread`` loop: claim the next due
``queued`` job, run it on its OWN ``Memovox``/store connection, persist every
transition (crash-resumable). Default concurrency is 1 (deterministic, serial). No
broker, no framework — ``threading`` + ``sqlite3`` + ``hashlib`` only. All logging
goes to STDERR (MCP owns stdout).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import threading
import time
import uuid
from typing import Optional

from ..config import Config

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"
_TERMINAL = (SUCCEEDED, FAILED)
_DEFAULT_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 2.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    args_json TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    next_run_at REAL NOT NULL DEFAULT 0,
    result_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_due ON jobs (state, next_run_at);
-- Partial UNIQUE index = the idempotent (kind, args_hash) de-dup enforced ATOMICALLY at
-- the DB layer, so two concurrent enqueue() callers (separate connections, each its own
-- JobStore via the ThreadingHTTPServer) cannot both pass a SELECT-then-INSERT window and
-- create duplicate non-terminal jobs (which would run the same multi-minute ingest twice).
-- Only ACTIVE (queued|running) rows participate, so a fresh job is still enqueueable once
-- the prior one reaches a terminal state.
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_active_dedup
    ON jobs (kind, args_hash) WHERE state IN ('queued', 'running');
"""


def _args_hash(args: dict) -> str:
    return hashlib.sha256(json.dumps(args or {}, sort_keys=True).encode("utf-8")).hexdigest()


class JobStore:
    """Persistence + idempotent enqueue for jobs (own connection, WAL + busy_timeout)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure()
        # check_same_thread=False: a JobWorker creates its JobStore on the spawning
        # thread but uses it on the worker thread (never concurrently — requeue
        # completes before start()). WAL + busy_timeout serialize cross-connection writes.
        self.conn = sqlite3.connect(str(config.db_path), timeout=5.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def enqueue(self, kind: str, args: Optional[dict] = None, *,
                max_attempts: int = _DEFAULT_MAX_ATTEMPTS) -> str:
        """Enqueue a job; returns the existing id if an identical (kind, args_hash)
        job is still non-terminal (idempotent de-dup), else a fresh queued job.

        The de-dup is enforced by the ``idx_jobs_active_dedup`` partial UNIQUE index, so it
        is ATOMIC under concurrent enqueue() from separate connections: ``INSERT OR IGNORE``
        either wins (rowcount 1 -> our id) or is ignored because an active duplicate exists
        (rowcount 0 -> we return that existing job's id). No SELECT-then-INSERT race window."""
        args = args or {}
        h = _args_hash(args)
        job_id = uuid.uuid4().hex
        now = time.time()
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO jobs (job_id, kind, args_hash, args_json, state, attempts, "
            "max_attempts, next_run_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
            (job_id, kind, h, json.dumps(args, sort_keys=True), QUEUED, max_attempts, now, now, now),
        )
        self.conn.commit()
        if cur.rowcount == 1:
            return job_id
        # Ignored: an active (queued|running) job with this (kind, args_hash) already
        # exists -> return its id (idempotent). The fallback to our own id covers only the
        # vanishing race where that active job reached a terminal state in between.
        existing = self.conn.execute(
            "SELECT job_id FROM jobs WHERE kind=? AND args_hash=? AND state IN (?, ?) "
            "ORDER BY created_at, job_id LIMIT 1",
            (kind, h, QUEUED, RUNNING),
        ).fetchone()
        return existing["job_id"] if existing else job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def claim_next(self) -> Optional[dict]:
        """Atomically flip the oldest due queued job to running and return it. The
        UPDATE is guarded by ``state='queued'`` + a rowcount check so two workers
        (concurrency>1) can never both claim the same job — only the winner's
        conditional UPDATE matches a row."""
        now = time.time()
        with self.conn:  # busy_timeout serializes the conditional UPDATE across workers
            row = self.conn.execute(
                "SELECT * FROM jobs WHERE state=? AND next_run_at<=? "
                "ORDER BY created_at, job_id LIMIT 1",
                (QUEUED, now),
            ).fetchone()
            if not row:
                return None
            cur = self.conn.execute(
                "UPDATE jobs SET state=?, attempts=attempts+1, updated_at=? "
                "WHERE job_id=? AND state=?",
                (RUNNING, now, row["job_id"], QUEUED),
            )
            if cur.rowcount != 1:
                return None  # another worker won the race
        return self.get_job(row["job_id"])

    def mark_succeeded(self, job_id: str, result: dict) -> None:
        # ``AND state='running'`` so a zombie/duplicate runner (e.g. a job re-claimed by a
        # second worker after requeue_stuck_running) cannot resurrect a job that has already
        # reached a terminal state — the final record stays consistent.
        self.conn.execute(
            "UPDATE jobs SET state=?, result_json=?, error=NULL, updated_at=? "
            "WHERE job_id=? AND state=?",
            (SUCCEEDED, json.dumps(result), time.time(), job_id, RUNNING),
        )
        self.conn.commit()

    def mark_failed_or_retry(self, job_id: str, error: str) -> str:
        """Record a failed attempt: retry with exponential backoff while attempts
        remain, else mark failed. Returns the resulting state."""
        job = self.get_job(job_id)
        now = time.time()
        if job and job["attempts"] < job["max_attempts"]:
            delay = _BACKOFF_BASE_S * (2 ** (job["attempts"] - 1))
            self.conn.execute(
                "UPDATE jobs SET state=?, next_run_at=?, error=?, updated_at=? "
                "WHERE job_id=? AND state=?",
                (QUEUED, now + delay, error[:500], now, job_id, RUNNING),
            )
            self.conn.commit()
            return QUEUED
        self.conn.execute(
            "UPDATE jobs SET state=?, error=?, updated_at=? WHERE job_id=? AND state=?",
            (FAILED, error[:500], now, job_id, RUNNING),
        )
        self.conn.commit()
        return FAILED

    def requeue_stuck_running(self) -> int:
        """Reset jobs left ``running`` by a crashed worker so they are re-claimable on
        restart (crash resumability), returning how many were reset/terminalized.

        A job that hard-crashes the worker PROCESS (segfault in a native ML backend, OOM
        kill, SIGKILL) never reaches :meth:`mark_failed_or_retry`, which is the ONLY place
        ``max_attempts`` was enforced. The old "reset every running row" then re-ran it on
        every restart with attempts climbing past max_attempts UNBOUNDED — an infinite crash
        loop that re-downloads / re-ASRs an expensive ingest forever. So a job that has
        already used all its attempts is terminalized here (-> FAILED) instead of being
        re-queued; the rest are re-queued for immediate re-claim (crash resumability, which
        the worker-drain paths depend on)."""
        now = time.time()
        n = 0
        with self.conn:
            rows = self.conn.execute(
                "SELECT job_id, attempts, max_attempts FROM jobs WHERE state=?", (RUNNING,)
            ).fetchall()
            for r in rows:
                if r["attempts"] >= r["max_attempts"]:
                    self.conn.execute(
                        "UPDATE jobs SET state=?, error=?, updated_at=? "
                        "WHERE job_id=? AND state=?",
                        (FAILED, "worker died; max attempts exhausted", now, r["job_id"], RUNNING),
                    )
                else:
                    self.conn.execute(
                        "UPDATE jobs SET state=?, next_run_at=0, updated_at=? "
                        "WHERE job_id=? AND state=?",
                        (QUEUED, now, r["job_id"], RUNNING),
                    )
                n += 1
        return n


def run_job(mv, job: dict) -> dict:
    """Dispatch a job to the matching SDK call on the worker's OWN Memovox. Each
    kind returns a JSON-serializable result dict."""
    kind = job["kind"]
    args = json.loads(job["args_json"] or "{}")
    if kind == "consolidate":
        return mv.consolidate()
    if kind == "sync":
        return mv.sync().to_dict()
    if kind == "ingest":
        return mv.ingest(args["source"], **{k: v for k, v in args.items() if k != "source"}).__dict__
    raise ValueError(f"Unknown job kind: {kind!r}")


class JobWorker(threading.Thread):
    """A serial (concurrency=1) worker thread draining the job queue. ``once=True``
    drains the queue then stops (for tests / cron); else it polls. Resumable: resets
    stuck ``running`` jobs to ``queued`` on start. Logs to stderr only."""

    def __init__(self, mv, *, once: bool = False, poll_interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.mv = mv
        self.once = once
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self.jobs = JobStore(mv.config)
        self.jobs.requeue_stuck_running()

    def stop(self) -> None:
        self._stop_event.set()

    def drain(self) -> int:
        """Run every currently-due job until none remain; return the count run."""
        ran = 0
        while not self._stop_event.is_set():
            job = self.jobs.claim_next()
            if not job:
                break
            self._execute(job)
            ran += 1
        return ran

    def _execute(self, job: dict) -> None:
        try:
            result = run_job(self.mv, job)
            self.jobs.mark_succeeded(job["job_id"], result if isinstance(result, dict) else {})
        except Exception as exc:  # noqa: BLE001 - record + retry, never crash the worker
            state = self.jobs.mark_failed_or_retry(job["job_id"], f"{type(exc).__name__}: {exc}")
            print(f"job {job['job_id']} ({job['kind']}) failed -> {state}: {exc}", file=sys.stderr)

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.drain()
            if self.once:
                break
            self._stop_event.wait(self.poll_interval)
        self.jobs.close()
