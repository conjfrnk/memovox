"""M3.3 — stdlib SQLite-backed job runner (queued→running→succeeded|failed).

threading + sqlite3 only; no broker, no framework. Default serial (concurrency=1).
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.config import Config
from memovox.serving.jobs import JobStore, _args_hash


class JobStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()
        self.jobs = JobStore(self.config)

    def tearDown(self):
        self.jobs.close()
        self._tmp.cleanup()

    def test_enqueue_returns_queued_job(self):
        jid = self.jobs.enqueue("consolidate", {})
        job = self.jobs.get_job(jid)
        self.assertEqual(job["state"], "queued")
        self.assertEqual(job["attempts"], 0)
        self.assertEqual(job["kind"], "consolidate")

    def test_idempotent_dedup_while_nonterminal(self):
        a = self.jobs.enqueue("consolidate", {"topic": "x"})
        b = self.jobs.enqueue("consolidate", {"topic": "x"})
        self.assertEqual(a, b)  # same non-terminal (kind, args_hash) -> same job
        rows = self.jobs.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        self.assertEqual(rows, 1)

    def test_args_hash_is_order_independent(self):
        a = self.jobs.enqueue("ingest", {"a": 1, "b": 2})
        b = self.jobs.enqueue("ingest", {"b": 2, "a": 1})  # same args, different order
        self.assertEqual(a, b)

    def test_distinct_args_are_distinct_jobs(self):
        a = self.jobs.enqueue("ingest", {"url": "x"})
        b = self.jobs.enqueue("ingest", {"url": "y"})
        self.assertNotEqual(a, b)

    def test_claim_next_never_double_claims(self):
        self.jobs.enqueue("consolidate", {})
        first = self.jobs.claim_next()
        self.assertIsNotNone(first)
        self.assertEqual(first["state"], "running")
        # a second claim finds no due queued job (the only one is now running)
        self.assertIsNone(self.jobs.claim_next())

    def test_reenqueue_after_terminal_creates_new_job(self):
        a = self.jobs.enqueue("consolidate", {})
        self.jobs.conn.execute("UPDATE jobs SET state='succeeded' WHERE job_id=?", (a,))
        self.jobs.conn.commit()
        b = self.jobs.enqueue("consolidate", {})  # prior is terminal -> fresh job
        self.assertNotEqual(a, b)

    def test_concurrent_enqueue_dedups_atomically(self):
        # Separate connections (as the ThreadingHTTPServer gives each request) racing the
        # same (kind, args) must yield ONE row, not N — the partial UNIQUE index enforces it.
        import threading
        from memovox.serving.jobs import JobStore
        stores = [JobStore(self.config) for _ in range(12)]
        barrier = threading.Barrier(len(stores))
        ids: list = []
        lock = threading.Lock()

        def worker(js):
            barrier.wait()  # align the enqueue window to maximize the race
            jid = js.enqueue("ingest", {"source": "https://youtu.be/x"})
            with lock:
                ids.append(jid)

        threads = [threading.Thread(target=worker, args=(js,)) for js in stores]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for js in stores:
            js.close()
        rows = self.jobs.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind='ingest'").fetchone()[0]
        self.assertEqual(rows, 1, "concurrent identical enqueues must create exactly one job")
        self.assertEqual(len(set(ids)), 1, "all callers must get the same job id")

    def test_legacy_duplicate_active_jobs_dont_brick_init(self):
        # A pre-round-15 store could hold duplicate ACTIVE (kind, args_hash) rows (the old
        # enqueue race). Opening a JobStore must dedup them, not raise on CREATE UNIQUE INDEX.
        self.jobs.conn.execute("DROP INDEX IF EXISTS idx_jobs_active_dedup")  # mimic legacy schema
        h = _args_hash({"source": "x"})
        for n, jid in enumerate(("j1", "j2", "j3")):
            self.jobs.conn.execute(
                "INSERT INTO jobs(job_id,kind,args_hash,args_json,state,max_attempts,created_at,updated_at) "
                "VALUES(?,?,?,?,?,3,?,?)", (jid, "ingest", h, "{}", "queued", float(n), float(n)))
        self.jobs.conn.commit()
        js2 = JobStore(self.config)  # __init__ must dedup then build the index without raising
        active = js2.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind='ingest' AND state IN ('queued','running')"
        ).fetchone()[0]
        js2.close()
        self.assertEqual(active, 1, "duplicate active jobs must collapse to a single keeper")

    def test_enqueue_never_returns_phantom_id_under_vanishing_race(self):
        # If the active duplicate terminalizes between INSERT OR IGNORE and the fallback
        # lookup, enqueue must RETRY (insert a real job), never hand back an un-inserted uuid.
        h = _args_hash({"source": "x"})
        self.jobs.conn.execute(
            "INSERT INTO jobs(job_id,kind,args_hash,args_json,state,max_attempts,created_at,updated_at) "
            "VALUES('dup','ingest',?,?, 'queued',3,1.0,1.0)", (h, "{}"))
        self.jobs.conn.commit()
        real = self.jobs.conn

        class _Proxy:  # sqlite3.Connection.execute is read-only, so wrap the whole connection
            def __init__(self, conn):
                self._conn = conn
                self._fired = False

            def execute(self, sql, params=()):
                if not self._fired and sql.strip().startswith("SELECT job_id FROM jobs WHERE kind"):
                    self._conn.execute("UPDATE jobs SET state='succeeded' WHERE job_id='dup'")
                    self._fired = True  # the active dup "vanishes" right as enqueue looks it up
                return self._conn.execute(sql, params)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        self.jobs.conn = _Proxy(real)
        try:
            jid = self.jobs.enqueue("ingest", {"source": "x"})
        finally:
            self.jobs.conn = real
        job = self.jobs.get_job(jid)
        self.assertIsNotNone(job, "enqueue must never return a phantom (un-inserted) id")
        self.assertEqual(job["state"], "queued", "the work must actually be queued")

    def test_requeue_stuck_terminalizes_exhausted_attempts(self):
        # A job whose worker process hard-crashes (never records failure) must NOT re-run
        # forever: once it has used all its attempts requeue terminalizes it (-> FAILED)
        # instead of looping; a job with attempts remaining is still re-queued for retry.
        exhausted = self.jobs.enqueue("consolidate", {"a": 1}, max_attempts=2)
        retriable = self.jobs.enqueue("consolidate", {"a": 2}, max_attempts=3)
        self.jobs.conn.execute("UPDATE jobs SET state='running', attempts=2 WHERE job_id=?",
                               (exhausted,))
        self.jobs.conn.execute("UPDATE jobs SET state='running', attempts=1 WHERE job_id=?",
                               (retriable,))
        self.jobs.conn.commit()
        self.jobs.requeue_stuck_running()
        ej, rj = self.jobs.get_job(exhausted), self.jobs.get_job(retriable)
        self.assertEqual(ej["state"], "failed")           # at max_attempts -> terminal, no loop
        self.assertEqual(rj["state"], "queued")           # attempts remain -> still re-claimable


class JobWorkerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        from memovox import Memovox
        self.mv = Memovox(store=self.dir / "store", llm_backend="none")
        vtt = self.dir / "t.en.vtt"
        vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:09.000\nThe chunk size is 512 tokens.\n",
                       encoding="utf-8")
        self.mv.ingest(str(vtt), source_url="https://youtu.be/abc123")

    def tearDown(self):
        self._tmp.cleanup()

    def test_worker_runs_queued_consolidate(self):
        from memovox.serving.jobs import JobStore, JobWorker
        inline = self.mv.consolidate()
        jobs = JobStore(self.mv.config)
        jid = jobs.enqueue("consolidate", {})
        JobWorker(self.mv, once=True).drain()
        job = jobs.get_job(jid)
        jobs.close()
        self.assertEqual(job["state"], "succeeded")
        import json
        self.assertEqual(json.loads(job["result_json"]).keys(), inline.keys())  # same shape

    def test_failed_job_retries_then_exhausts(self):
        from memovox.serving import jobs as jobs_mod
        from memovox.serving.jobs import JobStore, JobWorker
        jobs = JobStore(self.mv.config)
        jid = jobs.enqueue("consolidate", {}, max_attempts=2)
        worker = JobWorker(self.mv, once=True)
        with mock.patch.object(jobs_mod, "run_job", side_effect=RuntimeError("boom")):
            worker._execute(jobs.claim_next())  # attempt 1 -> requeued (backoff)
            self.assertEqual(jobs.get_job(jid)["state"], "queued")
            self.assertEqual(jobs.get_job(jid)["attempts"], 1)
            # force it due now, fail again -> exhausted -> failed
            jobs.conn.execute("UPDATE jobs SET next_run_at=0 WHERE job_id=?", (jid,))
            jobs.conn.commit()
            worker._execute(jobs.claim_next())
        job = jobs.get_job(jid)
        jobs.close()
        self.assertEqual(job["state"], "failed")
        self.assertIn("boom", job["error"])

    def test_auto_worker_is_single_and_stoppable(self):
        # enqueue auto-spawns ONE worker; a 2nd enqueue reuses it (no duplicate
        # thread under the lock); close() stops it cleanly.
        self.mv.enqueue_consolidate()
        w1 = self.mv._worker
        self.assertIsNotNone(w1)
        self.assertTrue(w1.is_alive())
        self.mv._ensure_worker()  # idempotent
        self.assertIs(self.mv._worker, w1)  # same worker, not a duplicate
        self.mv.close()
        self.assertIsNone(self.mv._worker)
        w1.join(timeout=5)
        self.assertFalse(w1.is_alive())  # stopped, thread + connection released

    def test_worker_resumable_resets_stuck_running(self):
        from memovox.serving.jobs import JobStore, JobWorker
        jobs = JobStore(self.mv.config)
        jid = jobs.enqueue("consolidate", {})
        jobs.conn.execute("UPDATE jobs SET state='running' WHERE job_id=?", (jid,))
        jobs.conn.commit()
        JobWorker(self.mv, once=True)  # __init__ requeues stuck running
        self.assertEqual(jobs.get_job(jid)["state"], "queued")
        jobs.close()


if __name__ == "__main__":
    import unittest.mock  # noqa: F401
    unittest.main()
