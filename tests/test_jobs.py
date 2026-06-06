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
from memovox.serving.jobs import JobStore


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
