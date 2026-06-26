"""Round-15 robustness: numerics (NaN/Inf), synthesize DoS cap, sync-cursor atomicity.

Dimensions the 41-video free-path stress driver never exercises — surfaced by the
adversarial stress panel and confirmed reproducible before fixing.
"""

from __future__ import annotations

import itertools
import pathlib
import sqlite3
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox import sync_state
from memovox.config import Config
from memovox.loom.backends.sqlite import SqliteVectorIndex
from memovox.loom.store import LoomStore
from memovox.vectormath import pack_floats


class VectorIndexNonFiniteTest(unittest.TestCase):
    """[9] NaN/Inf must not defeat the (-score, id) determinism guarantee of the per-leg sort."""

    def _db(self, ordered_rows):
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.execute("CREATE TABLE vectors(moment_id TEXT PRIMARY KEY, dim INT, vec BLOB, space TEXT)")
        for mid, vec in ordered_rows:
            c.execute("INSERT INTO vectors VALUES (?,?,?,?)", (mid, len(vec), pack_floats(vec), "text"))
        return c

    def test_nan_stored_vector_does_not_poison_sort(self):
        rows = {"v#m0": [1.0, 0.0], "v#m1": [float("nan"), 0.0],
                "v#m2": [0.8, 0.6], "v#m3": [0.6, 0.8]}
        results = set()
        for order in itertools.permutations(rows):
            conn = self._db([(m, rows[m]) for m in order])
            idx = SqliteVectorIndex(conn, normalize_vectors=True)
            results.add(tuple(m for m, _ in idx.search([1.0, 0.0], top_k=3)))
        self.assertEqual(len(results), 1,
                         "top-k must be identical regardless of DB row order (NaN dropped, not sorted)")
        self.assertNotIn("v#m1", results.pop(), "the NaN row must not float into the results")

    def test_nonfinite_query_vector_returns_empty(self):
        idx = SqliteVectorIndex(self._db([("v#m0", [1.0, 0.0])]), normalize_vectors=True)
        self.assertEqual(idx.search([float("nan"), 0.0], top_k=3), [])
        self.assertEqual(idx.search([float("inf"), 1.0], top_k=3), [])


class SynthesizeClusterCapTest(unittest.TestCase):
    """[5] a giant union-find consensus cluster must not drive uncapped O(k^2) NLI."""

    def test_large_cluster_bounds_pairwise_nli(self):
        from memovox.augur.synthesize import _CLUSTER_MEMBER_CAP, synthesize
        from memovox.backends.embed import HashingEmbedder
        from memovox.backends.nli import LexicalNLI
        from memovox.loom.models import Claim, Moment, Video

        class CountingNLI(LexicalNLI):
            calls = 0

            def classify(self, p, h):
                CountingNLI.calls += 1
                return super().classify(p, h)

        tmp = tempfile.TemporaryDirectory()
        cfg = Config(store=pathlib.Path(tmp.name) / "s").ensure()
        store = LoomStore(cfg)
        emb = HashingEmbedder(dim=128)
        nli = CountingNLI()
        text = "Chinchilla scaling laws require substantially more training tokens for optimal compute."
        n = _CLUSTER_MEMBER_CAP + 60  # exceed the cap so it must engage
        for i in range(n):
            v = f"vid{i:04d}"
            store.upsert_video(Video(video_id=v, source_url="https://x/" + v, title=v, content_hash=v))
            m = v + "#m0000"
            store.add_moment(Moment(moment_id=m, video_id=v, t_start_s=0.0, t_end_s=10.0,
                                    transcript=text, speaker_id="spk_0", index=0), emb.embed_one(text))
            store.add_claim(Claim(claim_id=v + ".c0", moment_id=m, video_id=v, text=text,
                                  subject=text, salience=0.6, t_start_s=0.0, t_end_s=10.0,
                                  speaker_id="spk_0", status="committed"))
        syn = synthesize(store, "Chinchilla scaling training tokens compute", nli=nli)
        store.close()
        tmp.cleanup()
        uncapped = n * (n - 1) // 2                 # what _cluster_contradicts would walk uncapped
        bound = _CLUSTER_MEMBER_CAP * (_CLUSTER_MEMBER_CAP - 1) // 2 + 2 * n
        self.assertLess(CountingNLI.calls, uncapped, "the member cap must reduce the pair count")
        self.assertLessEqual(CountingNLI.calls, bound, "NLI work is bounded by C(cap,2)")
        self.assertFalse(syn.low_evidence)            # a genuine on-topic synthesis was produced
        self.assertTrue(syn.citations)                # ...with citations, not an empty refusal


class SyncCursorAtomicTest(unittest.TestCase):
    """[7] concurrent cursor writers must not lose ids (lost-update -> needless re-ingest)."""

    def test_concurrent_mark_seen_preserves_all_ids(self):
        tmp = tempfile.TemporaryDirectory()
        cfg = Config(store=pathlib.Path(tmp.name) / "s").ensure()
        LoomStore(cfg).close()  # initialize the db file
        url = "https://www.youtube.com/@chan"
        errs = []

        def worker(i):
            try:
                st = LoomStore(cfg)
                sync_state.mark_seen(st, url, f"vid{i}")
                st.close()
            except Exception as exc:  # noqa: BLE001
                errs.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        st = LoomStore(cfg)
        seen = sync_state.seen_ids(st, url)
        st.close()
        tmp.cleanup()
        self.assertEqual(errs, [], "no writer should error")
        self.assertEqual(len(seen), 20, "every concurrently-marked id must survive (no lost update)")

    def test_append_meta_robust_to_open_transaction(self):
        import json
        tmp = tempfile.TemporaryDirectory()
        cfg = Config(store=pathlib.Path(tmp.name) / "s").ensure()
        store = LoomStore(cfg)
        try:
            # leave an UNCOMMITTED write pending so the connection is mid-transaction; the
            # bare BEGIN IMMEDIATE would otherwise raise "transaction within a transaction".
            store.conn.execute("INSERT INTO meta (key, value) VALUES ('pending', '1')")
            self.assertTrue(store.conn.in_transaction)
            store.append_meta_json_id("k", "vid1")  # must not raise
            self.assertEqual(json.loads(store.get_meta("k")), ["vid1"])
        finally:
            store.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
