"""M0.2 W7 — the opt-in scale harness runs, and ANN backends gate cleanly.

eval/scale.py is NEVER part of the CI gate (not imported by eval/harness.py); this
thin test only proves it runs at small N and that optional ANN vector backends
raise BackendUnavailable (never crash, never silently fall into the gate) when
their dependency is absent.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.errors import BackendUnavailable
from memovox.loom.backends import get_vector_index


class ScaleHarnessTest(unittest.TestCase):
    def test_scale_harness_runs_small_n(self):
        from eval.scale import run

        r = run(n=200, dim=32, k=10, queries=8, seed=7)
        self.assertEqual(r["n"], 200)
        self.assertGreaterEqual(r["p95_ms"], 0.0)         # emits a p95 latency number
        self.assertEqual(r["recall_at_k"], 1.0)            # exact index vs itself == 1.0


class AnnBackendGatingTest(unittest.TestCase):
    def test_ann_backends_unavailable_by_default(self):
        for name in ("lance", "qdrant"):
            with self.assertRaises(BackendUnavailable):
                get_vector_index(name, conn=None)

    def test_sqlite_still_resolves(self):
        from memovox.loom.backends.sqlite import SqliteVectorIndex

        self.assertIsInstance(get_vector_index("sqlite", conn=None), SqliteVectorIndex)


if __name__ == "__main__":
    unittest.main()
