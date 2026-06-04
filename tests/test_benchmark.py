"""M3.4 — backend A/B benchmark harness.

The free row is the frozen snapshot CI gates; upgrade rows auto-shrink away when
their deps are absent (the bare/CI machine). The benchmark adds NO new gated number
beyond the FREE row clearing the existing thresholds.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from eval.harness import (
    FREE_CONFIG,
    BackendConfig,
    _FREE_BACKENDS,
)


class InjectionByteIdentityTest(unittest.TestCase):
    def test_run_eval_default_equals_explicit_free_config(self):
        import json

        from eval.harness import run_eval
        a = run_eval()
        b = run_eval(config=FREE_CONFIG)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


class BackendConfigTest(unittest.TestCase):
    def test_free_config_backend_kwargs(self):
        kw = FREE_CONFIG.backend_kwargs()
        # every *_backend slot in _FREE_BACKENDS is pinned identically
        for k, v in _FREE_BACKENDS.items():
            if k.endswith("_backend"):
                self.assertEqual(kw[k], v, k)

    def test_free_config_covers_every_settings_backend_slot(self):
        # global discipline: a future-added *_backend Settings field must be pinned
        # in the FREE snapshot, or this fails CI.
        import dataclasses

        from memovox.config import Settings
        slots = {f.name for f in dataclasses.fields(Settings) if f.name.endswith("_backend")}
        self.assertTrue(slots)  # sanity
        self.assertEqual(slots - set(FREE_CONFIG.backend_kwargs()), set())

    def test_upgrade_config_overrides_only_named_slots(self):
        cfg = BackendConfig(name="st+deberta", embed_backend="sentence-transformers",
                            nli_backend="deberta-nli")
        kw = cfg.backend_kwargs()
        self.assertEqual(kw["embed_backend"], "sentence-transformers")
        self.assertEqual(kw["nli_backend"], "deberta-nli")
        self.assertEqual(kw["llm_backend"], "none")  # unmentioned slots stay free


class AvailableConfigsTest(unittest.TestCase):
    def test_bare_machine_shrinks_to_free_only(self):
        from unittest import mock

        from eval import harness
        # every optional backend unavailable -> exactly [FREE] (the CI path)
        with mock.patch("memovox.backends.backend_status", return_value={}):
            configs = harness.available_configs()
        self.assertEqual([c.name for c in configs], ["free"])

    def test_upgrade_appears_when_available(self):
        from unittest import mock

        from eval import harness
        status = {"rerank": {"cross-encoder": True}, "embed": {"sentence-transformers": True},
                  "nli": {"deberta-nli": True}}
        with mock.patch("memovox.backends.backend_status", return_value=status):
            names = [c.name for c in harness.available_configs()]
        self.assertEqual(names[0], "free")
        self.assertIn("free+cross-encoder", names)
        self.assertIn("st+deberta", names)
        self.assertEqual(names[1:], sorted(names[1:]))  # deterministic name order

    def test_visual_configs_declared_unrankable(self):
        from eval import harness
        rows = harness.unrankable_configs()
        self.assertTrue(rows)
        for name, reason in rows:
            self.assertIsInstance(reason, str)
            self.assertTrue(reason)  # explicit reason, never silently dropped


class RunBenchmarkTest(unittest.TestCase):
    def test_free_row_metric_identical_to_run_eval(self):
        import json

        from eval.harness import run_benchmark, run_eval
        results = run_benchmark()
        names = [n for n, _ in results]
        self.assertEqual(names[0], "free")  # FREE first
        free_report = dict(results)["free"]
        self.assertEqual(json.dumps(free_report, sort_keys=True),
                         json.dumps(run_eval(), sort_keys=True))

    def test_two_runs_identical(self):
        from eval.harness import _benchmark_json, run_benchmark
        self.assertEqual(_benchmark_json(run_benchmark()), _benchmark_json(run_benchmark()))

    def test_cli_benchmark_json_exits_zero(self):
        from eval.harness import main
        self.assertEqual(main(["--benchmark", "--json"]), 0)

    def test_cli_assert_no_regression_gates_free_row(self):
        from eval.harness import _check_thresholds, main
        self.assertEqual(main(["--benchmark", "--assert-no-regression"]), 0)
        # the gate runs _check_thresholds on the FREE row — a sub-threshold report fails
        bad = {"retrieval": {"hit_rate": 0.0}, "groundedness": 0.0,
               "contradiction": {"f1": 0.0}, "synthesis": {"groundedness": 0.0}}
        self.assertTrue(_check_thresholds(bad))


if __name__ == "__main__":
    unittest.main()
