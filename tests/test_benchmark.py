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


if __name__ == "__main__":
    unittest.main()
