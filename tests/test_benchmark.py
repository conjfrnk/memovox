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


if __name__ == "__main__":
    unittest.main()
