"""models_dir override: the model CACHE should be separable from the data STORE, so a
fresh store (which the stress driver wipes each run) doesn't force a multi-GB re-download
of BGE-M3 / cross-encoder. Default behavior (store/models) is unchanged."""
import os
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.config import Config


class ModelsDirOverrideTest(unittest.TestCase):
    def test_default_is_store_subdir(self):
        # Hermetic regardless of the ambient environment: clear any inherited
        # MEMOVOX_MODELS_DIR so this asserts the genuine DEFAULT (store/models).
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("MEMOVOX_MODELS_DIR", None)
            cfg = Config(store="/tmp/some_store_xyz")
            self.assertEqual(cfg.models_dir, cfg.store / "models")

    def test_env_override_points_outside_store(self):
        with mock.patch.dict(os.environ, {"MEMOVOX_MODELS_DIR": "/tmp/shared_mv_models"}):
            cfg = Config(store="/tmp/some_store_xyz")
            self.assertEqual(cfg.models_dir, pathlib.Path("/tmp/shared_mv_models"))


if __name__ == "__main__":
    unittest.main()
