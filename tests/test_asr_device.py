"""M0.3 W4 — fail loud on silent CPU placement of a heavy ASR model (spec §9)."""

from __future__ import annotations

import os
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import memovox.backends.asr_whisper as whisper_mod
from memovox.backends.asr_whisper import WhisperASR, _check_device_placement
from memovox.config import Settings
from memovox.errors import DevicePlacementError, MemovoxError


class DeviceGuardTest(unittest.TestCase):
    def test_load_raises_on_silent_cpu_for_large_model(self):
        with mock.patch.object(whisper_mod, "_cuda_available", return_value=False):
            asr = WhisperASR(config=None, model="large-v3", device="auto", allow_cpu=False)
            with self.assertRaises(DevicePlacementError):
                asr._load()  # raises BEFORE importing faster_whisper

    def test_device_placement_error_is_memovox_error(self):
        self.assertTrue(issubclass(DevicePlacementError, MemovoxError))

    def test_allow_cpu_escape_does_not_trip_guard(self):
        # no raise even on CPU + large when allow_cpu is set
        _check_device_placement("large-v3", "cpu", True)

    def test_explicit_cpu_large_trips(self):
        with self.assertRaises(DevicePlacementError):
            _check_device_placement("large-v3", "cpu", False)

    def test_small_model_never_trips(self):
        for small in ("tiny", "base", "small"):
            _check_device_placement(small, "cpu", False)  # no raise

    def test_cuda_device_does_not_trip(self):
        _check_device_placement("large-v3", "cuda", False)  # explicit cuda -> fine


class SettingsAsrDeviceTest(unittest.TestCase):
    def test_defaults(self):
        s = Settings()
        self.assertEqual(s.asr_device, "auto")
        self.assertEqual(s.asr_compute_type, "default")
        self.assertFalse(s.asr_allow_cpu)

    def test_env_coercion(self):
        with mock.patch.dict(os.environ, {"MEMOVOX_ASR_ALLOW_CPU": "1"}):
            self.assertTrue(Settings.from_env().asr_allow_cpu)


if __name__ == "__main__":
    unittest.main()
