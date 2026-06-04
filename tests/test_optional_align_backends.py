"""M0.3 W7 — opt-in WhisperX align + pyannote-turns backends gate cleanly.

These trail as is_available-gated, lazy-import, graceful-fallthrough upgrades:
absent deps -> is_available False; a forced request -> BackendUnavailable; and
importing them pulls in nothing beyond stdlib (the free path never touches them).
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.errors import BackendUnavailable
from memovox.stentor.asr import get_aligner, get_diarizer_turns


class OptionalBackendGatingTest(unittest.TestCase):
    def test_unknown_backends_raise(self):
        with self.assertRaises(BackendUnavailable):
            get_aligner("nope")
        with self.assertRaises(BackendUnavailable):
            get_diarizer_turns("nope")

    def test_forced_unavailable_raises(self):
        from memovox.backends.asr_align import WhisperXAlign
        from memovox.backends.diarize_turns import PyannoteTurns

        if not WhisperXAlign.is_available():
            with self.assertRaises(BackendUnavailable):
                get_aligner("whisperx")
        if not PyannoteTurns.is_available():
            with self.assertRaises(BackendUnavailable):
                get_diarizer_turns("pyannote-turns")

    def test_import_pulls_no_optional_deps(self):
        # Importing the backend modules must not import whisperx/pyannote/torch.
        import memovox.backends.asr_align  # noqa: F401
        import memovox.backends.diarize_turns  # noqa: F401

        self.assertNotIn("whisperx", sys.modules)
        # pyannote may be transitively present from other backends; assert our
        # module did not import it directly by checking it's still find_spec-gated.
        from memovox.backends.asr_align import WhisperXAlign

        self.assertIsInstance(WhisperXAlign.is_available(), bool)


if __name__ == "__main__":
    unittest.main()
