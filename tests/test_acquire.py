"""M3.2 W1 — enumerate_source: flat-playlist expansion (metadata-only, no download).

All yt-dlp interaction is monkeypatched; make test never touches the network.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.config import Config
from memovox.errors import AcquisitionError
from memovox.stentor.acquire import EnumeratedEntry, enumerate_source

_PLAYLIST_JSON = json.dumps({
    "id": "PL123", "title": "My Playlist",
    "entries": [
        {"id": "abc123", "url": "https://www.youtube.com/watch?v=abc123", "title": "Video 1"},
        {"id": "def456", "url": "https://www.youtube.com/watch?v=def456", "title": "Video 2"},
        {"id": "ghi789", "url": "ghi789", "title": "Video 3"},  # bare id form
    ],
})
_SINGLE_JSON = json.dumps({"id": "solo99", "title": "Solo", "webpage_url": "https://youtu.be/solo99"})


class EnumerateSourceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, stdout, *, which="/usr/bin/yt-dlp", returncode=0):
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return mock.Mock(returncode=returncode, stdout=stdout, stderr="")

        with mock.patch("shutil.which", return_value=which), \
                mock.patch("subprocess.run", side_effect=_fake_run):
            entries = enumerate_source(self.config, "https://www.youtube.com/playlist?list=PL123")
        return entries, captured.get("cmd")

    def test_expands_playlist_metadata_only(self):
        entries, cmd = self._run(_PLAYLIST_JSON)
        self.assertEqual(len(entries), 3)
        self.assertIsInstance(entries[0], EnumeratedEntry)
        self.assertEqual([e.video_id for e in entries], ["yt:abc123", "yt:def456", "yt:ghi789"])
        self.assertEqual(entries[2].url, "https://youtu.be/ghi789")  # bare id -> watch url
        # metadata-only: --flat-playlist present, NO download flags
        self.assertIn("--flat-playlist", cmd)
        for forbidden in ("-f", "bestaudio/best", "-o", "--write-subs"):
            self.assertNotIn(forbidden, cmd)

    def test_single_video_enumerates_to_one(self):
        entries, _ = self._run(_SINGLE_JSON)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].video_id, "yt:solo99")

    def test_missing_yt_dlp_raises(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(AcquisitionError):
                enumerate_source(self.config, "https://x/playlist")


if __name__ == "__main__":
    unittest.main()
