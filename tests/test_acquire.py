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
from memovox.stentor.acquire import EnumeratedEntry, acquire, enumerate_source

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


class AcquireUrlTest(unittest.TestCase):
    """_acquire_url must (a) probe the real stream for is_video instead of
    hardcoding True (so an audio-only download doesn't make Tessera shell ffmpeg
    onto a streamless file), and (b) request a video format only on --with-video."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "s").ensure()

    def tearDown(self):
        self._tmp.cleanup()

    def _fake_download(self, *, ext="m4a"):
        """Return a subprocess.run stub that writes a fake media file + info.json."""
        def _run(cmd, **kwargs):
            (self.config.media_dir / f"vid1.{ext}").write_bytes(b"\x00\x00")
            (self.config.media_dir / "vid1.info.json").write_text(
                json.dumps({"id": "vid1", "title": "T", "webpage_url": "https://youtu.be/vid1"})
            )
            self._cmd = cmd
            return mock.Mock(returncode=0, stdout="", stderr="")
        return _run

    def _acquire(self, *, has_video, want_video=False):
        with mock.patch("shutil.which", return_value="/usr/bin/yt-dlp"), \
                mock.patch("subprocess.run", side_effect=self._fake_download()), \
                mock.patch("memovox.audio.probe",
                           return_value={"has_video": has_video, "has_audio": True,
                                         "duration": 5.0, "codecs": []}):
            return acquire(self.config, "https://youtu.be/vid1", want_video=want_video)

    def test_audio_only_download_is_not_video(self):
        meta = self._acquire(has_video=False)
        self.assertFalse(meta.is_video)
        # default (audio) request must NOT ask yt-dlp for video
        self.assertIn("bestaudio/best", self._cmd)

    def test_video_present_is_detected(self):
        meta = self._acquire(has_video=True, want_video=True)
        self.assertTrue(meta.is_video)
        # --with-video must request a video+audio format, not bestaudio-only
        joined = " ".join(self._cmd)
        self.assertIn("bv", joined)
        self.assertNotEqual(self._cmd[self._cmd.index("-f") + 1], "bestaudio/best")


if __name__ == "__main__":
    unittest.main()
