"""M2.3 — answer-with-video clip stitching (spec §5/§8).

Pure, deterministic arithmetic over cited spans: merge adjacent/overlapping
windows per video into minimal deep-linked clips. The free path adds Answer.clips
without touching text/citations; render_clip is an opt-in ffmpeg no-op on CI.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.augur import Clip, stitch_clips
from memovox.augur.types import Citation
from memovox.loom.models import Video
from memovox.util import deep_link, deep_link_range


class DeepLinkRangeTest(unittest.TestCase):
    def test_youtube_ranged(self):
        self.assertEqual(
            deep_link_range("https://youtu.be/abc123", 750.4, 845.9),
            "https://www.youtube.com/watch?v=abc123&start=750&end=845")

    def test_non_youtube_falls_back_to_start_only(self):
        url = "https://example.com/talk.mp4"
        self.assertEqual(deep_link_range(url, 10.0, 40.0), deep_link(url, 10.0))

    def test_none_url(self):
        self.assertIsNone(deep_link_range(None, 1.0, 2.0))


class ClipTypeTest(unittest.TestCase):
    def test_clip_to_dict(self):
        c = Clip(video_id="yt:a", t_start_s=0.0, t_end_s=60.0, title="talk",
                 deep_link="https://www.youtube.com/watch?v=a&start=0&end=60",
                 citation_indices=[1, 2])
        d = c.to_dict()
        self.assertEqual(d["duration_s"], 60.0)
        self.assertEqual(d["citation_indices"], [1, 2])
        self.assertEqual(d["video_id"], "yt:a")


def _cit(i, video_id, t0, t1):
    return Citation(index=i, video_id=video_id, moment_id=f"{video_id}#m{i}",
                    t_start_s=t0, t_end_s=t1, title="talk")


class StitchTest(unittest.TestCase):
    def setUp(self):
        self.videos = {"yt:a": Video("yt:a", "https://youtu.be/a", "talk a"),
                       "yt:b": Video("yt:b", "https://youtu.be/b", "talk b")}

    def test_merges_adjacent_and_overlapping_per_video(self):
        cits = [_cit(1, "yt:a", 0, 30), _cit(2, "yt:a", 28, 60), _cit(3, "yt:a", 200, 240)]
        clips = stitch_clips(cits, videos=self.videos, merge_gap_s=2.5)
        self.assertEqual(len(clips), 2)
        self.assertEqual((clips[0].t_start_s, clips[0].t_end_s), (0, 60))
        self.assertEqual(clips[0].citation_indices, [1, 2])
        self.assertEqual((clips[1].t_start_s, clips[1].t_end_s), (200, 240))

    def test_does_not_merge_across_videos(self):
        cits = [_cit(1, "yt:a", 0, 30), _cit(2, "yt:b", 10, 40)]
        clips = stitch_clips(cits, videos=self.videos, merge_gap_s=2.5)
        self.assertEqual(len(clips), 2)
        self.assertEqual({c.video_id for c in clips}, {"yt:a", "yt:b"})

    def test_clips_non_overlapping_and_sorted(self):
        cits = [_cit(1, "yt:a", 100, 130), _cit(2, "yt:a", 0, 30), _cit(3, "yt:a", 200, 240)]
        clips = stitch_clips(cits, videos=self.videos, merge_gap_s=2.5)
        starts = [c.t_start_s for c in clips]
        self.assertEqual(starts, sorted(starts))
        for a, b in zip(clips, clips[1:]):
            self.assertLessEqual(a.t_end_s, b.t_start_s)

    def test_idempotent(self):
        cits = [_cit(1, "yt:a", 0, 30), _cit(2, "yt:a", 28, 60)]
        once = stitch_clips(cits, videos=self.videos, merge_gap_s=2.5)
        twice = stitch_clips(
            [_cit(c.citation_indices[0], c.video_id, c.t_start_s, c.t_end_s) for c in once],
            videos=self.videos, merge_gap_s=2.5)
        self.assertEqual([(c.t_start_s, c.t_end_s, c.video_id) for c in once],
                         [(c.t_start_s, c.t_end_s, c.video_id) for c in twice])

    def test_ranged_deep_link_on_clip(self):
        clips = stitch_clips([_cit(1, "yt:a", 5, 65)], videos=self.videos, merge_gap_s=2.5)
        self.assertEqual(clips[0].deep_link, "https://www.youtube.com/watch?v=a&start=5&end=65")


class RenderClipTest(unittest.TestCase):
    def setUp(self):
        self.video = Video("yt:a", "https://youtu.be/a", "talk a")
        self.clip = Clip(video_id="yt:a", t_start_s=10.0, t_end_s=40.0)

    def test_noop_without_media_or_ffmpeg(self):
        from memovox.augur import render_clip
        # no media_path -> None, writes nothing (the CI/free path)
        self.assertIsNone(render_clip(self.video, self.clip, media_path=None, out_dir="/tmp/x"))
        self.assertIsNone(render_clip(self.video, self.clip,
                                      media_path="/does/not/exist.mp4", out_dir="/tmp/x"))

    def test_builds_ffmpeg_cmd(self):
        import tempfile
        from unittest import mock
        from memovox.augur import render_clip

        with tempfile.TemporaryDirectory() as tmp:
            media = pathlib.Path(tmp) / "src.mp4"
            media.write_bytes(b"fake")
            out_dir = pathlib.Path(tmp) / "out"
            captured = {}

            def _fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                captured["kwargs"] = kwargs
                (out_dir / "yt-a_10-40.mp4").write_bytes(b"clip")
                return mock.Mock(returncode=0)

            with mock.patch("memovox.audio.which_ffmpeg", return_value="/usr/bin/ffmpeg"), \
                    mock.patch("subprocess.run", side_effect=_fake_run):
                out = render_clip(self.video, self.clip, media_path=str(media), out_dir=out_dir)
            cmd = captured["cmd"]
            self.assertEqual(cmd[0], "/usr/bin/ffmpeg")
            self.assertIn("10.0", cmd)   # t_start
            self.assertIn("40.0", cmd)   # t_end
            # -ss AND -to MUST precede -i: as input options they seek to an ABSOLUTE
            # input timestamp, so the cut spans [t0, t1] of the source. Moving -to
            # after -i silently makes it output-relative (wrong, huge duration).
            self.assertLess(cmd.index("-ss"), cmd.index("-i"))
            self.assertLess(cmd.index("-to"), cmd.index("-i"))
            self.assertEqual(captured["kwargs"]["stdout"], __import__("subprocess").DEVNULL)
            self.assertIn("timeout", captured["kwargs"])  # never hangs CI
            self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
