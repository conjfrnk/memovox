"""Round-15c robustness: fixes from the third (final) stress panel.

Redaction completeness (digest .md on the filesystem), digest-filename injectivity,
config coercion/validation parity (env + config.json), graph/lexical video scoping,
and self-healing FTS backfill — the DB->filesystem boundary and config layer the
prior panels did not reach.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.config import Config, Settings
from memovox.errors import ConfigError
from memovox.loom.models import Moment, Video
from memovox.loom.store import LoomStore
from memovox.util import digest_filename


class RedactionDigestTest(unittest.TestCase):
    """[0]/[2] delete_video must scrub the on-disk digest; filenames must be injective."""

    def _mv(self, tmp):
        from memovox import Memovox
        return Memovox(store=tmp, llm_backend="none")

    def test_delete_video_removes_digest_file(self):
        tmp = tempfile.mkdtemp()
        mv = self._mv(tmp)
        vtt = pathlib.Path(tmp) / "t.en.vtt"
        vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:09.000\nsecret chunk size is 512.\n",
                       encoding="utf-8")
        rep = mv.ingest(str(vtt), source_url="https://youtu.be/secret12345")
        self.assertTrue(list(mv.config.digests_dir.glob("*.md")), "ingest wrote a digest")
        self.assertTrue(mv.delete_video(rep.video_id))
        self.assertEqual(list(mv.config.digests_dir.glob("*.md")), [],
                         "redaction must leave no digest .md on disk")
        mv.close()

    def test_digest_filename_is_injective_on_case(self):
        # case-/punctuation-distinct (but valid, distinct) ids must not collide on one file
        self.assertNotEqual(digest_filename("yt:abcDEF12345"), digest_filename("yt:abcdef12345"))
        self.assertNotEqual(digest_filename("yt:a_b"), digest_filename("yt:a-b"))

    def test_delete_video_scrubs_downloaded_media_not_siblings(self):
        tmp = tempfile.mkdtemp()
        mv = self._mv(tmp)
        mv.config.ensure()
        media = mv.config.media_dir
        # pre-place downloaded artifacts for THIS yt video + a sibling video's media
        (media / "abc123XYZ_0.mp4").write_text("av-bytes", encoding="utf-8")
        (media / "abc123XYZ_0.info.json").write_text("{}", encoding="utf-8")
        (media / "abc123XYZ_0.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
        (media / "other999id99.mp4").write_text("sibling", encoding="utf-8")
        vtt = pathlib.Path(tmp) / "t.en.vtt"
        vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:09.000\nchunk size is 512.\n",
                       encoding="utf-8")
        rep = mv.ingest(str(vtt), source_url="https://youtu.be/abc123XYZ_0")
        self.assertEqual(rep.video_id, "yt:abc123XYZ_0")
        mv.delete_video(rep.video_id)
        names = {p.name for p in media.iterdir()}
        self.assertNotIn("abc123XYZ_0.mp4", names, "downloaded media must be redacted")
        self.assertNotIn("abc123XYZ_0.info.json", names)
        self.assertNotIn("abc123XYZ_0.en.vtt", names)
        self.assertIn("other999id99.mp4", names, "a sibling video's media must survive")
        mv.close()


class ConfigValidationTest(unittest.TestCase):
    """[3]/[4] env AND config.json values coerce/validate identically (no gate-bypass / crash)."""

    def test_non_finite_env_float_raises_clean(self):
        for bad in ("nan", "inf", "-inf"):
            os.environ["MEMOVOX_ANSWER_RELEVANCE_FLOOR"] = bad
            try:
                with self.assertRaises(ConfigError):
                    Settings.from_env()
            finally:
                del os.environ["MEMOVOX_ANSWER_RELEVANCE_FLOOR"]

    def test_bad_int_env_raises_clean_not_valueerror(self):
        os.environ["MEMOVOX_TOP_K"] = "abc"
        try:
            with self.assertRaises(ConfigError):
                Settings.from_env()
        finally:
            del os.environ["MEMOVOX_TOP_K"]

    def test_config_json_string_values_are_coerced(self):
        s = Settings().merged({"asr_allow_cpu": "false", "top_k": "8", "local_only": "no"})
        self.assertIs(s.asr_allow_cpu, False, "JSON string 'false' must become bool False")
        self.assertIs(s.local_only, False, "JSON string 'no' must become bool False")
        self.assertEqual(s.top_k, 8)
        self.assertIsInstance(s.top_k, int, "JSON string '8' must become int 8")

    def test_typed_overrides_unchanged(self):
        # the SDK kwargs path (already-typed values) must be a no-op coercion
        s = Settings().merged({"llm_backend": "none", "top_k": 5, "local_only": True})
        self.assertEqual(s.llm_backend, "none")
        self.assertEqual(s.top_k, 5)
        self.assertIs(s.local_only, True)


class GraphScopingTest(unittest.TestCase):
    """[1] in-video scoping must match the exact video component, not a string prefix."""

    def test_expand_does_not_leak_prefix_sibling_video(self):
        from memovox.augur.traverse import expand
        tmp = tempfile.TemporaryDirectory()
        store = LoomStore(Config(store=pathlib.Path(tmp.name) / "s").ensure())
        for vid in ("yt:a", "yt:ab"):  # "yt:a" is a strict prefix of "yt:ab"
            store.upsert_video(Video(video_id=vid, source_url="u", title=vid, content_hash=vid))
            m = f"{vid}#m0000"
            store.add_moment(Moment(moment_id=m, video_id=vid, t_start_s=0, t_end_s=1,
                                    transcript="x", index=0))
            from memovox.loom.models import Claim
            store.add_claim(Claim(claim_id=f"{m}.c00", moment_id=m, video_id=vid, text="x",
                                  status="committed"))
        store.add_edge("yt:a#m0000.c00", "SUPPORTS", "yt:ab#m0000.c00",
                       src_type="Claim", dst_type="Claim", video_id="yt:a")
        out = expand(store, ["yt:a#m0000"], rels=["SUPPORTS"], hops=1, video_id="yt:a")
        store.close()
        tmp.cleanup()
        self.assertEqual(out, [], "a sibling video (yt:ab) must NOT leak into yt:a-scoped expansion")


class FtsSelfHealTest(unittest.TestCase):
    """[5] FTS backfill must self-heal (not be a one-shot flag) across a no-fts5 write."""

    def test_backfill_heals_after_flag_set(self):
        tmp = tempfile.TemporaryDirectory()
        cfg = Config(store=pathlib.Path(tmp.name) / "s").ensure()
        store = LoomStore(cfg)
        if not store.fts:
            store.close(); tmp.cleanup(); self.skipTest("fts5 unavailable")
        store.upsert_video(Video(video_id="v", source_url="u", title="t",
                                 content_hash="h", pipeline_version="p"))
        store.add_moment(Moment(moment_id="v#m0", video_id="v", t_start_s=0, t_end_s=1,
                                transcript="alpha beta gamma", index=0), [1.0, 0.0])
        store.set_meta("fts_backfilled", "1")  # simulate the OLD sticky one-shot flag
        store.close()
        # mimic a moment written on a no-fts5 build: drop the index while the flag stays set
        raw = sqlite3.connect(str(cfg.db_path))
        raw.execute("DROP TABLE IF EXISTS moments_fts")
        raw.commit(); raw.close()
        store2 = LoomStore(cfg)  # must self-heal despite the stale flag
        try:
            self.assertTrue(store2.lexical_search("alpha"),
                            "FTS must reconcile even with the legacy flag set")
        finally:
            store2.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
