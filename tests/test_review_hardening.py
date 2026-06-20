"""Regression tests for the multi-agent adversarial review hardening pass (2026-06-20).

Each test pins a defect surfaced and execution-verified by the 5-dimension review
panel (ingestion robustness, provenance integrity, determinism, state integrity,
serving/security). Grouped by dimension; every test fails on the pre-fix code.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import time
import unittest


# --------------------------------------------------------------------------- #
# Ingestion & transcript-parsing robustness
# --------------------------------------------------------------------------- #


class TestIngestRobustness(unittest.TestCase):
    def test_url_paren_strip_is_linear_not_redos(self):
        # F1: _URL_PAREN_RE was O(n^2) on "(http://x" repeated (a long cue hangs).
        from memovox.stentor import transcript as T

        payload = "(https://x" * 20000
        t0 = time.perf_counter()
        T._URL_PAREN_RE.sub(" ", payload)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.5, f"URL-strip took {elapsed:.3f}s — quadratic ReDoS")

    def test_url_paren_strip_correctness_preserved(self):
        from memovox.stentor import transcript as T

        self.assertEqual(T._URL_PAREN_RE.sub(" ", "a](https://example.com/x) b"), "a  b")
        # bare (url)
        self.assertEqual(T._URL_PAREN_RE.sub(" ", "wiki (https://en.wikipedia.org/Foo) end"),
                         "wiki   end")

    def test_youtube_id_rejects_lookalike_hosts(self):
        # F2: str.lstrip("www.") collapsed ww./wyoutube/etc onto youtube.com.
        from memovox import util

        for bad in ("https://ww.youtube.com/watch?v=PWNED",
                    "https://wyoutube.com/watch?v=PWNED",
                    "https://wwwyoutube.com/watch?v=PWNED",
                    "https://w.youtube.com/watch?v=PWNED"):
            self.assertIsNone(util.youtube_id(bad), f"misidentified {bad!r} as YouTube")
        # genuine hosts still work
        self.assertEqual(util.youtube_id("https://www.youtube.com/watch?v=REAL"), "REAL")
        self.assertEqual(util.youtube_id("https://youtube.com/watch?v=REAL"), "REAL")
        self.assertEqual(util.youtube_id("https://m.youtube.com/watch?v=REAL"), "REAL")
        self.assertEqual(util.youtube_id("https://youtu.be/REAL"), "REAL")

    def test_lookalike_host_does_not_rewrite_deeplink(self):
        from memovox import util

        # a non-YouTube source keeps its own domain in the deep link (no rewrite to youtu.be)
        link = util.deep_link("https://ww.youtube.com/watch?v=PWNED", 42)
        self.assertNotIn("youtu.be/PWNED", link)
        self.assertIn("ww.youtube.com", link)

    def test_malformed_but_valid_json_transcript_does_not_crash(self):
        # F3: list-of-strings / null items / non-numeric timings raised uncaught.
        from memovox.stentor import transcript as T

        for payload in (["line one", "line two"],
                        [{"start": [1, 2], "text": "hi"}],
                        {"segments": [None, {"start": 0, "text": "x"}]},
                        [{"text": "ok", "words": ["nope", None, {"word": "w"}]}],
                        "not even a list"):
            p = pathlib.Path(tempfile.mktemp(suffix=".json"))
            p.write_text(json.dumps(payload), encoding="utf-8")
            try:
                segs = T.load_transcript(p)  # must not raise
            except Exception as exc:  # noqa: BLE001
                self.fail(f"load_transcript crashed on {payload!r}: {type(exc).__name__}: {exc}")
            self.assertIsInstance(segs, list)
        # the one well-formed segment in the dict case is recovered
        p = pathlib.Path(tempfile.mktemp(suffix=".json"))
        p.write_text(json.dumps({"segments": [None, {"start": 0, "text": "real text"}]}),
                     encoding="utf-8")
        segs = T.load_transcript(p)
        self.assertEqual([s.text for s in segs], ["real text"])

    def test_html_entity_annotations_are_decoded_before_event_strip(self):
        # F4: &#91;applause&#93; / &#9834; survived still-escaped into claim text.
        from memovox.stentor import transcript as T

        # entity-encoded [applause] becomes a timeline event, not speech
        text, events = T.clean_text("And then &#91;applause&#93; the crowd.")
        self.assertNotIn("[applause]", text)
        self.assertIn("applause", events)
        # entity-encoded music note marks the line as a [music] event (no speech emitted)
        out = T.clean_segments([T.Segment(0, 3, "&#9834; secret lyrics &#9834;")])
        kinds = [(s.kind, s.text) for s in out]
        self.assertTrue(any(k == "event" and t == "[music]" for k, t in kinds), kinds)
        self.assertFalse(any(k == "speech" and "secret lyrics" in t for k, t in kinds), kinds)

    def test_block_with_arrow_in_content_but_no_timestamp_is_skipped(self):
        # F5: a non-timestamp line containing "-->" was accepted as timing, dropping
        # the real first content line.
        from memovox.stentor import transcript as T

        segs = T.parse_cues("WEBVTT\n\nThis block has text with --> arrow\nmore text\n")
        self.assertEqual(segs, [], f"malformed block was mis-parsed: {segs}")

    def test_arrow_in_real_cue_body_is_preserved(self):
        from memovox.stentor import transcript as T

        segs = T.parse_cues(
            "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nif x --> y then done\n")
        self.assertEqual(len(segs), 1)
        self.assertIn("-->", segs[0].text)

    def test_srt_timestamps_still_parse(self):
        from memovox.stentor import transcript as T

        segs = T.parse_cues("1\n00:00:01,000 --> 00:00:02,500\nhello world\n")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].text, "hello world")
        self.assertAlmostEqual(segs[0].start, 1.0)
        self.assertAlmostEqual(segs[0].end, 2.5)


# --------------------------------------------------------------------------- #
# State integrity / idempotency / redaction
# --------------------------------------------------------------------------- #


def _store_with(tmp):
    from memovox.config import Config
    from memovox.loom import LoomStore
    return LoomStore(Config(store=pathlib.Path(tmp) / "s").ensure())


def _ingest_vtt(mv, body, **kw):
    p = pathlib.Path(tempfile.mktemp(suffix=".vtt"))
    p.write_text(body, encoding="utf-8")
    return mv.ingest(str(p), **kw)


class TestRedactionStateIntegrity(unittest.TestCase):
    def _named_video(self, name="Alice"):
        from memovox.sdk import Memovox
        tmp = tempfile.mkdtemp()
        mv = Memovox(store=str(pathlib.Path(tmp) / "s"))
        return mv

    def test_delete_video_removes_named_speaker_rows(self):
        # state-F1: per-video speaker rows (incl. resolved NAME = PII) survived redaction.
        mv = self._named_video()
        rep = _ingest_vtt(mv, "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n"
                              "<v Alice>Ribosomes synthesize proteins from amino acids.\n",
                          title="Bio")
        from memovox.loom import LoomStore
        s = LoomStore(mv.config)
        before = s.conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
        self.assertGreater(before, 0)
        self.assertTrue(s.delete_video(rep.video_id))
        rows = s.conn.execute("SELECT speaker_id, resolved_name FROM speakers").fetchall()
        # no speaker row may reference the deleted video, and no named PII may remain
        self.assertEqual([dict(r) for r in rows], [], f"orphaned speaker rows: {[dict(r) for r in rows]}")
        s.close()
        mv.close()

    def test_delete_all_videos_leaves_no_canonical_speakers(self):
        mv = self._named_video()
        r1 = _ingest_vtt(mv, "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n<v Alice>Topic one here.\n", title="A")
        r2 = _ingest_vtt(mv, "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n<v Alice>Topic two here.\n", title="B")
        from memovox.loom import LoomStore
        s = LoomStore(mv.config)
        # deleting only A keeps the canonical row (B still references spk:alice)
        s.delete_video(r1.video_id)
        canon = s.conn.execute("SELECT COUNT(*) FROM speakers WHERE speaker_id LIKE 'spk:%'").fetchone()[0]
        self.assertEqual(canon, 1, "canonical speaker GC'd while another video still references it")
        # deleting B too removes the now-orphaned canonical identity
        s.delete_video(r2.video_id)
        self.assertEqual(s.conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0], 0)
        s.close()
        mv.close()

    def test_delete_video_under_legacy_variable_limit(self):
        # state-F2: a >499-node IN-list exceeded the legacy 999 bound-variable limit and
        # half-committed. Chunked two-pass delete must succeed and clean cross-video edges.
        import sqlite3
        from memovox.loom import Moment, Video, Claim
        from memovox.backends.embed import HashingEmbedder
        tmp = tempfile.mkdtemp()
        s = _store_with(tmp)
        emb = HashingEmbedder(dim=16)
        s.upsert_video(Video(video_id="vid:big", source_url="https://x/b", title="B", content_hash="b"))
        s.upsert_video(Video(video_id="vid:small", source_url="https://x/s", title="S", content_hash="s"))
        for i in range(600):
            mid = f"vid:big#m{i:04d}"
            s.add_moment(Moment(mid, "vid:big", float(i), float(i + 1), f"t{i}", "spk", index=i),
                         emb.embed_one(f"t{i}"))
            s.add_claim(Claim(claim_id=f"{mid}.c00", moment_id=mid, video_id="vid:big",
                              text=f"claim {i}", status="committed", t_start_s=float(i), t_end_s=float(i + 1)))
        s.add_moment(Moment("vid:small#m0000", "vid:small", 0.0, 1.0, "s", "spk", index=0), emb.embed_one("s"))
        s.add_edge("vid:small#m0000", "CONTRADICTS", "vid:big#m0000.c00",
                   src_type="Moment", dst_type="Claim", video_id="vid:small")
        try:
            s.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        except Exception:  # pragma: no cover - older python
            self.skipTest("setlimit unsupported")
        self.assertTrue(s.delete_video("vid:big"))
        s.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 250000)
        self.assertEqual(s.conn.execute("SELECT COUNT(*) FROM videos WHERE video_id='vid:big'").fetchone()[0], 0)
        self.assertEqual(
            s.conn.execute("SELECT COUNT(*) FROM edges WHERE dst='vid:big#m0000.c00'").fetchone()[0], 0,
            "orphaned cross-video edge into a deleted claim")
        s.close()

    def test_delete_video_is_atomic_on_midway_failure(self):
        # The whole delete must roll back if a later step raises (no half-redaction).
        from memovox.loom import Moment, Video
        from memovox.backends.embed import HashingEmbedder
        tmp = tempfile.mkdtemp()
        s = _store_with(tmp)
        emb = HashingEmbedder(dim=16)
        s.upsert_video(Video(video_id="vid:x", source_url="https://x", title="X", content_hash="x"))
        s.add_moment(Moment("vid:x#m0000", "vid:x", 0.0, 1.0, "hello", "spk", index=0), emb.embed_one("hello"))
        boom = RuntimeError("injected mid-delete failure")

        def _raise(_delta):
            raise boom

        s._decrement_ledger = _raise  # last write inside the transaction
        with self.assertRaises(RuntimeError):
            s.delete_video("vid:x")
        # rolled back: the video and its moment are still present
        self.assertIsNotNone(s.get_video("vid:x"))
        self.assertEqual(s.conn.execute("SELECT COUNT(*) FROM moments WHERE video_id='vid:x'").fetchone()[0], 1)
        s.close()


# --------------------------------------------------------------------------- #
# Determinism / ordering
# --------------------------------------------------------------------------- #


class TestDeterministicOrdering(unittest.TestCase):
    def test_graph_expand_breaks_ties_by_moment_id(self):
        # determinism-F1: same-hop neighbors share score 1/(hop+1); their order came from
        # set-iteration (hash-seed-dependent). Must now sort by (-score, moment_id).
        from memovox.augur.traverse import expand
        from memovox.loom import Moment, Video, Claim
        from memovox.backends.embed import HashingEmbedder
        tmp = tempfile.mkdtemp()
        s = _store_with(tmp)
        emb = HashingEmbedder(dim=16)
        s.upsert_video(Video(video_id="vid:seed", source_url="https://x/seed", title="seed", content_hash="seed"))
        s.add_moment(Moment("vid:seed#m0000", "vid:seed", 0.0, 1.0, "seed claim text", "spk", index=0), emb.embed_one("seed"))
        s.add_claim(Claim(claim_id="vid:seed#m0000.c00", moment_id="vid:seed#m0000", video_id="vid:seed",
                          text="seed claim", status="committed", t_start_s=0.0, t_end_s=1.0))
        # several cross-video neighbors, each one CONTRADICTS hop from the seed
        for tag in ("delta", "alpha", "foxtrot", "bravo", "echo", "charlie", "golf", "hotel"):
            v = f"vid:{tag}"
            s.upsert_video(Video(video_id=v, source_url=f"https://x/{tag}", title=tag, content_hash=tag))
            s.add_moment(Moment(f"{v}#m0000", v, 0.0, 1.0, f"{tag} text", "spk", index=0), emb.embed_one(tag))
            s.add_claim(Claim(claim_id=f"{v}#m0000.c00", moment_id=f"{v}#m0000", video_id=v,
                              text=f"{tag} claim", status="committed", t_start_s=0.0, t_end_s=1.0))
            s.add_edge("vid:seed#m0000.c00", "CONTRADICTS", f"{v}#m0000.c00",
                       src_type="Claim", dst_type="Claim", video_id="vid:seed")
        result = expand(s, ["vid:seed#m0000"], rels=["CONTRADICTS", "SUPPORTS"], hops=1)
        ids = [mid for mid, _ in result]
        scores = [sc for _, sc in result]
        self.assertEqual(len(set(scores)), 1, "expected all neighbors to tie on score")
        self.assertEqual(ids, sorted(ids), f"graph expand not deterministically ordered: {ids}")
        s.close()

    def test_rrf_fuse_breaks_ties_by_item_id(self):
        # determinism-F2: equal fused scores must rank by id, independent of leg order.
        from memovox.augur.retrieve import rrf_fuse
        leg_a = [("m_z", 0.9), ("m_x", 0.5)]
        leg_b = [("m_z", 0.9), ("m_y", 0.5)]  # m_x and m_y tie at the same RRF rank
        order1 = [m for m, _ in rrf_fuse([leg_a, leg_b], top_k=10)]
        order2 = [m for m, _ in rrf_fuse([leg_b, leg_a], top_k=10)]
        self.assertEqual(order1, order2, "rrf_fuse order depends on leg order under ties")
        # the tied pair is ordered by id
        self.assertLess(order1.index("m_x"), order1.index("m_y"))


# --------------------------------------------------------------------------- #
# Provenance / citation integrity
# --------------------------------------------------------------------------- #

import re as _re


def _bio_store(tmp):
    from memovox.loom import LoomStore, Moment, Video
    from memovox.config import Config
    from memovox.backends.embed import HashingEmbedder
    store = LoomStore(Config(store=pathlib.Path(tmp) / "s").ensure())
    emb = HashingEmbedder(dim=64)
    store.upsert_video(Video(video_id="vid:a", source_url="https://youtu.be/AAAAAAAAAAA",
                             title="Bio", content_hash="a"))
    m0 = Moment("vid:a#m0000", "vid:a", 10.0, 20.0,
                "Ribosomes synthesize proteins from amino acids.", "spk_0", index=0)
    store.add_moment(m0, emb.embed_one(m0.transcript))
    return store, emb


def _make_llm(output):
    from memovox.backends.base import LLMBackend

    class FakeLLM(LLMBackend):
        is_generative = True

        def complete(self, prompt, *, system=None, max_tokens=512, temperature=0.0):
            return output

    return FakeLLM()


def _all_sentences_cited(text):
    from memovox.util import split_sentences
    return bool(text.strip()) and all(_re.search(r"\[\d+\]", s) for s in split_sentences(text))


class TestProvenanceIntegrity(unittest.TestCase):
    def test_uncited_llm_answer_falls_back_to_grounded(self):
        # provenance-F1 (CRIT): an uncited fabricated answer must not be surfaced.
        from memovox import augur
        from memovox.config import Settings
        tmp = tempfile.mkdtemp()
        store, emb = _bio_store(tmp)
        llm = _make_llm("Ribosomes are made of titanium. They were invented by Napoleon.")
        ans = augur.ask(store, "what are ribosomes?", embedder=emb, llm=llm, settings=Settings())
        self.assertNotIn("Napoleon", ans.text, "uncited fabrication surfaced")
        # whatever is surfaced is citation-grounded (or the low-evidence refusal)
        self.assertTrue(_all_sentences_cited(ans.text) or ans.low_evidence, ans.text)
        store.close()

    def test_dangling_marker_llm_answer_rejected(self):
        from memovox import augur
        from memovox.config import Settings
        tmp = tempfile.mkdtemp()
        store, emb = _bio_store(tmp)
        llm = _make_llm("Ribosomes are made entirely of lipids and were discovered in 1823. [9]")
        ans = augur.ask(store, "what are ribosomes?", embedder=emb, llm=llm, settings=Settings())
        markers = {int(m) for m in _re.findall(r"\[(\d+)\]", ans.text)}
        valid = {c.index for c in ans.citations}
        self.assertFalse(markers - valid, f"dangling markers surfaced: {markers - valid}")
        self.assertNotIn("lipids", ans.text)
        store.close()

    def test_valid_grounded_llm_answer_passes_through(self):
        # a compliant, fully-cited answer is preserved unchanged.
        from memovox import augur
        from memovox.config import Settings
        tmp = tempfile.mkdtemp()
        store, emb = _bio_store(tmp)
        llm = _make_llm("Ribosomes synthesize proteins from amino acids. [1]")
        ans = augur.ask(store, "what do ribosomes do?", embedder=emb, llm=llm, settings=Settings())
        self.assertEqual(ans.text, "Ribosomes synthesize proteins from amino acids. [1]")
        self.assertFalse(ans.low_evidence)
        store.close()

    def test_uncited_llm_synthesis_falls_back(self):
        # provenance-F1 applies to synthesize_topic too.
        from memovox.augur.synthesize import synthesize
        from memovox.config import Settings
        from memovox.loom import Moment, Video
        tmp = tempfile.mkdtemp()
        store, emb = _bio_store(tmp)
        # a second video so the cross-corpus floor is met
        store.upsert_video(Video(video_id="vid:b", source_url="https://youtu.be/BBBBBBBBBBB",
                                 title="Bio2", content_hash="b"))
        m = Moment("vid:b#m0000", "vid:b", 5.0, 9.0,
                   "Ribosomes synthesize proteins from amino acids.", "spk_0", index=0)
        store.add_moment(m, emb.embed_one(m.transcript))
        from memovox.backends.base import NLIBackend

        class _NLI(NLIBackend):
            name = "lexical"

            def classify(self, premise, hypothesis):
                from memovox.backends.base import NLIResult
                return NLIResult(label="entailment", entail=0.99, contradict=0.0, neutral=0.01)
        # commit claims directly
        from memovox.loom import Claim
        for v, mid in (("vid:a", "vid:a#m0000"), ("vid:b", "vid:b#m0000")):
            store.add_claim(Claim(claim_id=f"{mid}.c00", moment_id=mid, video_id=v,
                                  text="Ribosomes synthesize proteins from amino acids.",
                                  subject="Ribosomes", status="committed", t_start_s=5.0, t_end_s=9.0,
                                  salience=0.9, entailment_score=0.9))
        llm = _make_llm("Ribosomes are made of titanium and were invented by Napoleon.")
        syn = synthesize(store, "ribosomes", nli=_NLI(), llm=llm, embedder=emb, settings=Settings())
        self.assertNotIn("Napoleon", syn.text, "uncited synthesis surfaced")
        self.assertTrue(_all_sentences_cited(syn.text) or syn.low_evidence, syn.text)
        store.close()

    def test_provenance_withheld_for_unsupported_claim(self):
        # provenance-F2: an unsupported claim must not hand back a trusted deep link.
        from memovox.loom import Claim
        from memovox.sdk import Memovox
        tmp = tempfile.mkdtemp()
        store, emb = _bio_store(tmp)
        store.add_claim(Claim(claim_id="vid:a#m0000.c00", moment_id="vid:a#m0000", video_id="vid:a",
                              text="Ribosomes are made of pure gold and cure cancer.",
                              subject="Ribosomes", t_start_s=10.0, t_end_s=20.0,
                              speaker_id="spk_0", status="unsupported", entailment_score=0.02))
        store.add_claim(Claim(claim_id="vid:a#m0000.c01", moment_id="vid:a#m0000", video_id="vid:a",
                              text="Ribosomes synthesize proteins.", subject="Ribosomes",
                              t_start_s=10.0, t_end_s=20.0, speaker_id="spk_0",
                              status="committed", entailment_score=0.91))
        store.close()
        mv = Memovox(store=str(pathlib.Path(tmp) / "s"))
        bad = mv.get_provenance("vid:a#m0000.c00")
        self.assertIsNone(bad["provenance"], "served provenance for an unsupported claim")
        self.assertFalse(bad["trusted"])
        good = mv.get_provenance("vid:a#m0000.c01")
        self.assertIsNotNone(good["provenance"])
        self.assertTrue(good["trusted"])
        self.assertIn("deep_link", good["provenance"])
        mv.close()


# --------------------------------------------------------------------------- #
# Serving / config robustness
# --------------------------------------------------------------------------- #


class TestServingRobustness(unittest.TestCase):
    def test_wrong_type_body_fields_return_clean_400(self):
        # serving-F1: non-string fields reached .strip()/a SQLite param and surfaced a
        # raw Python/SQLite error string in a 500. Now rejected at the boundary (mv unused).
        from memovox.server import routes

        cases = [
            (routes.route_query, {"query": 12345}),
            (routes.route_query, {"query": "ok", "video_id": ["x"]}),
            (routes.route_query, {"question": {"a": 1}}),
            (routes.route_synthesize, {"topic": 999}),
            (routes.route_ingest, {"source": ["not", "a", "string"]}),
            (routes.route_ingest, {"source": "f.vtt", "title": 7}),
        ]
        for fn, body in cases:
            status, payload, _ = fn(None, body)  # validation short-circuits before mv use
            self.assertEqual(int(status), 400, f"{fn.__name__} {body} -> {status}")
            self.assertIn("must be a string", payload["error"], payload)
            # no internal Python/SQLite detail leaked
            self.assertNotIn("object has no attribute", payload["error"])
            self.assertNotIn("binding parameter", payload["error"])

    def test_missing_required_fields_still_400(self):
        from memovox.server import routes
        for fn, body, frag in [
            (routes.route_ingest, {}, "source"),
            (routes.route_query, {}, "query"),
            (routes.route_synthesize, {}, "topic"),
        ]:
            status, payload, _ = fn(None, body)
            self.assertEqual(int(status), 400)
            self.assertIn(frag, payload["error"])

    def test_oversized_request_body_is_not_allocated(self):
        # serving-F2: _body() read the full advertised Content-Length into memory.
        import io
        from memovox.sdk import Memovox
        from memovox.server.rest import make_handler

        tmp = tempfile.mkdtemp()
        mv = Memovox(store=str(pathlib.Path(tmp) / "s"))
        Handler = make_handler(mv)
        h = Handler.__new__(Handler)
        h.headers = {"Content-Length": str(h.MAX_BODY_BYTES + 1)}
        # if _body honored the length it would try to read this; we give it only a few bytes
        h.rfile = io.BytesIO(b'{"x":1}')
        self.assertEqual(h._body(), {})  # rejected without reading the advertised length
        self.assertEqual(h.rfile.tell(), 0, "oversized body was read into memory")
        # a normal small body still parses
        h2 = Handler.__new__(Handler)
        payload = b'{"query":"hello"}'
        h2.headers = {"Content-Length": str(len(payload))}
        h2.rfile = io.BytesIO(payload)
        self.assertEqual(h2._body(), {"query": "hello"})
        mv.close()


# --------------------------------------------------------------------------- #
# Round 2 — residual defects found by the re-review of the round-1 fixes
# --------------------------------------------------------------------------- #


class TestRound2Hardening(unittest.TestCase):
    def test_grounding_gate_rejects_uncited_continuations(self):
        # provenance round-2 (HIGH): split_sentences let lowercase / ; / , / no-punct
        # continuations smuggle uncited prose past the gate.
        from memovox.augur.answer import _llm_citations_valid as V
        from memovox.augur.types import Citation
        c = [Citation(index=1, video_id="v", moment_id="v#m0", t_start_s=0, t_end_s=1,
                      source_text="Ribosomes synthesize proteins."),
             Citation(index=2, video_id="v", moment_id="v#m1", t_start_s=0, t_end_s=1,
                      source_text="x")]
        reject = [
            "Real fact [1]. they were invented by Napoleon.",   # lowercase continuation
            "Real fact [1], uncited run-on about Napoleon",     # comma run-on
            "Real fact [1]; uncited clause about Napoleon",      # semicolon clause
            "Uncited middle clause. Real fact [1].",            # uncited leading clause
            "X. [1] uncited middle. Y. [2]",                    # uncited middle (markers after period)
            "X. Y.",                                             # fully uncited
            "Real fact [9].",                                    # dangling marker
        ]
        accept = [
            "Ribosomes synthesize proteins from amino acids. [1]",  # marker after period
            "Claim one. [1] Claim two. [2]",                        # multi, markers after periods
            "Claim one [1]. Claim two [2].",                        # multi, markers before periods
            "Ribosomes synthesize proteins [1]",                    # single, no terminal punct
            "Fact one [1]; fact two [2].",                          # both clauses cited
        ]
        for t in reject:
            self.assertFalse(V(t, c), f"should reject: {t!r}")
        for t in accept:
            self.assertTrue(V(t, c), f"should accept: {t!r}")

    def test_uncited_continuation_not_surfaced_via_ask(self):
        from memovox import augur
        from memovox.config import Settings
        tmp = tempfile.mkdtemp()
        store, emb = _bio_store(tmp)
        llm = _make_llm("Ribosomes synthesize proteins [1]. they were secretly invented by Napoleon.")
        ans = augur.ask(store, "what are ribosomes?", embedder=emb, llm=llm, settings=Settings())
        self.assertNotIn("Napoleon", ans.text)
        self.assertTrue(_all_sentences_cited(ans.text) or ans.low_evidence, ans.text)
        store.close()

    def test_synthesize_cite_cites_every_sentence(self):
        # provenance round-2 (MED): a multi-sentence claim must get a marker per sentence.
        from memovox.augur.synthesize import _cite
        out = _cite("Ribosomes synthesize proteins. They also assemble amino acids", 3)
        self.assertTrue(_all_sentences_cited(out), out)

    def test_nonfinite_json_timings_coerced_finite(self):
        # ingest round-2 (MED): NaN/inf/1e400 timings crashed the deep-link / hms layer.
        from memovox.stentor import transcript as T
        from memovox import util
        for payload in ([{"start": "inf", "end": "2", "text": "a real claim"}],
                        [{"start": "NaN", "end": 1, "text": "claim"}],
                        [{"start": 1e400, "end": 2, "text": "claim"}]):
            segs = T.parse_json(payload)
            self.assertEqual(len(segs), 1)
            import math
            self.assertTrue(math.isfinite(segs[0].start), f"non-finite start survived: {payload}")
            self.assertTrue(math.isfinite(segs[0].end))
            # downstream formatters no longer crash
            util.deep_link("https://youtu.be/ABC", segs[0].start)
            util.seconds_to_hms(segs[0].start)

    def test_null_json_text_dropped_not_stringified(self):
        # ingest round-2 (LOW): text:null became the literal word "None".
        from memovox.stentor import transcript as T
        self.assertEqual(T.parse_json([{"start": 0, "end": 1, "text": None}]), [])
        # a list/dict text is also dropped, not stringified
        self.assertEqual(T.parse_json([{"start": 0, "end": 1, "text": ["a"]}]), [])

    def test_export_bad_format_returns_400_not_500(self):
        # serving round-2 (MED): /export?format=<bad> raised ValueError -> 500 leak.
        from memovox.sdk import Memovox
        from memovox.server import routes
        tmp = tempfile.mkdtemp()
        mv = Memovox(store=str(pathlib.Path(tmp) / "s"))
        rep = _ingest_vtt(mv, "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nhello world here.\n",
                          source_url="https://youtu.be/abc123")
        for bad in ("xml", "md/../../etc", "txt"):
            status, payload, _ = routes.route_export(mv, rep.video_id, {"format": bad})
            self.assertEqual(int(status), 400, f"{bad} -> {status}")
        # a real format still works
        status, _, _ = routes.route_export(mv, rep.video_id, {"format": "json"})
        self.assertEqual(int(status), 200)
        mv.close()


if __name__ == "__main__":
    unittest.main()
