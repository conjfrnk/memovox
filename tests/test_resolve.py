"""Tests for the entity-linker backend (W2.2).

The default :class:`NullLinker` path is deterministic and dependency-free; the
optional :class:`WikidataLinker` only *adds* a ``wikidata_qid`` (and may set the
canonical label) without changing the slug-derived ``entity_id``. These tests
must run with ZERO network access under the hermetic env (``HF_HUB_OFFLINE=1``),
so the live-Wikidata test is skipped unless actually online.
"""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends import get_entity_linker
from memovox.backends.entity_link import Canonical, NullLinker, WikidataLinker
from memovox.config import Config
from memovox.loom import Claim, LoomStore, Moment, Video
from memovox.loom.models import STATUS_UNSUPPORTED
from memovox.loom.resolve import resolve_entities


class NullLinkerTest(unittest.TestCase):
    def test_canonicalize_basic(self):
        result = get_entity_linker("none").canonicalize("Transformer")
        self.assertEqual(result, Canonical("ent:transformer", "Transformer", None))

    def test_slug_forms(self):
        linker = get_entity_linker("none")
        self.assertEqual(linker.canonicalize("GPT-4").entity_id, "ent:gpt-4")
        self.assertEqual(linker.canonicalize("New York").entity_id, "ent:new-york")
        self.assertEqual(
            linker.canonicalize("Albert Einstein").entity_id, "ent:albert-einstein"
        )

    def test_deterministic_and_idempotent(self):
        # Same instance, two calls.
        linker = get_entity_linker("none")
        first = linker.canonicalize("Transformer")
        second = linker.canonicalize("Transformer")
        self.assertEqual(first.entity_id, second.entity_id)
        # Two separate instances also agree.
        other = NullLinker()
        self.assertEqual(other.canonicalize("Transformer").entity_id, first.entity_id)
        # The surface name is preserved verbatim, only the id is slugged.
        self.assertEqual(first.name, "Transformer")

    def test_unknown_backend_raises(self):
        from memovox.errors import BackendUnavailable

        with self.assertRaises(BackendUnavailable):
            get_entity_linker("does-not-exist")


class AutoSelectionTest(unittest.TestCase):
    def test_auto_falls_back_to_null_offline(self):
        # Under the hermetic env (HF_HUB_OFFLINE=1) WikidataLinker.is_available()
        # returns False, so "auto" must resolve to the NullLinker.
        linker = get_entity_linker("auto")
        self.assertIsInstance(linker, NullLinker)
        self.assertEqual(linker.name, "none")


class WikidataFallbackTest(unittest.TestCase):
    def test_graceful_fallback_on_network_error(self):
        # Prove graceful degradation WITHOUT touching the network: force urlopen
        # to raise and assert the result is identical to the NullLinker output.
        linker = WikidataLinker()
        with mock.patch(
            "memovox.backends.entity_link.urllib.request.urlopen",
            side_effect=OSError("no network"),
        ):
            result = linker.canonicalize("Transformer")
        self.assertEqual(result, Canonical("ent:transformer", "Transformer", None))

    def test_offline_is_not_available(self):
        # The hermetic env sets HF_HUB_OFFLINE=1, so is_available() must short
        # circuit to False with no socket probe / hang.
        self.assertFalse(WikidataLinker.is_available())

    def test_success_keeps_slug_id(self):
        # Exercise the SUCCESS branch offline: a successful lookup must only add
        # the QID + canonical label; the entity_id stays slug-derived from the
        # ORIGINAL surface ("Einstein"), never the label or the QID.
        payload = json.dumps(
            {"search": [{"id": "Q937", "label": "Albert Einstein"}]}
        ).encode("utf-8")
        resp = mock.MagicMock()
        resp.read.return_value = payload
        resp.__enter__.return_value = resp  # match the `with urlopen(...) as resp`
        with mock.patch(
            "memovox.backends.entity_link.urllib.request.urlopen",
            return_value=resp,
        ):
            result = WikidataLinker().canonicalize("Einstein")
        self.assertEqual(result.entity_id, "ent:einstein")  # NOT ent:albert-einstein, NOT Q937
        self.assertEqual(result.wikidata_qid, "Q937")
        self.assertEqual(result.name, "Albert Einstein")


class WikidataLiveTest(unittest.TestCase):
    @unittest.skipUnless(WikidataLinker.is_available(), "wikidata offline")
    def test_canonicalize_online(self):  # pragma: no cover - network dependent
        result = WikidataLinker().canonicalize("Albert Einstein")
        # The id stays slug-derived regardless of connectivity; only the QID is
        # added when online.
        self.assertEqual(result.entity_id, "ent:albert-einstein")
        self.assertIsNotNone(result.wikidata_qid)
        self.assertTrue(result.wikidata_qid.startswith("Q"))


# --------------------------------------------------------------------------- #
# W2.3 — cross-corpus entity resolution
# --------------------------------------------------------------------------- #


class _ResolveTestBase(unittest.TestCase):
    """A two-video store built by hand: both videos mention 'Chinchilla'.

    Deterministic and network-free (NullLinker via ``get_entity_linker("none")``),
    so the slug ids match exactly what the pipeline writes offline.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.linker = get_entity_linker("none")

        for vid in ("yt:a", "yt:b"):
            self.store.upsert_video(Video(video_id=vid, source_url=None, title=vid))

        # talk_a: Transformer (a-only) + Chinchilla (shared).
        self.store.add_moment(Moment("yt:a#m0000", "yt:a", 0.0, 30.0,
                                      "The Transformer architecture matters.", index=0))
        self.store.add_moment(Moment("yt:a#m0001", "yt:a", 30.0, 60.0,
                                      "The Chinchilla study showed scaling.", index=1))
        # talk_b: Chinchilla (shared) + Llama (b-only).
        self.store.add_moment(Moment("yt:b#m0000", "yt:b", 0.0, 30.0,
                                      "The Chinchilla recipe is useful.", index=0))
        self.store.add_moment(Moment("yt:b#m0001", "yt:b", 30.0, 60.0,
                                      "The Llama family reused the ratio.", index=1))

        self.claims = [
            Claim("yt:a#m0000.c00", "yt:a#m0000", "yt:a",
                  "The Transformer architecture matters.", subject="Transformer",
                  t_start_s=0.0, t_end_s=30.0),
            Claim("yt:a#m0001.c00", "yt:a#m0001", "yt:a",
                  "The Chinchilla study showed scaling.", subject="Chinchilla",
                  t_start_s=30.0, t_end_s=60.0),
            Claim("yt:b#m0000.c00", "yt:b#m0000", "yt:b",
                  "The Chinchilla recipe is useful.", subject="Chinchilla",
                  t_start_s=0.0, t_end_s=30.0),
            Claim("yt:b#m0001.c00", "yt:b#m0001", "yt:b",
                  "The Llama family reused the ratio.", subject="Llama",
                  t_start_s=30.0, t_end_s=60.0),
        ]
        for c in self.claims:
            self.store.add_claim(c)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class ResolveEntitiesTest(_ResolveTestBase):
    def test_same_entity_across_videos_is_one_node(self):
        resolve_entities(self.store, self.claims, linker=self.linker)

        ents = self.store.list_entities()
        shared = [e for e in ents if e.entity_id == "ent:chinchilla"]
        self.assertEqual(len(shared), 1)

        mids = self.store.entity_mentions("ent:chinchilla")  # -> [claim_id]
        claims = self.store.get_claims(mids)
        self.assertEqual(len({c.video_id for c in claims}), 2)

    def test_single_video_entities_stay_local(self):
        resolve_entities(self.store, self.claims, linker=self.linker)
        # Transformer (talk_a only) and Llama (talk_b only) each span one video.
        for ent_id in ("ent:transformer", "ent:llama"):
            claims = self.store.get_claims(self.store.entity_mentions(ent_id))
            self.assertEqual(len({c.video_id for c in claims}), 1)

    def test_mentions_edge_carries_provenance(self):
        resolve_entities(self.store, self.claims, linker=self.linker)
        out = self.store.neighbors("yt:a#m0001.c00", rel="MENTIONS")
        dsts = {e["dst"] for e in out}
        self.assertIn("ent:chinchilla", dsts)
        edge = next(e for e in out if e["dst"] == "ent:chinchilla")
        self.assertEqual(edge["src_type"], "Claim")
        self.assertEqual(edge["dst_type"], "Entity")
        self.assertEqual(edge["video_id"], "yt:a")
        self.assertEqual(edge["t_start_s"], 30.0)
        self.assertEqual(edge["t_end_s"], 60.0)

    def test_idempotent_resolution(self):
        resolve_entities(self.store, self.claims, linker=self.linker)
        n_ents = len(self.store.list_entities())
        n_edges = len(self.store.edges(rel="MENTIONS"))
        n_mentions = len(self.store.entity_mentions("ent:chinchilla"))
        # Re-resolving the same claims must not duplicate anything.
        resolve_entities(self.store, self.claims, linker=self.linker)
        self.assertEqual(len(self.store.list_entities()), n_ents)
        self.assertEqual(len(self.store.edges(rel="MENTIONS")), n_edges)
        self.assertEqual(len(self.store.entity_mentions("ent:chinchilla")), n_mentions)

    def test_only_committed_claims_resolved(self):
        # An unsupported claim's mentions must not enter the graph.
        rejected = Claim("yt:a#m0000.c99", "yt:a#m0000", "yt:a",
                         "The Mamba model is overhyped.", subject="Mamba",
                         status=STATUS_UNSUPPORTED, t_start_s=0.0, t_end_s=30.0)
        self.store.add_claim(rejected)
        resolve_entities(self.store, self.claims + [rejected], linker=self.linker)
        self.assertIsNone(self.store.get_entity("ent:mamba"))


class ResolveGoldenCorpusTest(unittest.TestCase):
    """End-to-end through the real pipeline on the golden corpus.

    Mirrors the eval harness's free-stack ingest; the shared golden entity is
    'Chinchilla' (in both talks), so it must resolve to a single node whose
    mentions span both videos.
    """

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from memovox import Memovox

        cls._tmp = tempfile.TemporaryDirectory(prefix="memovox-resolve-")
        golden = pathlib.Path(__file__).resolve().parent.parent / "eval" / "golden"
        cls.mv = Memovox(
            store=cls._tmp.name,
            embed_backend="hashing", nli_backend="lexical", asr_backend="captions",
            llm_backend="none", vlm_backend="none", ocr_backend="none",
            entity_backend="none",
        )
        cls.video_ids = []
        for vtt in sorted(golden.glob("*.en.vtt")):
            report = cls.mv.ingest(str(vtt))
            cls.video_ids.append(report.video_id)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_chinchilla_unified_across_both_videos(self):
        with LoomStore(self.mv.config) as store:
            ent = store.get_entity("ent:chinchilla")
            self.assertIsNotNone(ent)
            claims = store.get_claims(store.entity_mentions("ent:chinchilla"))
            self.assertGreaterEqual(len({c.video_id for c in claims}), 2)

    def test_resolution_is_idempotent_across_reingest(self):
        with LoomStore(self.mv.config) as store:
            ents_before = len(store.list_entities())
            edges_before = len(store.edges(rel="MENTIONS"))
        # Re-ingest the same corpus (unchanged) — must be a no-op for the graph.
        golden = pathlib.Path(__file__).resolve().parent.parent / "eval" / "golden"
        for vtt in sorted(golden.glob("*.en.vtt")):
            self.mv.ingest(str(vtt))
        with LoomStore(self.mv.config) as store:
            self.assertEqual(len(store.list_entities()), ents_before)
            self.assertEqual(len(store.edges(rel="MENTIONS")), edges_before)


if __name__ == "__main__":
    unittest.main()
