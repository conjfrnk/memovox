"""Tests for the entity-linker backend (W2.2).

The default :class:`NullLinker` path is deterministic and dependency-free; the
optional :class:`WikidataLinker` only *adds* a ``wikidata_qid`` (and may set the
canonical label) without changing the slug-derived ``entity_id``. These tests
must run with ZERO network access under the hermetic env (``HF_HUB_OFFLINE=1``),
so the live-Wikidata test is skipped unless actually online.
"""

import json
import unittest
from unittest import mock

from memovox.backends import get_entity_linker
from memovox.backends.entity_link import Canonical, NullLinker, WikidataLinker


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


if __name__ == "__main__":
    unittest.main()
