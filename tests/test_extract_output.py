"""M-X W3 (X.4) — schema-validated, deterministic, answer-distinct extraction output."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.assay.extract_output import extract_document, validate_document
from memovox.cli import main
from memovox.loom.models import CLAIM_TYPES, Moment

_CLAIM_KEYS = {"claim_id", "text", "subject", "predicate", "object", "type",
               "t_start_s", "t_end_s", "salience"}


def _moment():
    return Moment("v#m0000", "v", 0.0, 12.0,
                  "The Transformer architecture is the foundation of modern language models.",
                  "spk_0", index=0)


class ExtractDocumentTest(unittest.TestCase):
    def test_schema_shape_and_claim_types(self):
        doc = extract_document(_moment())
        self.assertEqual(set(doc), {"version", "claims", "entities"})
        self.assertTrue(doc["claims"])
        for c in doc["claims"]:
            self.assertEqual(set(c), _CLAIM_KEYS)
            self.assertIn(c["type"], CLAIM_TYPES)
        validate_document(doc)  # no raise

    def test_deterministic(self):
        m = _moment()
        a, b = extract_document(m), extract_document(m)
        self.assertEqual(a, b)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_distinct_from_answer_shape(self):
        doc = extract_document(_moment())
        # the extracted KNOWLEDGE structure, NOT a generated answer
        self.assertNotIn("text", doc)
        self.assertNotIn("citations", doc)
        self.assertNotIn("strategy", doc)
        self.assertIn("claims", doc)
        self.assertIn("entities", doc)

    def test_free_path_uses_no_llm(self):
        doc = extract_document(_moment(), llm=None)  # rule-based
        self.assertTrue(doc["claims"])

    def test_validate_rejects_bad_type(self):
        with self.assertRaises(ValueError):
            validate_document({"version": "1.0", "entities": [], "claims": [
                {"claim_id": "x", "text": "t", "subject": "s", "predicate": "p",
                 "object": "o", "type": "NOT_A_TYPE", "t_start_s": 0.0, "t_end_s": 1.0,
                 "salience": 0.0}]})


class ExtractCliTest(unittest.TestCase):
    def test_extract_command_prints_schema_valid_json_to_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            vtt = d / "talk.en.vtt"
            vtt.write_text("WEBVTT\n\n00:00:02.000 --> 00:00:12.000\n"
                           "The Transformer architecture is the foundation of language models.\n",
                           encoding="utf-8")
            store = str(d / "store")
            main(["--store", store, "--llm", "none", "ingest", str(vtt),
                  "--source-url", "https://youtu.be/abc123"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--store", store, "extract", "yt:abc123"])
            self.assertEqual(code, 0)
            doc = json.loads(out.getvalue())   # valid JSON on stdout
            validate_document(doc)
            self.assertIn("claims", doc)


if __name__ == "__main__":
    unittest.main()
