"""Tests for the observability spine (M0.1): tracer, stderr logging, budget."""

from __future__ import annotations

import io
import json
import logging
import sys
import unittest
from contextlib import redirect_stdout

from memovox.observe import Span, Tracer, get_logger


class TracerSpanTest(unittest.TestCase):
    def test_span_context_manager_records_wall_and_ok(self):
        tracer = Tracer()
        with tracer.span("asr") as span:
            self.assertIsInstance(span, Span)
            self.assertEqual(span.stage, "asr")
        self.assertGreaterEqual(span.wall_ms, 0.0)
        self.assertEqual(span.status, "ok")
        # the span is collected on the tracer's ordered trace
        self.assertIn(span, tracer.spans)

    def test_span_records_error_status_and_reraises(self):
        tracer = Tracer()
        with self.assertRaises(ValueError):
            with tracer.span("claims") as span:
                raise ValueError("boom")
        self.assertEqual(span.status, "error")
        self.assertGreaterEqual(span.wall_ms, 0.0)

    def test_counters_and_caps_accumulate_into_to_dict(self):
        tracer = Tracer()
        with tracer.span("claims") as span:
            span.add_counter("claims", 5)
            span.add_counter("claims", 2)  # accumulates
            span.add_cap("max_claims", limit=600, dropped=3)
        d = span.to_dict()
        self.assertEqual(set(d), {"stage", "wall_ms", "status", "counters", "caps"})
        self.assertEqual(d["counters"]["claims"], 7)
        self.assertEqual(d["caps"], [{"name": "max_claims", "limit": 600, "dropped": 3}])


class StructuredLoggingTest(unittest.TestCase):
    def test_logger_single_handler_targets_stderr(self):
        logger = get_logger("memovox.pipeline")
        handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        self.assertEqual(len(handlers), 1)
        self.assertIs(handlers[0].stream, sys.stderr)

    def test_get_logger_is_idempotent_no_handler_stacking(self):
        a = get_logger("memovox.idem")
        before = len(a.handlers)
        b = get_logger("memovox.idem")
        self.assertIs(a, b)
        self.assertEqual(len(b.handlers), before)

    def test_emits_one_sorted_json_object_per_record(self):
        logger = get_logger("memovox.jsonfmt")
        handler = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler))
        buf = io.StringIO()
        saved = handler.stream
        handler.stream = buf
        try:
            logger.info("span", extra={"span": {"stage": "asr", "status": "ok"}})
        finally:
            handler.stream = saved
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["span"]["stage"], "asr")
        # keys are emitted in sorted order
        self.assertEqual(lines[0], json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def test_logging_never_writes_to_stdout(self):
        out = io.StringIO()
        with redirect_stdout(out):
            tracer = Tracer(logger=get_logger("memovox.stdoutcheck"))
            with tracer.span("asr") as span:
                span.add_counter("x", 1)
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
