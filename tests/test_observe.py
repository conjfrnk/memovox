"""Tests for the observability spine (M0.1): tracer, stderr logging, budget."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

from memovox.config import Settings
from memovox.errors import MemovoxError
from memovox.observe import (
    Budget,
    BudgetExceeded,
    Span,
    Tracer,
    _NullOtelSpan,
    get_logger,
    otel_available,
)


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

    def test_logging_failure_never_breaks_or_masks(self):
        # Observability logging must never break the traced op, and in the error
        # path must never mask the ORIGINAL exception with a logging exception.
        class _BoomLogger:
            def info(self, *a, **k):
                raise RuntimeError("log boom")

            def warning(self, *a, **k):
                raise RuntimeError("log boom")

        tracer = Tracer(logger=_BoomLogger())
        with tracer.span("ok") as sp:  # clean exit must not raise despite log boom
            sp.add_counter("x", 1)
        self.assertEqual(sp.status, "ok")

        with self.assertRaises(ValueError):  # the real error, not RuntimeError
            with tracer.span("err"):
                raise ValueError("real error")

    def test_logging_never_writes_to_stdout(self):
        out = io.StringIO()
        with redirect_stdout(out):
            tracer = Tracer(logger=get_logger("memovox.stdoutcheck"))
            with tracer.span("asr") as span:
                span.add_counter("x", 1)
        self.assertEqual(out.getvalue(), "")


class BudgetTest(unittest.TestCase):
    def test_soft_budget_records_overage_without_raising(self):
        budget = Budget(max_units=10, mode="soft")
        overage = budget.charge(15)
        self.assertTrue(budget.exceeded)
        self.assertEqual(budget.overage, 5)
        self.assertEqual(overage, 5)

    def test_hard_budget_raises_budget_exceeded(self):
        budget = Budget(max_units=10, mode="hard")
        with self.assertRaises(BudgetExceeded):
            budget.charge(15)

    def test_budget_exceeded_is_a_memovox_error(self):
        # so the CLI's `except (MemovoxError, ...)` catches hard-mode failures
        self.assertTrue(issubclass(BudgetExceeded, MemovoxError))

    def test_under_budget_is_not_exceeded(self):
        budget = Budget(max_units=10, mode="hard")
        self.assertEqual(budget.charge(4), 0)
        self.assertEqual(budget.charge(6), 0)  # exactly at the cap is not over
        self.assertFalse(budget.exceeded)

    def test_none_max_units_means_unbounded(self):
        budget = Budget(max_units=None, mode="hard")
        budget.charge(10_000)  # never raises, never exceeded
        self.assertFalse(budget.exceeded)
        self.assertEqual(budget.overage, 0)


class OtelTest(unittest.TestCase):
    def test_null_otel_span_methods_are_callable_noops(self):
        with _NullOtelSpan() as entered:
            entered.set_attribute("k", "v")
            entered.end()  # harmless no-ops, no exception

    def test_tracer_stdlib_path_when_otel_disabled(self):
        tracer = Tracer(otel_enabled=False)
        with tracer.span("asr") as sp:
            sp.add_counter("x", 1)
        self.assertEqual(sp.status, "ok")
        self.assertGreaterEqual(sp.wall_ms, 0.0)

    def test_otel_enabled_without_package_degrades_gracefully(self):
        import memovox.observe as obs

        obs._otel_warned = False  # reset the one-time warning guard
        logger = get_logger("memovox.oteltest")
        handler = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler))
        buf = io.StringIO()
        saved = handler.stream
        handler.stream = buf
        try:
            tracer = Tracer(otel_enabled=True, logger=logger)
            with tracer.span("asr") as sp:  # must NOT raise though otel is absent
                pass
        finally:
            handler.stream = saved
        self.assertEqual(sp.status, "ok")
        if not otel_available():
            self.assertIn("opentelemetry", buf.getvalue().lower())

    def test_settings_otel_enabled_default_false(self):
        self.assertFalse(Settings().otel_enabled)


class SettingsBudgetTest(unittest.TestCase):
    def test_default_budget_mode_is_soft(self):
        self.assertEqual(Settings().budget_mode, "soft")

    def test_from_env_coerces_budget_mode(self):
        with mock.patch.dict(os.environ, {"MEMOVOX_BUDGET_MODE": "hard"}):
            self.assertEqual(Settings.from_env().budget_mode, "hard")


if __name__ == "__main__":
    unittest.main()
