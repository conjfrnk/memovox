"""Observability spine (spec §7 Observability, §9 budgets/throughput).

A pure-stdlib measurement layer the whole pipeline leans on:

* :class:`Tracer` / :class:`Span` — a ``time.perf_counter`` tracer recording, per
  pipeline stage, ``{stage, wall_ms, status, counters, caps}``. Spans collect into
  a per-run, ordered trace.
* :func:`get_logger` — the *single* structured-JSON logging hook. It writes only to
  **stderr** (stdout is reserved for MCP JSON-RPC and human CLI output), with a
  fixed sorted key order so emitted lines are structurally stable.

Everything is deterministic on the free path. Volatile fields (``wall_ms``,
timestamps) are recorded but are excluded from every byte-identity / threshold
assertion, since they are machine-dependent.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from .errors import BudgetExceeded

# status vocabulary for a span
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_DEGRADED = "degraded"

_CONFIGURED_FLAG = "_memovox_json_configured"


@dataclass
class Span:
    """One pipeline stage's timing + counters + cap events.

    ``counters`` are deterministic (claim counts, frames sampled, NLI calls);
    ``caps`` records every place a limit truncated input as a structured event
    ``{name, limit, dropped}`` instead of a silent slice. ``wall_ms`` is volatile.
    """

    stage: str
    wall_ms: float = 0.0
    status: str = STATUS_OK
    counters: Dict[str, float] = field(default_factory=dict)
    caps: List[dict] = field(default_factory=list)

    def add_counter(self, name: str, value: float = 1) -> float:
        """Accumulate ``value`` into the named counter; return the running total."""
        self.counters[name] = self.counters.get(name, 0) + value
        return self.counters[name]

    def add_cap(self, name: str, *, limit: int, dropped: int) -> dict:
        """Record a structured cap event (a limit truncated ``dropped`` items)."""
        event = {"name": name, "limit": int(limit), "dropped": int(dropped)}
        self.caps.append(event)
        return event

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "wall_ms": self.wall_ms,
            "status": self.status,
            "counters": dict(self.counters),
            "caps": list(self.caps),
        }


class Budget:
    """A per-video token/compute budget (spec §9).

    "Units" are a coarse, deterministic count defined per stage (moments
    processed + frames sampled + NLI calls), not LLM tokens — so the free path
    has a meaningful, stable budget. ``mode="soft"`` (the free default) records
    overage and never raises; ``mode="hard"`` raises :class:`BudgetExceeded` as
    soon as the running total passes ``max_units``. ``max_units=None`` is
    unbounded.
    """

    def __init__(self, max_units: Optional[int] = None, mode: str = "soft") -> None:
        self.max_units = max_units
        self.mode = mode
        self.used: float = 0

    @property
    def exceeded(self) -> bool:
        return self.max_units is not None and self.used > self.max_units

    @property
    def overage(self) -> float:
        if self.max_units is None:
            return 0
        return max(0, self.used - self.max_units)

    def charge(self, units: float, *, label: Optional[str] = None) -> float:
        """Add ``units`` to the running total; return the current overage.

        In hard mode, raises :class:`BudgetExceeded` when the total exceeds the
        cap. In soft mode (default) it only records — callers read ``exceeded`` /
        ``overage`` afterward.
        """
        self.used += units
        if self.exceeded and self.mode == "hard":
            where = f" at {label}" if label else ""
            raise BudgetExceeded(
                f"budget exceeded{where}: used {self.used} > {self.max_units} units"
            )
        return self.overage

    def to_dict(self) -> dict:
        return {
            "max_units": self.max_units,
            "used": self.used,
            "mode": self.mode,
            "exceeded": self.exceeded,
            "overage": self.overage,
        }


class Tracer:
    """Collects an ordered list of :class:`Span` for one top-level operation.

    Use ``with tracer.span("asr") as span:`` around each stage. On clean exit the
    span is stamped ``status="ok"`` with its ``wall_ms``; if the body raises, the
    span is stamped ``status="error"`` and the exception re-raised (never
    swallowed). Each completed span is logged as one JSON line to stderr.
    """

    def __init__(self, name: str = "memovox", *, logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.spans: List[Span] = []
        self._logger = logger if logger is not None else get_logger("memovox.trace")

    @contextmanager
    def span(self, stage: str) -> Iterator[Span]:
        sp = Span(stage=stage)
        self.spans.append(sp)
        t0 = time.perf_counter()
        try:
            yield sp
        except BaseException:
            sp.status = STATUS_ERROR
            sp.wall_ms = (time.perf_counter() - t0) * 1000.0
            self._log_span(sp)
            raise
        else:
            sp.wall_ms = (time.perf_counter() - t0) * 1000.0
            self._log_span(sp)

    def _log_span(self, sp: Span) -> None:
        self._logger.info("span", extra={"span": sp.to_dict()})

    def find(self, stage: str) -> Optional[Span]:
        """Return the most recent span for ``stage`` (or ``None``)."""
        for sp in reversed(self.spans):
            if sp.stage == stage:
                return sp
        return None

    def to_dict(self) -> dict:
        """The metrics payload attached to reports — the ordered span list."""
        return {"name": self.name, "spans": [s.to_dict() for s in self.spans]}


class _JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object with a fixed, sorted key order.

    ``ts`` is a volatile field (record creation time); it is emitted for real
    observability but excluded from every byte-identity assertion in the suite.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "logger": record.name,
            "level": record.levelname,
            "msg": record.getMessage(),
            "ts": record.created,
        }
        span = getattr(record, "span", None)
        if span is not None:
            payload["span"] = span
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def get_logger(name: str = "memovox") -> logging.Logger:
    """Return the structured logger for ``name``, configured once, to **stderr**.

    Idempotent: repeated calls return the same logger without stacking handlers.
    The handler is hard-wired to ``sys.stderr`` so no structured log can ever leak
    onto stdout (where the MCP server speaks JSON-RPC).
    """
    logger = logging.getLogger(name)
    if not getattr(logger, _CONFIGURED_FLAG, False):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # don't double-emit through the root handler
        setattr(logger, _CONFIGURED_FLAG, True)
    return logger
