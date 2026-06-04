# Phase 4+ progress tracker

Status legend: ⬜ not started · 🟡 in progress · ✅ done.
Update a row when you start/finish a track; tick the detailed workstream checklist
inside each track doc (its *Definition of done*) as you go. Keep this file and each
doc's `Status:` line in sync.

_Last updated: 2026-06-04 — Wave 0 COMPLETE (M0.1+M0.2+M0.3+M-X) merged to main (353 pass / 2 skip; 7 gates green)._

## Baseline (keep green at every commit)

- `make test` → 353 pass / 2 skip (was 247 at Phase-3; +36 M0.1, +26 M0.2, +33 M0.3, +11 M-X)
- `python -m eval.harness --assert-thresholds` → `hit_rate≥0.6`, `groundedness≥0.8`,
  `contradiction.f1≥0.5`, `synthesis.groundedness≥0.8`, `parity==1.0`,
  `incremental_equivalence==1.0`, `span_unchanged==1.0` (+ ungated `observability`, `span_accuracy`)

## Wave 0 — foundations

| Track | Status | Branch | W done | Notes |
|-------|:------:|--------|:------:|-------|
| M0.1 Observability & metrics spine | ✅ | phase4-observability | 8/8 | merged; owns the stderr log hook + consolidate cap site |
| M0.2 Scale & storage + incremental consolidation | ✅ | phase4-scale-storage | 7/7 | merged; owns incremental consolidation (rowid watermark) + storage ABCs |
| M0.3 Word-precise spans / device / ingest-signature | ✅ | phase4-word-spans | 7/7 | merged; owns the `pipeline.ingest` signature |
| M-X Cross-cutting hardening & disciplines | ✅ | phase4-cross-cutting | 3/3+ledger | merged; owns the 2 eval disciplines (snapshot completeness + thin-fixture guard) + X.4 extract |

## Wave 1 — multimodal payoff + eval

| Track | Status | Branch | W done | Notes |
|-------|:------:|--------|:------:|-------|
| M1.1 Visual retrieval + named defaults | ⬜ | — | 0/7 | owns visual fixtures + `visual_result` seam |
| M1.2 Eval expansion (talk_c, span/citation, gates) | ⬜ | — | 0/9 | owns the talk_c re-baseline (serialize) |

## Wave 2 — answer pipeline (build in this order; all touch `augur/answer.ask`)

| Track | Status | Branch | W done | Notes |
|-------|:------:|--------|:------:|-------|
| M2.1 Cross-encoder rerank | ⬜ | — | 0/4 | first of the trio |
| M2.2 Agentic planner | ⬜ | — | 0/5 | after rerank |
| M2.3 Answer-with-video clips | ⬜ | — | 0/7 | after planner; needs M0.3 spans |

## Wave 3 — library ops, deployment, measurement

| Track | Status | Branch | W done | Notes |
|-------|:------:|--------|:------:|-------|
| M3.1 Decay & versioning | ⬜ | — | 0/5 | default-OFF; needs M0.3 published_at |
| M3.2 Subscriptions & incremental sync | ⬜ | — | 0/7 | owns sync rewrite; consumes M0.2 |
| M3.3 Serving & deployment | ⬜ | — | 0/8 | consumes M0.2 + M3.2 |
| M3.4 Backend A/B benchmark | ⬜ | — | 0/6 | last; ranks M2.1/M1.1 slots |

## Open questions awaiting a human decision

Cross-track ones are summarized in [`README.md`](README.md#cross-track-decisions-to-confirm-early);
each track doc has its own *Open questions* section. Resolve a track's questions
before (or early in) building it.
