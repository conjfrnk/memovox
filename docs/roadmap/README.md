# memovox Phase 4+ roadmap — execution docs

This directory is the **followable** roadmap: one execution-ready doc per track,
broken into TDD-sized workstreams (red → green → verify → commit), grounded in the
current code with real `file:line` references.

- **Strategic rationale** (why these tracks, how they were derived): [`../superpowers/plans/2026-06-03-phase4-roadmap.md`](../superpowers/plans/2026-06-03-phase4-roadmap.md)
- **How we execute** (TDD rhythm, branch/commit conventions, the two global disciplines): [`CONVENTIONS.md`](CONVENTIONS.md)
- **Live status / checklist**: [`PROGRESS.md`](PROGRESS.md)

## Status snapshot

Phases 0–3 are **done** on `main`. Baseline to keep green at every step:

- `make test` → **247 pass / 2 skip**
- `python -m eval.harness --assert-thresholds` → gates: `hit_rate≥0.6`,
  `groundedness≥0.8`, `contradiction.f1≥0.5`, `synthesis.groundedness≥0.8`.

## How to use these docs

1. Pick the next track per the **build order** below (Wave 0 → 3; respect
   `Depends on`).
2. Open its doc, create a branch `phase4-<track>` (see [`CONVENTIONS.md`](CONVENTIONS.md)).
3. Execute its **Workstreams** top to bottom, one commit each, keeping the gates
   green. Tick the doc's **Definition of done** as you go.
4. Update [`PROGRESS.md`](PROGRESS.md) and the doc's `Status:` line.
5. Resolve that track's **Open questions** before (or early in) building it.

## The tracks

| Track | Title | Wave | Effort | Depends on | W | Doc |
|-------|-------|:----:|:------:|------------|:-:|-----|
| **M0.1** | Observability & metrics spine | 0 | L | — | 8 | [m0-1-observability.md](m0-1-observability.md) |
| **M0.2** | Scale & storage core + incremental consolidation | 0 | XL | M0.1 | 7 | [m0-2-scale-storage.md](m0-2-scale-storage.md) |
| **M0.3** | Word-precise spans, fail-loud device, ingest-signature owner | 0 | M | — | 7 | [m0-3-word-spans-asr.md](m0-3-word-spans-asr.md) |
| **M-X** | Cross-cutting hardening & global disciplines | 0 | M | folds in | 4 | [mx-cross-cutting-hardening.md](mx-cross-cutting-hardening.md) |
| **M1.1** | Visual retrieval leg + named visual defaults | 1 | L | M0.2 | 7 | [m1-1-visual-retrieval.md](m1-1-visual-retrieval.md) |
| **M1.2** | Eval expansion (graph leg, span/citation, gate promotions) | 1 | L | M1.1, M0.3 | 9 | [m1-2-eval-expansion.md](m1-2-eval-expansion.md) |
| **M2.1** | Cross-encoder rerank stage | 2 | M | — (coord M2.2/M2.3) | 4 | [m2-1-rerank.md](m2-1-rerank.md) |
| **M2.2** | Agentic multi-step query planner | 2 | M | M2.1 | 5 | [m2-2-agentic-planner.md](m2-2-agentic-planner.md) |
| **M2.3** | Answer-with-video clip stitching | 2 | M | M0.3, M2.2 | 7 | [m2-3-answer-with-video.md](m2-3-answer-with-video.md) |
| **M3.1** | Decay & versioning | 3 | M | M0.3 (published_at) | 5 | [m3-1-decay-versioning.md](m3-1-decay-versioning.md) |
| **M3.2** | Subscriptions & incremental sync | 3 | L | M0.2 | 7 | [m3-2-subscriptions-sync.md](m3-2-subscriptions-sync.md) |
| **M3.3** | Serving & deployment | 3 | L | M0.2, M3.2 | 8 | [m3-3-serving-deployment.md](m3-3-serving-deployment.md) |
| **M3.4** | Backend A/B benchmark harness | 3 | M | M2.1, M1.1 | 6 | [m3-4-backend-benchmark.md](m3-4-backend-benchmark.md) |

## Build order (critical path)

```
M0.1 observability ─▶ M0.2 scale+incremental-consolidation ─▶ M0.3 word-spans+device
       │                        │                                    │
       │   (M-X disciplines apply throughout)                        │
       └────────────┬──────────┴──────────────┬─────────────────────┘
                    ▼                          ▼
           M1.1 visual retrieval ───────▶ M1.2 eval expansion (talk_c, span/citation, gates)
                    │
                    ▼
     M2.1 rerank ─▶ M2.2 agentic planner ─▶ M2.3 clips    (all touch augur/answer.ask — in this order)
                    │
                    ▼
     M3.1 decay      M3.2 subscriptions(→M0.2)     M3.3 serving(→M0.2,M3.2)     M3.4 benchmark (last)
```

**Recommended first track:** **M0.1** — pure stdlib, highest leverage (gives every
later track its measurement spine), and it makes the contended `consolidate.py`
cap-site a single edit so M0.2 doesn't collide there.

## Single-owner reconciliations (do not duplicate across tracks)

The completeness review caught three areas that were independently re-implementing
shared work. These are now single-owned — other tracks **consume**, never reimplement:

- **Incremental consolidation** → owned by **M0.2**. M3.2 and M3.3 consume it.
- **The `memovox sync` rewrite** → owned by **M3.2**. M3.3 consumes it.
- **The `pipeline.ingest()` signature** (adding `published_at=`, `visual_result=`,
  `modality=`, ASR device knobs as keyword-only) → owned by **M0.3**. M1.1, M3.1
  layer on; they don't re-churn the parameter list.
- **The single span/citation-accuracy metric** → owned by **M1.2** (the ASR track's
  `span_iou` and the eval track's `citation_accuracy` are the *same* metric).
- **The visual golden fixtures + `visual_result` injection seam** → owned by **M1.1**;
  M1.2 consumes them.
- **Two previously-unowned spec stages were added as first-class tracks:** the
  cross-encoder **rerank** (§5) is **M2.1**; the named **Surya OCR / Qwen2.5-VL**
  visual defaults (§7) live in **M1.1**.

## Cross-track decisions to confirm early

These open questions span tracks — settle them before the dependent track starts
(full list lives in each doc's *Open questions*):

- **Watermark cursor key** (M0.2): `max(claim_id)` lexicographic vs
  `videos.ingested_at`. M3.2/M3.3 sync→consolidate ordering depends on it.
- **Ingest-signature superset** (M0.3): land the full keyword-only superset in one
  commit, or per-consumer? Affects M1.1/M3.1 sequencing.
- **Rerank free default** (M2.1): identity vs a proven-non-regressing reorder.
- **Unified span metric name** (M1.2): canonical `span_accuracy` (pure fn
  `span_iou`) — M3.4 benchmark + dashboards consume it.
- **`modality` semantics** (M1.1): route the visual leg ON vs filter citations;
  reconcile with the MCP enum `[any,speech,visual]` vs `Moment.modality`'s
  `speech+slide`.
- **Frozen eval-settings snapshot scope** (M-X): toggles-only vs also numeric knobs.

## The two global disciplines (M-X — apply to every track)

1. **Thin-fixture gating** — every new metric lands **ungated**; it becomes a CI
   gate only once ≥3 stable golden items exercise it (the topic_f1/entity_f1/der
   lesson). Until then it ships with an *equivalence* assertion (off == today).
2. **Frozen eval-settings snapshot** — the harness must pin the growing surface of
   default-OFF flags (`decay_enabled`, `visual_retrieval`, `colpali_enabled`,
   `planner_agentic`, `budget_mode`, `rerank_backend`, `otel`, …), not just
   `_FREE_BACKENDS`, so a future default flip can never silently move a gate number.
