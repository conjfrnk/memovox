# memovox — Phase 4+ roadmap ("Scale & polish" + cross-cutting fidelity)

**Status:** planning (Phases 0–3 done + on `main`).
**Date:** 2026-06-03.
**Method:** grounded multi-agent inventory of spec §8–§12 vs the actual code → one
proposal per area → adversarial completeness critic. This doc is the reconciled
output (overlaps merged, missing spec items folded in, dependencies fixed).

Spec §11 names Phase 4 as "channel/playlist subscriptions + incremental sync,
answer-with-video clip stitching, ColPali visual retrieval, decay/versioning,
dashboards." The inventory found those plus a tail of **cross-cutting** items the
spec requires elsewhere (§5 rerank, §7 named backends, §9 throughput/observability/
scale, §10 eval dimensions, §12 risks) that are unbuilt. This roadmap covers all
of it, sequenced.

## Carry-forward non-negotiables (unchanged from Phases 0–3)

- Free / stdlib-only / deterministic / idempotent core; every model **and storage**
  slot has a deterministic fallback behind a common interface.
- Provenance is sacred; eval-driven (golden gates) — no merge regresses a gate.
- **New discipline (from the critique):** (a) a *frozen eval-settings snapshot* —
  the harness must pin the growing surface of default-OFF flags, not just
  `_FREE_BACKENDS`, so a future default flip can't silently move gate numbers;
  (b) *thin-fixture discipline* — every new metric lands **ungated**, and is gated
  only once ≥3 stable golden items exist (the topic_f1/entity_f1/der lesson).

## Planning fixes the critique forced (resolve before building)

1. **Single owner for incremental consolidation.** Three areas re-implemented a
   consolidation watermark/scoping. It lives **once** in the Scale track (M0.2);
   Sync and Serving *depend on it*, never reimplement it.
2. **Single owner for the sync rewrite.** Subscriptions and Serving both rewrote
   `Memovox.sync()`. It lives **once** in the Subscriptions track (M3.2); Serving
   consumes it.
3. **Resolve the dangling dependency.** `serving-deployment` depended on a
   non-existent "scale-incrementality" key — that is M0.2; Serving is now correctly
   blocked on it.
4. **Add two un-owned spec stages as first-class work:** a **cross-encoder rerank**
   leg (§5 "rerank with a cross-encoder", §3 diagram) and the **named visual
   defaults** Surya OCR + Qwen2.5-VL VLM (§7). Folded into M2.1 and M1.1.
5. **One span/citation-accuracy metric, not two.** `span_iou` (ASR track) and
   `citation_accuracy` (eval track) are the same metric — unified in M1.2.
6. **One visual golden subset + one `visual_result` injection seam** — owned by
   M1.1, consumed by M1.2 (not duplicated).
7. **Coordinate the `pipeline.ingest` signature.** `published_at=`, `visual_result=`,
   `modality=`, and ASR device knobs all touch its parameter list — change it once,
   in M0.3, with the others layered as keyword-only additions.

---

## The waves (reconciled tracks + build order)

Build order is the critic's recommended order, regrouped into waves. Each track is
TDD'd, committed per sub-task, and keeps all gates green. Effort: S/M/L/XL.

### Wave 0 — Foundations (unblock everything; prevent collisions)

**M0.1 Observability & metrics spine** — L — *do first.*
Per-stage cost/latency/counter tracer on `time.perf_counter` + structured logging
to **stderr** (never stdout — MCP speaks JSON-RPC there); a per-video token/compute
**budget** (soft default); and **surface every silent cap** (`find_contradictions`
`max_claims=600`, retrieve pool/top_k, frame caps) as structured events. Optional
OpenTelemetry behind `[otel]`. *Why first:* gives every later area its measurement
spine and the single logging hook, so latency/throughput claims are measurable and
the `consolidate.py` cap-site is edited once.
Gate: `observability` block — every stage emits a span (status ok, wall_ms≥0),
counters reconcile (Σ stage claim counts == committed+unsupported), a forced-small
budget surfaces a cap event. wall_ms magnitudes are **not** thresholded.

**M0.2 Scale & storage core (free-path)** — XL (core is L; ANN backends opt-in) —
*owns incremental consolidation.* `VectorIndex / LexicalIndex / GraphStore` ABCs
mirroring the model-backend registry, with the current SQLite impl as the free
default (a pure no-op refactor first, behind a byte-identical **parity gate**).
Kill the brute-force cosine on the free path (unit-normalize stored vectors +
batched `struct` scoring + optional FTS5 candidate prefilter). Make consolidation
**incremental + observable**: a `consolidation_watermark`, scan new-claims-vs-**all**
(never new-vs-new), replace the silent `max_claims` truncation with reported paging.
Optional ANN/Kùzu/Tantivy backends are **opt-in, never in the CI gate** (defer to
opportunistic work). *Why second:* satisfies Serving's dependency and gives Sync
its incremental consolidate.
Gate: free-path **parity** (top-k byte-identical to today) + **incremental==full**
equivalence on the golden corpus; opt-in `eval/scale.py` for p95/recall.

**M0.3 Word-precise spans + fail-loud device (free-path of the ASR track)** — M.
Thread optional `Word`s through `SegmentRef`/`Moment.segments`; `locate_span`
tightens a claim's `(t0,t1)` to the matched word window when words exist (identity
fallback when absent → free path unchanged). Add `DevicePlacementError` so
large-v3 silently on CPU **fails loud** (with `--allow-cpu` escape). *Why here:*
clip + citation-accuracy gates must target word-precise spans, not the cue-granular
ceiling. **Owns the `pipeline.ingest` signature change** (keyword-only additions).
Gate: `span_iou` ungated-then-gated on a word-bearing fixture; **free VTT corpus
span output asserted unchanged** (no silent widening). Optional WhisperX/pyannote
backends (turns + voiceprints) trail as opt-in.

### Wave 1 — The multimodal payoff (the core thesis bet) + the eval that proves it

**M1.1 Visual retrieval leg + named visual defaults** — L. Wire the already-stored
per-keyframe visual vectors as a **4th RRF leg** (default OFF; fires only on
`modality/strategy=='visual'`), kill the dead MCP `modality` param, add a
deterministic **frame-type classifier** (slide/document/diagram vs talking-head).
Add a `VisualEmbedder` interface; **opt-in** ColPali multi-vector late-interaction
gated to slide/doc/diagram frames, plus the spec's named **Surya OCR** and
**Qwen2.5-VL VLM** backends (free path keeps the signature/NullVLM). **Owns the
visual golden fixtures + the `visual_result` injection seam.** *Guard:* never cosine
the 256-d hashing **text** vector against the 256-d grayscale **visual** signature —
add a space tag, not just a comment.
Gate: `multimodal` block (transcript-only vs tri-modal `hit_rate` lift on on-screen-
only QA), ungated-then-gated.

**M1.2 Eval expansion** — L — *serialize; owns the talk_c re-baseline.* Add
`talk_c.en.vtt` whose gold answer is reachable **only via a CONTRADICTS/SUPPORTS
edge** (exercise the §5 graph leg end-to-end), the unified **span/citation-accuracy**
metric, the **keyframe-efficiency** curve (adaptive vs uniform), `topic_f1`, and
promote `entity_f1`/`der` to gates once talk_c numbers are verified. Re-baseline
contradictions/entities/speakers gold **in one commit**.
Gate: graph-only QA item misses without the graph leg, hits with it; the rest
ungated-then-gated per the thin-fixture discipline.

### Wave 2 — "rerank → grounded answer + clips" (the answer pipeline; coordinate — all touch `answer.ask`)

**M2.1 Cross-encoder rerank** — M — *MISSING area, now first-class.* A rerank stage
between fused retrieval and answer synthesis, behind a `Reranker` backend interface
with a **deterministic free fallback** (e.g. lexical/cross-overlap reorder) and an
optional cross-encoder upgrade. Default behavior must keep gates byte-identical
(free reranker = identity or a proven non-regressing reorder).
Gate: rerank improves (or holds) `mrr/ndcg` on golden; off==today.

**M2.2 Agentic multi-step planner** — M. Decompose multi-part questions into
sub-queries → per-sub-query strategy/modality → execute through existing legs
(multi-hop where applicable) → fuse → one cited answer; surface the `plan`.
Deterministic decomposer free path; LLM decomposer optional with guaranteed
fallback. Single-clause input degrades **exactly** to today.
Gate: `plan.subquery_recall` on a 2-part golden item; single-clause path
byte-identical citations.

**M2.3 Answer-with-video clip stitching** — M. Merge adjacent/overlapping cited
Moment spans per video into minimal deep-linked **clips** ("the 92 seconds…"),
ranged YouTube deep links, surfaced through Answer/REST/CLI/MCP. Pure arithmetic;
optional ffmpeg concat only when local media present. Now feeds on word-precise
spans (M0.3) and reranked/visual-aware retrieval.
Gate: `clip.coverage` (IoU≥0.3 vs gold clip) + non-overlap/idempotency invariants.

### Wave 3 — Library operations, deployment, and measurement

**M3.1 Decay & versioning** — M — default OFF. Recency down-weighting (reusing the
consensus half-life model), demotion of fully-superseded moments, `claim_history`
lineage + a `timeline` surface; **published_at injection** for local sources (so
decay/temporal stories work on local files and the golden corpus).
Gate: `decay` block (recent-first ordering, superseded excluded) — default-off run
byte-identical to today.

**M3.2 Subscriptions & incremental sync** — L — *owns the sync rewrite; consumes
M0.2.* `enumerate_source` (yt-dlp `--flat-playlist`) → skip already-seen via a
persisted `sync_state` cursor **before download** → per-entry error isolation →
batched incremental consolidation. `subscribe`/`subscriptions`/`unsubscribe` CLI.
Gate: `incremental` block — incremental==batch equivalence + idempotent re-sync
(2nd sync ingests 0).

**M3.3 Serving & deployment** — L — *blocked on M0.2.* A stdlib threading + SQLite
**job runner** (queued→running→succeeded/failed, backoff retry) so `consolidate`/
`sync`/`ingest` run async/resumable; non-blocking MCP `consolidate` + `job_status`;
optional **FastAPI behind `[serve]`** sharing one `routes.py` with the stdlib server;
`memovox-worker` console script + `Dockerfile`/`DEPLOY.md`; a `local_only` privacy
toggle. *Honesty:* default worker concurrency = 1 (deterministic); >1 is opt-in and
out of the eval gate (the real throughput win waits on M-X.1 frame parallelism).
Gate: `serving.equivalent` — background+drained == inline consolidate; stdlib==FastAPI
JSON parity.

**M3.4 Backend A/B benchmark harness** — M — *last; most to rank by now.*
Parameterize the eval harness over named `BackendConfig`s; emit a deterministic
ranking table; auto-shrink to the single free row on a bare machine. Now there are
real upgrade slots (embedder, NLI, rerank, Surya/Qwen) to compare.
Gate: free row metric-identical to `run_eval()`; determinism (two runs identical).

### Cross-cutting hardening (fold into the waves; tracked so they aren't dropped)

- **M-X.1 Visual-track parallelism** (§9 "visual track is THE bottleneck") — the per-
  keyframe loop is serial; parallelize frame work (process/thread pool) — the only
  real throughput lever. Fold into M1.1 / Serving; measured by M0.1.
- **M-X.2 Reliability: persisted visual-backfill flag** (§9 graceful degradation) —
  persist "visual layer missing" on Video/Moment + a targeted re-process path
  (not whole-video re-ingest). + **ffprobe ASR-readiness gate** (§4.1) + **captions-
  as-fused-prior** (§9 cost lever). Fold into M0.3 / M3.2.
- **M-X.3 Claim-granularity tuning** (§12) — a claims-per-moment vs groundedness/
  salience curve + an extraction-prompt/granularity knob. Fold into M1.2 eval.
- **M-X.4 Structured-JSON extraction output mode** (§5) — schema-targeted entity/
  claim extraction output (distinct from the current Answer JSON dump). Small.
- **M-X.5 ToS / private-by-default posture** (§12/§9) — beyond `local_only`: respect
  source terms, private-by-default store, a retention/redaction surface. Policy +
  small code.

---

## Critical-path summary (recommended order)

```
M0.1 observability  →  M0.2 scale+incremental-consolidation (free core)  →  M0.3 word-spans+device
        │                         │                                              │
        └──────────────┬─────────┴───────────────┬──────────────────────────────┘
                       ▼                          ▼
              M1.1 visual retrieval  ───────►  M1.2 eval expansion (talk_c, span/citation, gates)
                       │
                       ▼
        M2.1 rerank → M2.2 agentic planner → M2.3 clips   (coordinate: all touch answer.ask)
                       │
                       ▼
        M3.1 decay   M3.2 subscriptions(→M0.2)   M3.3 serving(→M0.2)   M3.4 benchmark(last)
```

The spec's *literal* Phase 4 bullets are M1.1 (ColPali/visual), M2.3 (answer-with-
video), M3.1 (decay/versioning), M3.2 (subscriptions/sync), and "dashboards"
(= M0.1 observability surfaced + M3.4). Everything else is the cross-cutting
fidelity/scale/quality the spec requires elsewhere and which the inventory found
unbuilt.

## Top risks to manage (from the adversarial pass)

1. **Eval thinness / gate flakiness** is the dominant risk — every new gate is ~1–3
   hand-authored items; honor the land-ungated-then-gate discipline and grow the
   golden corpus before hard-gating.
2. **Determinism erosion** from a growing surface of default-OFF flags — pin a frozen
   eval-settings snapshot, not just `_FREE_BACKENDS`.
3. **Incremental-consolidation equivalence is genuinely hard** (hub-token quadratic +
   the `dedup_claims`/`find_contradictions` cap disagreement) and a 2–3 video corpus
   won't exercise the failure — validate at synthetic scale.
4. **Word-window span tightening vs the verification gate** — the displayed citation
   span must not drift from the span the NLI gate actually verified (provenance-is-
   sacred); keep premise construction segment-granular and assert it.
5. **MCP stdout discipline** repo-wide — all logging to stderr; no stray `print` in the
   job worker.
6. **Threading + SQLite** under the job runner — per-job connections, WAL +
   busy_timeout, serial-by-default; concurrency is opt-in and ungated.
7. **talk_c re-baseline blast radius** — adding a 3rd golden video perturbs several
   metrics at once; serialize that commit against all other harness edits.

## Definition of done (per wave)

- Wave 0: measurement spine live; free-path scale parity + incremental==full;
  word-precise spans with the free corpus unchanged.
- Wave 1: a measurable multimodal lift on on-screen-only knowledge; the §5 graph leg
  exercised end-to-end by talk_c.
- Wave 2: multi-part questions fully answered, reranked, and returned as stitched
  deep-linked clips.
- Wave 3: subscriptions sync incrementally; consolidation runs async/resumable;
  optional FastAPI/worker deploy; backends are ranked, not assumed.
