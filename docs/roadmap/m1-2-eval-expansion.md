# M1.2 — Eval expansion (graph leg, span/citation, keyframe, gate promotions)

> **Wave:** 1 · **Effort:** L · **Status:** not started
> **Depends on:** M1.1 (visual fixtures + `visual_result` injection seam), M0.3 (word-precise spans + `pipeline.ingest` signature) · **Owns (single-owner concerns):** the `talk_c` re-baseline (a single serialized commit — coordinate against ALL other harness edits) · **Blocks:** none
> **Spec:** §10 (eval dimensions), §12 (risks / claim-granularity)

## Goal
Grow the golden corpus and the metric surface so the eval harness actually *proves*
the Phase-4 capability bets instead of asserting them. Add a third golden talk
(`talk_c.en.vtt`) whose gold answer is reachable **only** through a
CONTRADICTS/SUPPORTS graph edge — exercising the §5 graph leg end-to-end with a QA
item that **misses** on dense+lexical and **hits** with the graph leg on. Unify the
two competing span metrics (`span_iou` from the ASR track, `citation_accuracy` from
the eval track) into ONE span/citation-accuracy metric over gold spans. Add the
keyframe-efficiency curve (adaptive `select_keyframes` vs uniform sampling at equal
accuracy), `topic_f1`, and a §12 claim-granularity curve. Promote `entity_f1`/`der`
to hard gates *only after* the `talk_c` numbers are verified, and land a
frozen-eval-settings snapshot guard so the growing surface of default-OFF flags
cannot silently move gate numbers. Every new metric lands **ungated**, gated only
once ≥3 stable golden items exist.

## Why it matters
The whole program is "eval-driven: no merge regresses a gate." Today the harness
gates only four numbers (`hit_rate`, `groundedness`, `contradiction.f1`,
`synthesis.groundedness`) over a **2-talk** corpus, and the §5 graph-retrieval leg
— shipped in `augur/retrieve.py` and `augur/traverse.py` — is exercised only by
hand-built unit stores in `tests/test_graph_retrieval.py`, never end-to-end through
`mv.ask()` over the golden corpus. That means the marquee multi-hop capability has
**no golden gate**: a regression that silently disables the graph leg in the real
ask path would pass CI. This track closes that gap and turns the
already-shipped-but-unmeasured fidelity work (word-precise spans from M0.3, visual
retrieval from M1.1, adaptive keyframing in `tessera/keyframes.py`) into guarded,
non-regressing capabilities. It is the eval that makes Wave 1's "multimodal +
graph payoff" claim falsifiable.

## Scope (reconciled)

In scope:
- **`talk_c.en.vtt` + a `graph_only` QA item** whose gold moment shares no query
  terms with the question and is reachable only via a CONTRADICTS/SUPPORTS edge.
  Assertion: the item **misses** with the graph leg off, **hits** with it on.
- **Unified span/citation-accuracy metric** — merge `span_iou` (ASR track) and
  `citation_accuracy` (eval track) into ONE metric computed over gold
  `(t_start_s, t_end_s)` spans: IoU of each cited citation's span against the gold
  span for its QA item. Lands ungated.
- **Keyframe-efficiency curve** — adaptive `select_keyframes` vs uniform sampling at
  equal accuracy (frames-kept ratio at matched coverage). Lands ungated; an
  informational curve in the report, not a threshold.
- **`topic_f1`** — pairwise clustering F1 over `eval/golden/topics.json` gold topic
  clusters vs the induced `topic_id` partition. Lands ungated.
- **Promote `entity_f1` / `der` to gates** — currently computed and already 1.0 on
  the 2-talk corpus but deliberately UNGATED. Promote ONLY after verifying `talk_c`
  does not perturb them below threshold.
- **FOLDED IN (§12 claim-granularity curve)** — claims-per-moment vs
  groundedness/salience, surfaced as an informational curve (the lever for the
  M-X.3 extraction-granularity knob). Ungated.
- **Re-baseline `contradictions.json` / `entities.json` / `speakers.json`** to
  cover `talk_c` — **in ONE commit** (the single-owner re-baseline this track owns).
- **Frozen-eval-settings snapshot guard** — a test that pins the full set of
  default-OFF flags / non-default `Settings`, not just `_FREE_BACKENDS`, so a future
  default flip cannot silently move gate numbers.

Non-goals / deferrals:
- **No new retrieval/graph/keyframe *implementation*.** The graph leg
  (`augur/traverse.py`), adaptive keyframing (`tessera/keyframes.py`), and
  word-precise spans (M0.3) already exist; this track only *measures* them. If the
  `graph_only` item cannot be made to hit, that is a finding to escalate, not a
  license to rewrite the leg here.
- **The `visual_result` injection seam and visual golden fixtures are owned by
  M1.1** — this track *consumes* them for any visual-anchored span/keyframe asserts;
  it does not define them.
- **Word-precise span plumbing (`Word`s through `SegmentRef`/`Moment.segments`,
  `locate_span` tightening) is owned by M0.3** — the unified span metric *targets*
  word-precise spans but does not implement them.
- **No hard gate on any new metric in its first landing commit** — thin-fixture
  discipline (gate only at ≥3 stable items).
- **ANN/Kùzu/Tantivy backends remain opt-in and out of the gate** (M0.2 concern).

## Current state (grounded)

- **Harness + gates:** `eval/harness.py` pins the free stack in `_FREE_BACKENDS`
  (`eval/harness.py:62-70`) and gates exactly four numbers — `_HIT_RATE_GATE = 0.6`,
  `_GROUNDEDNESS_GATE = 0.8`, `_CONTRADICTION_F1_GATE = 0.5`,
  `_SYNTHESIS_GROUNDEDNESS_GATE = 0.8` (`eval/harness.py:642-649`), enforced in
  `_check_thresholds` (`eval/harness.py:656-670`). `entity_f1`/`der` are computed
  (`_compute_report`, `eval/harness.py:600-604`) and returned but **deliberately
  ungated** — the comment at `eval/harness.py:638-648` and the test
  `test_entity_f1_and_der_are_ungated` (`tests/test_eval.py:355-357`) document this.
  `topic_f1` is explicitly **not** a golden gate today (comment at
  `eval/harness.py:646-648`); topic quality is covered only by
  `tests/test_topics.py`.
- **Verified baseline (run just now):** `make test` => **247 pass / 2 skip**;
  `python -m eval.harness --assert-thresholds` => `hit_rate 1.0`, `groundedness 1.0`,
  `entity_f1 1.0`, `der 1.0`, `contradiction.f1 1.0`, `synthesis.groundedness 1.0`,
  all gates pass. So `entity_f1`/`der` are *already* at 1.0 on the 2-talk
  corpus — the only thing blocking their promotion is `talk_c` not yet existing to
  prove the number is stable across a 3-talk corpus.
- **Golden corpus:** `eval/golden/` holds `talk_a.en.vtt`, `talk_b.en.vtt`,
  `qa.json` (5 items, `eval/golden/qa.json:1-27`), `entities.json`
  (`eval/golden/entities.json`), `speakers.json` (`eval/golden/speakers.json`),
  `contradictions.json` (one `talk_a`/`talk_b` pair, `eval/golden/contradictions.json:1-8`),
  and the `README.md` documenting the shared/distinct design
  (`eval/golden/README.md`). There is **no** `topics.json` yet, and **no** gold span
  field on `qa.json` items.
- **Graph leg (the thing `talk_c` must exercise):** `retrieve(..., use_graph=True)`
  fuses a third GRAPH leg via `expand(...)` (`augur/retrieve.py:30-65`); `expand`
  walks claim→claim SUPPORTS/CONTRADICTS/ELABORATES edges in both directions
  (`augur/traverse.py:32-78`). Crucially, in the **ask** path the leg is turned on
  **only** when the planner returns `strategy == "contradiction"`
  (`augur/answer.py:81`, with `graph_rels = ["CONTRADICTS", "SUPPORTS"]` at
  `augur/answer.py:86`). The planner routes to `"contradiction"` only when the query
  contains one of `_CONTRADICTION = ("contradict", "disagree", "conflict",
  "inconsistent", "dispute")` (`augur/planner.py:21,32-33`). **This is the seam: the
  `graph_only` QA item's question text must contain one of those trigger words** so
  `mv.ask()` actually turns the graph leg on. `traverse.py:18-22` also notes
  ELABORATES is intra-moment-only today, so the edge must be CONTRADICTS or SUPPORTS.
- **End-to-end graph coverage gap:** `tests/test_graph_retrieval.py` proves the leg
  works on hand-built stores (`test_graph_off_omits_only_linked_moment` /
  `test_graph_on_surfaces_only_linked_moment`, `tests/test_graph_retrieval.py:105-116`)
  but never goes through `mv.ask()` or the golden harness. The QA collector
  `_retrieval_and_groundedness` (`eval/harness.py:319-341`) calls `ing.mv.ask(item["q"])`
  and scores `relevant_moment_substrings` — it has everything needed to assert a
  graph-only hit/miss, but no item currently exercises it.
- **Span metric:** `assay/spans.py:locate_span` (`assay/spans.py:19-42`) tightens a
  claim's span to the best-overlap segment window; `Citation` carries
  `t_start_s`/`t_end_s` (`augur/types.py:11-21`). There is **no** span/citation
  accuracy metric in the harness, and `qa.json` carries no gold span. M0.3 threads
  `Word`s for word-precise spans; this metric targets those.
- **Keyframe selection:** `tessera/keyframes.py:select_keyframes`
  (`tessera/keyframes.py:19-39`) is the adaptive info-gain selector; defaults
  `keyframe_min_gain = 0.12`, `keyframe_per_scene_cap = 8` (`config.py:50-51`). No
  uniform-baseline comparison or efficiency curve exists.
- **Topics:** `loom/topics.py:induce_topics` (`loom/topics.py:71-107`) clusters
  moments into `topic:<slug>` nodes and stamps `Moment.topic_id`; pairwise F1
  machinery already exists as `clustering_f1` (`eval/harness.py:169-185`), reused for
  `entity_f1`/`der`. So `topic_f1` is a thin addition: a gold cluster file + a
  read-the-persisted-`topic_id` collector mirroring `_entity_clusters`
  (`eval/harness.py:344-412`).
- **Snapshot guard:** none exists. `_FREE_BACKENDS` (`eval/harness.py:62-70`) pins
  the seven model backends, but `Settings` (`config.py:27-71`) has a growing surface
  of behavioral defaults (`visual_enabled=True`, `top_k=8`, `rrf_k=60`,
  `contradiction_threshold=0.55`, `keyframe_min_gain`, `topic_similarity`, etc.) that
  are NOT pinned, and `voiceprint_backend="auto"` is a default-OFF-on-free-path flag
  not in `_FREE_BACKENDS`.
- **Consolidation cap (blast-radius source):** `find_contradictions` truncates to
  `max_claims=600` (`loom/consolidate.py:56-67`) and is invoked by the contradiction
  metric path via `ing.mv.contradictions()` (`eval/harness.py:520`,
  `loom/consolidate.py:229`). A third talk grows the cross-video claim pairing space,
  so `contradiction.f1` can move when `talk_c` lands.

## Free-path guarantee
- **Everything new is computed under the already-pinned free stack.** `talk_c` is a
  WEBVTT caption file ingested via the `captions` ASR backend (no audio/model), the
  same path `talk_a`/`talk_b` use; `_ingest_golden` (`eval/harness.py:229-238`)
  applies `_FREE_BACKENDS` unchanged.
- **The graph-only item routes through the existing free planner** — its question
  text triggers `strategy == "contradiction"` (`augur/planner.py:32-33`), turning on
  the deterministic `expand()` walk; no model, no network.
- **No default flips.** `visual_enabled` stays `True` but the visual leg fires only
  on `modality/strategy=='visual'` (M1.1 concern); the span/keyframe/claim-granularity
  curves are **read-only** measurements over the persisted store — they add report
  keys, never change pipeline behavior.
- **New metrics land UNGATED.** Adding report keys and ungated curves cannot change
  `_check_thresholds` output, so `--assert-thresholds` stays byte-identical until a
  metric is explicitly promoted (and only after ≥3 stable items).
- **Must stay byte-identical until the re-baseline commit:** the four current gate
  numbers over `talk_a`+`talk_b`. The re-baseline commit (W7) is the *only* commit
  permitted to move them, and only because `talk_c` legitimately enlarges the
  corpus — it must be serialized against all other harness edits.
- **The snapshot guard (W8) is the enforcement mechanism:** it freezes the full
  default-OFF / non-default-`Settings` surface so a later track's default flip can't
  silently move a gate.

## Workstreams

> **Ordering rule:** W1–W6 are additive and ungated (safe to interleave with other
> tracks). **W7 (re-baseline) and any gate promotion (W9) are the serialized,
> single-owner commits** — land them last, after `talk_c` numbers are verified, and
> coordinate against every other harness edit. M0.3 (word spans) must be merged
> before W3 asserts word-precise IoU; M1.1 (visual fixtures) before any
> visual-anchored span/keyframe assert.

### W1 — `talk_c.en.vtt` fixture + graph-only QA item · M
- **Files:** create `eval/golden/talk_c.en.vtt`; modify `eval/golden/qa.json`,
  `eval/golden/README.md`.
- **Design:** `talk_c` must contain a claim that **CONTRADICTS or SUPPORTS** a claim
  in `talk_a`/`talk_b` (so the free consolidation path writes the cross-video edge),
  AND the gold-answer moment's transcript must share **no** query terms with the QA
  question (so dense+lexical cannot surface it — mirror the `photosynthesis`/
  `mitochondria` design in `tests/test_graph_retrieval.py:47-53`). The QA question
  **must contain a planner contradiction trigger word** (`contradict`, `disagree`,
  `conflict`, `inconsistent`, `dispute`) so `mv.ask()` sets
  `use_graph=True` (`augur/answer.py:81`, `augur/planner.py:32-33`). Example shape:
  a `talk_c` moment by a new speaker that *disagrees with* the talk_a "scaling laws
  will continue to hold" claim using different vocabulary, reachable only by walking
  the CONTRADICTS edge from the term-matching seed.
- **Red (failing test first):** add to `tests/test_eval.py` a
  `test_graph_only_item_misses_without_graph_and_hits_with_graph`: ingest the golden
  corpus, call `mv.ask(graph_only_q)` (graph on via planner) — assert the gold
  moment id is in the citations; then call `retrieve(..., use_graph=False)` (or an
  ask with a non-trigger paraphrase) — assert the gold moment is **absent**. Fails
  today because no such item/fixture exists.
- **Green (implement):** author `talk_c.en.vtt` and append the `graph_only` QA item
  to `qa.json` with `relevant_moment_substrings` matching the graph-reachable moment.
  Tune the vocabulary until off-misses / on-hits.
- **Verify:** the new test passes; `make test` stays green; re-run
  `python -m eval.harness` and **record** the new `entity_f1`/`der`/`contradiction.f1`/
  `topic_f1` numbers (do NOT gate yet — this is the verification input for W7/W9).
- **Commit:** `test(eval): talk_c graph-only QA item exercises the §5 graph leg e2e`

### W2 — unified span/citation-accuracy metric (machinery + unit tests) · M
- **Files:** modify `eval/harness.py` (add `span_iou(pred, gold)` pure fn +
  `_span_accuracy(ing, qa, ...)` collector + `"span_accuracy"` report key); modify
  `tests/test_eval.py`.
- **Decision (reconciliation):** ONE metric, named `span_accuracy` in the report
  (subsuming the ASR track's `span_iou` and the eval track's `citation_accuracy`).
  Definition: for each QA item carrying a gold `span` (a `[t0, t1]`), take the cited
  citation that resolves to the gold-relevant moment and compute interval IoU of its
  `(t_start_s, t_end_s)` (`augur/types.py:11`) vs the gold span; average over items
  that carry a gold span. Items with no gold span are skipped (back-compat).
- **Red (failing test first):** unit tests for the pure `span_iou` fn (exact overlap
  = 1.0, disjoint = 0.0, half-overlap = known value, zero-length guard) — mirrors the
  `TestRetrievalMetrics` style (`tests/test_eval.py:32-72`). Fails: fn does not exist.
- **Green (implement):** add the pure `span_iou` interval-IoU function and the
  collector; thread a `gold_span` field read off `qa.json` items (default `None`).
- **Verify:** unit tests pass; the report now carries `"span_accuracy"`; gates
  unchanged (ungated); `make test` green.
- **Commit:** `feat(eval): unified span/citation-accuracy metric (span_iou) — ungated`

### W3 — gold spans on QA items + word-precise span assert · S
- **Files:** modify `eval/golden/qa.json` (add `gold_span` to each item, esp. the
  word-bearing ones from M0.3); modify `tests/test_eval.py`.
- **Depends on:** M0.3 merged (word-precise spans), else IoU is only cue-granular.
- **Red (failing test first):** `test_span_accuracy_targets_word_precise_spans`:
  assert end-to-end `span_accuracy` over the golden corpus is ≥ a recorded floor
  (informational/ungated) AND that a word-bearing item's cited span is *tighter than*
  the full cue window (proving M0.3's `locate_span` tightening is reflected). Fails
  if spans are still cue-granular or M0.3 is absent.
- **Green (implement):** populate `gold_span` from the word-precise windows; if M0.3
  is not yet merged, land the spans matching cue windows and leave the
  tightness assert `@unittest.skip`-gated on word availability.
- **Verify:** `make test` green; `span_accuracy` reported; still ungated.
- **Commit:** `test(eval): gold spans + word-precise span-accuracy assertion`

### W4 — keyframe-efficiency curve (adaptive vs uniform) · M
- **Files:** modify `eval/harness.py` (add `keyframe_efficiency(...)` over
  `select_keyframes` vs a uniform sampler + `"keyframe_efficiency"` report key);
  modify `tests/test_eval.py`.
- **Design:** reuse `tessera/keyframes.py:select_keyframes`
  (`tessera/keyframes.py:19-39`) for the adaptive count; define a uniform baseline
  (fixed stride) over the same `FrameSig`/`Scene` inputs; report
  `frames_kept_adaptive / frames_kept_uniform` at **equal coverage** (e.g. uniform
  stride chosen so coverage matches adaptive's kept-set, then compare counts). Pure
  arithmetic over deterministic signatures — no media required if M1.1's visual
  fixtures provide signatures; otherwise synthesize deterministic `FrameSig`s in the
  test.
- **Red (failing test first):** `test_keyframe_efficiency_favors_adaptive`: on a
  synthetic scene with a static run + a slide-dense run, assert adaptive keeps
  **fewer** frames than uniform at matched coverage (ratio < 1.0). Fails: function
  absent.
- **Green (implement):** add the curve fn + report key.
- **Verify:** `make test` green; report carries `"keyframe_efficiency"`; ungated.
- **Commit:** `feat(eval): keyframe-efficiency curve (adaptive vs uniform) — ungated`

### W5 — `topic_f1` metric + `eval/golden/topics.json` · M
- **Files:** create `eval/golden/topics.json`; modify `eval/harness.py` (add
  `_topic_clusters(ing, gold_topics)` mirroring `_entity_clusters`,
  `eval/harness.py:344-412`, + `"topic_f1"` report key via the existing
  `clustering_f1`); modify `tests/test_eval.py`, `eval/golden/README.md`.
- **Design:** gold = clusters of logical moment keys that should share a topic; pred
  = group moments by the **persisted** `Moment.topic_id` (read-only, never
  re-run `induce_topics`), then `clustering_f1(pred, gold)` — exactly the
  read-the-persisted-graph regression-guard pattern of `_entity_clusters`. Format
  documented in README alongside the existing label-file docs.
- **Red (failing test first):** `test_topic_f1_present_and_real`: assert the report
  carries `topic_f1` as a `[0,1]` float; plus a unit-style test proving a no-op
  topic induction collapses to singletons → `topic_f1 == 0.0` (the guard property).
  Fails: key/collector absent.
- **Green (implement):** add `topics.json`, the collector, and the report key.
  Replaces the harness comment at `eval/harness.py:646-648` ("topic_f1 is
  deliberately NOT a golden gate") with the ungated-then-gate rationale.
- **Verify:** `make test` green; `topic_f1` reported; **ungated** (2→3 talks is still
  thin — gate later if ≥3 stable items emerge).
- **Commit:** `feat(eval): topic_f1 over golden/topics.json — ungated`

### W6 — §12 claim-granularity curve (FOLD IN) · S
- **Files:** modify `eval/harness.py` (add `_claim_granularity(ing, ...)` +
  `"claim_granularity"` report block); modify `tests/test_eval.py`.
- **Design:** read-only over the persisted store — for each committed claim count
  per moment, report `claims_per_moment` (mean) cross-tabbed against the existing
  `groundedness` and (if available) a salience proxy. Surfaces the lever for the
  M-X.3 extraction-granularity knob; informational only.
- **Red (failing test first):** `test_claim_granularity_block_present`: assert the
  report carries `claim_granularity` with `claims_per_moment >= 0.0` and is
  crash-safe on an empty store. Fails: block absent.
- **Green (implement):** add the block.
- **Verify:** `make test` green; ungated.
- **Commit:** `feat(eval): claim-granularity curve (claims/moment vs groundedness) — ungated`

### W7 — re-baseline contradictions/entities/speakers gold (SINGLE COMMIT, SERIALIZED) · M
- **Files:** modify `eval/golden/contradictions.json`, `eval/golden/entities.json`,
  `eval/golden/speakers.json`, `eval/golden/README.md` — **in ONE commit**.
- **Owns:** this is the single-owner `talk_c` re-baseline. Land it **after** W1–W6,
  **after** verifying numbers (W1 Verify), and **serialize against all other harness
  edits** (the M1.2 risk #7 from the program plan).
- **Red (failing test first):** none new — the existing golden end-to-end tests
  (`TestRunEvalGoldenCorpus`, `tests/test_eval.py:360-422`) become the guard. Before
  this commit `talk_c` introduces an unlabeled video; after it, the
  entity/speaker/contradiction collectors score `talk_c`'s mentions correctly.
- **Green (implement):** add `talk_c`'s entities to `entities.json` (e.g. its new
  unique entity + any shared one), its speakers to `speakers.json` (a new distinct
  identity, plus any cross-talk merge it introduces), and its contradiction pair(s)
  to `contradictions.json`. Update README's shared/distinct narrative.
- **Verify:** `python -m eval.harness` — record final `entity_f1`/`der`/
  `contradiction.f1`/`topic_f1`; confirm the four **existing** gates still pass
  (`--assert-thresholds`); `make test` green. **If `contradiction.f1` dropped below
  0.5 because the `max_claims=600` cap or a spurious cross-pair appeared, fix the
  fixture (not the cap) here.**
- **Commit:** `test(eval): re-baseline contradictions/entities/speakers gold for talk_c`

### W8 — frozen eval-settings snapshot guard · S
- **Files:** modify `eval/harness.py` (export a canonical
  `EVAL_SETTINGS_SNAPSHOT`/serializer over `_FREE_BACKENDS` + the non-default
  `Settings` surface); modify `tests/test_eval.py`.
- **Design:** the snapshot must pin the **full default-OFF / behavioral surface**,
  not just `_FREE_BACKENDS` (`eval/harness.py:62-70`) — include
  `voiceprint_backend`, `visual_enabled`, `top_k`, `rrf_k`,
  `contradiction_threshold`, `keyframe_min_gain`, `keyframe_per_scene_cap`,
  `topic_similarity`, `topic_min_size`, `consensus_jaccard`, `entailment_threshold`,
  `salience_floor` (all from `config.py:27-71`).
- **Red (failing test first):** `test_eval_settings_snapshot_is_frozen`: assert the
  computed snapshot equals a checked-in expected dict. Fails the day any track flips
  a default — forcing a conscious update + a fresh re-baseline.
- **Green (implement):** add the snapshot constant + the equality test.
- **Verify:** `make test` green; test fails loudly on any future default flip.
- **Commit:** `test(eval): frozen eval-settings snapshot guard (beyond _FREE_BACKENDS)`

### W9 — promote `entity_f1` / `der` to gates (SERIALIZED, AFTER VERIFICATION) · S
- **Files:** modify `eval/harness.py` (`_check_thresholds`, gate constants); modify
  `tests/test_eval.py`.
- **Precondition:** W7 merged AND the recorded `entity_f1`/`der` over the 3-talk
  corpus are stable at/above the chosen threshold across two runs (determinism).
- **Red (failing test first):** flip `test_entity_f1_and_der_are_ungated`
  (`tests/test_eval.py:355-357`) into `test_entity_f1_and_der_are_gated`: assert a
  report with `entity_f1`/`der` below threshold now produces a failure. Fails:
  `_check_thresholds` doesn't yet check them.
- **Green (implement):** add `_ENTITY_F1_GATE` / `_DER_GATE` (proposed `0.5`,
  matching the conservative `contradiction.f1` gate; confirm against the recorded
  numbers — they are 1.0 today, so `0.5` is a safe non-flaky floor) and the
  corresponding checks in `_check_thresholds` (`eval/harness.py:656-670`); update the
  `--assert-thresholds` help text (`eval/harness.py:688-692`).
- **Verify:** `python -m eval.harness --assert-thresholds` passes with all six gates;
  `make test` green.
- **Commit:** `feat(eval): promote entity_f1/der to gates after talk_c verification`

## Eval gate

- **The proving gate (graph leg):** the `graph_only` QA item from W1 **misses**
  without the graph leg and **hits** with it — asserted in `tests/test_eval.py`. This
  is the first end-to-end golden guard on the §5 graph-retrieval leg. It is a
  binary correctness assertion (hit/miss), not a thresholded number, so it lands as
  a hard test assertion immediately (it is not subject to the thin-fixture-floor
  rule the way an averaged metric is).
- **Promoted gates (W9), threshold `0.5` each:** `entity_f1 >= 0.5`, `der >= 0.5` —
  added to `--assert-thresholds` only **after** W7 verification confirms `talk_c`
  leaves them ≥ threshold (they are 1.0 today). Conservative `0.5` mirrors the
  existing `_CONTRADICTION_F1_GATE` and avoids flakiness on a 3-talk corpus.
- **Ungated-then-gate (thin-fixture discipline):** `span_accuracy`,
  `keyframe_efficiency`, `topic_f1`, and `claim_granularity` all land **ungated**
  (report keys + informational tests only). Each is gated only once ≥3 stable golden
  items exercise it — explicitly *not* in this track for `topic_f1`
  (3 talks is still thin) unless the recorded numbers prove stable.
- **Existing gates stay green:** `hit_rate >= 0.6`, `groundedness >= 0.8`,
  `contradiction.f1 >= 0.5`, `synthesis.groundedness >= 0.8`
  (`eval/harness.py:642-649`) — verified after the W7 re-baseline; the snapshot guard
  (W8) prevents silent drift.

## Risks & mitigations
- **talk_c perturbs `contradiction.f1` / `entity_f1` / `der` / `topic_f1`
  simultaneously (program risk #7).** Mitigation: serialize the re-baseline (W7) as
  ONE commit owned by this track; record all numbers in W1's Verify *before* the
  re-baseline; only promote gates (W9) after two-run-stable verification.
- **The `graph_only` item silently routes to `hybrid` instead of `contradiction`.**
  The graph leg in `ask()` is gated on a planner keyword (`augur/answer.py:81`,
  `augur/planner.py:32-33`). Mitigation: the W1 question text MUST contain a
  `_CONTRADICTION` trigger word; the test asserts both the on-hit and the off-miss so
  a routing regression is caught.
- **The gold moment leaks into dense/lexical via shared vocabulary**, making the
  "graph-only" claim false. Mitigation: copy the proven `tests/test_graph_retrieval.py`
  design — zero shared query terms between question and gold moment; assert the
  off-miss explicitly.
- **`find_contradictions` `max_claims=600` truncation (`loom/consolidate.py:63-67`)
  could drop a `talk_c` pair on a larger corpus.** Mitigation: at golden scale the
  cap is never hit; the W7 Verify checks `contradiction.f1` did not drop, and if it
  did, fix the fixture rather than the cap (cap is M0.2's concern).
- **Word-precise span dependency (M0.3) not yet merged when W3 runs.** Mitigation:
  W3's tightness assert is skip-guarded on word availability; gold spans land
  cue-granular and tighten automatically once M0.3 merges.
- **Visual fixtures (M1.1) not ready for W4's keyframe curve.** Mitigation: W4's test
  synthesizes deterministic `FrameSig`s so the curve machinery lands independent of
  M1.1; visual-fixture-anchored asserts are deferred to consume M1.1's seam.
- **Snapshot guard becomes a merge-friction nuisance.** Mitigation: that is the
  intent — a failed snapshot test forces a conscious default-flip + re-baseline,
  exactly the determinism-erosion defense (program risk #2).

## Definition of done
- [ ] `eval/golden/talk_c.en.vtt` exists; a `graph_only` QA item misses without the
      graph leg and hits with it (asserted in `tests/test_eval.py`).
- [ ] Unified `span_accuracy` (span_iou) metric in the report; gold spans on
      QA items; word-precise tightness asserted (or skip-guarded pending M0.3).
- [ ] `keyframe_efficiency` curve (adaptive vs uniform) reported, ungated.
- [ ] `topic_f1` + `eval/golden/topics.json` reported, ungated.
- [ ] `claim_granularity` curve reported, ungated.
- [ ] `contradictions.json` / `entities.json` / `speakers.json` re-baselined for
      `talk_c` in ONE serialized commit; the four existing gates still green.
- [ ] Frozen eval-settings snapshot guard added and passing.
- [ ] `entity_f1` / `der` promoted to gates (threshold `0.5`) only after verified
      stable over the 3-talk corpus.
- [ ] `make test` green (count grows from 247); `python -m eval.harness
      --assert-thresholds` passes with all gates.
- [ ] `eval/golden/README.md` updated to document `talk_c`, `topics.json`, and the
      `gold_span` field.

## Open questions
- **`entity_f1`/`der` gate threshold:** proposed `0.5` (conservative, matches
  `contradiction.f1`); they read 1.0 on the 2-talk corpus today. Confirm the
  threshold against the *recorded* 3-talk numbers before W9 — a human should sign off
  that the corpus is "stable enough to gate."
- **Should `topic_f1` be promoted at 3 talks, or held ungated?** The program
  discipline says gate only at ≥3 stable items; 3 talks is borderline. Default in
  this doc: **hold ungated** unless the recorded number is provably stable across
  runs. Confirm.
- **Salience proxy for the claim-granularity curve:** there is no first-class
  salience score persisted today (`salience_floor` exists in `config.py:61` but is a
  filter floor, not a per-claim score). Confirm whether the curve cross-tabs against
  `groundedness` only, or whether a salience proxy is in scope (defaulting to
  groundedness-only to avoid inventing a metric).
- **Metric name in the report:** this doc standardizes on `span_accuracy` (with the
  pure fn named `span_iou`) to unify the two track names. Confirm this is the name
  the M3.4 benchmark harness and any dashboards should consume.
