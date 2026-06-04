# M3.1 — Decay & versioning

> **Wave:** 3 · **Effort:** M · **Status:** not started
> **Depends on:** M0.3 (coordinate the `published_at` part of the `pipeline.ingest` signature) · **Owns (single-owner concerns):** none · **Blocks:** none
> **Spec:** §4.7 (Stage 7 — "Consensus scoring" / "Dedup & decay"), §2 (idempotent + versioned, "Nothing is ever silently deleted")

## Goal
Make recency a first-class, **default-OFF** retrieval signal and make the supersede lifecycle a first-class **read** surface. Concretely: (1) a deterministic recency multiplier on fused retrieval scores, reusing the exact half-life model already in `consensus.py`, with `reference_date` = the corpus-newest `published_at`; (2) demotion of moments whose committed claims are *all* superseded; (3) a `store.claim_history(claim_id)` read that walks the existing `superseded_by` lineage (nothing deleted); (4) a `timeline` surface across SDK/CLI/REST/MCP that reuses `loom/evolution.claim_evolution`; and (5) `published_at` injection for local sources so decay and the existing temporal/evolution stories actually fire on local files and the golden corpus. The default-OFF run must be **byte-identical** to today.

## Why it matters
The spec's Phase-4 bullet names "decay/versioning" and §4.7 promises "optionally down-weight stale facts" and "claim-evolution tracking … to trace how a position or a number changes over time." Today the recency model exists but is *only* used inside consensus scoring; retrieval ranking ignores publish date entirely, superseded claims are written but never demoted at query time, and the lineage is one-directional (`superseded_by`) with no read API. Most damaging: **local files (including the entire golden corpus) carry no `published_at`** (`_acquire_local` never sets it — `acquire.py:76`, `acquire.py:88`), so every temporal/evolution/decay capability silently no-ops on exactly the corpus we evaluate against. This track turns "Source A (2024) asserts X; Source B (2026) asserts ¬X" and "recent-first answers" into things a user can actually see and that the harness can prove.

## Scope (reconciled)
In scope:
- **Recency multiplier (default OFF).** A deterministic post-fusion re-weight of `retrieve()` scores, reusing the `0.5 ** (age_days / halflife)` half-life model from `consensus._recency_term` (`consensus.py:73-82`), with `reference_date` = newest `published_at` across the corpus. Gated behind a new `Settings.decay_enabled: bool = False` (and `decay_halflife_days`). Off ⇒ identity ⇒ byte-identical RRF output.
- **Superseded-moment demotion (default OFF, same flag).** A moment whose committed claims are *all* superseded (or which has claims but none committed) is demoted/excluded from results. Reuses `store.claims_for_moment(..., status=...)`.
- **`store.claim_history(claim_id)`** — read the supersede lineage (the chain of `superseded_by` pointers, plus inbound "superseded-by-me" claims), returning all versions in order, **nothing deleted** (§2). The write side already exists: `supersede_claim` (`store.py:371-381`).
- **`timeline` surface** across SDK/CLI/REST/MCP, **reusing** `loom/evolution.claim_evolution` (`evolution.py:87-137`) and its `EvolutionStep.to_dict()` — this is the user-facing "how did this change over time" read. (SDK `evolution()` exists at `sdk.py:94-103` and CLI `evolution` at `cli.py:123-139`; REST and MCP do **not** expose it yet — `rest.py` do_GET has no evolution route, `mcp.py` TOOLS has no evolution tool.)
- **`published_at` injection for local sources** — thread a `published_at=` keyword through the M0.3-owned `pipeline.ingest` signature down into `acquire`/`SourceMeta` so local files (and the dated golden variant) carry a publish date. **FOLD IN per reconciliation #7:** `published_at=` is added as a *keyword-only* argument to the `pipeline.ingest` signature **owned by M0.3**; this track must coordinate that one parameter with M0.3, not re-cut the signature independently.

FOLDED IN from the completeness review:
- The half-life model is **reused**, not re-derived (M3.1 must not fork a second decay constant; it consumes `consensus`'s `_RECENCY_HALFLIFE_DAYS` / its public form).
- The `published_at` injection is part of the **single coordinated `pipeline.ingest` signature change** owned by M0.3 (reconciliation #7), layered as a keyword-only addition.
- The new `decay_*` flags must be added to the **frozen eval-settings snapshot** (global discipline (b)), not just `_FREE_BACKENDS`, so a future default flip can't silently move gate numbers.
- The new `decay` metric lands **UNGATED first**, gated only once ≥3 stable golden items exist (global discipline (a)).

NON-GOALS / deferrals:
- No new ranking *model* — recency is pure arithmetic on existing scores; no learned recency, no ANN.
- No deletion or hard-expiry of stale facts — "decay" means *down-weight*, never delete (§2). Superseded claims remain fully fetchable.
- No change to the consensus scoring weights or the half-life constant value (reused as-is).
- No automatic flipping of `decay_enabled` to ON by default in this track (would not be byte-identical; deferred to a future default-flip once the gate is stable).
- Authority/source-count terms of consensus are *not* added to retrieval ranking here — only the recency multiplier.

## Current state (grounded)
What exists and is reused:
- **Half-life recency model** — `consensus._recency_term` (`consensus.py:73-82`): `0.5 ** (age_days / _RECENCY_HALFLIFE_DAYS)`, with `_RECENCY_HALFLIFE_DAYS = 365.0` (`consensus.py:34`), neutral `0.5` when dates are missing, `reference_date` chosen as corpus-newest in `clusters_from_groups` (`consensus.py:194-195`). This is the exact model M3.1 reuses for retrieval.
- **Supersede write side** — `store.supersede_claim(old_id, new_id)` sets `status='superseded'`, `superseded_by=new_id`, never deletes (`store.py:371-381`); `Claim.superseded_by` field exists (`models.py:105`); schema column exists (`store.py:75`); `STATUS_SUPERSEDED` constant (`models.py:34`). The live caller is `dedup_claims` (`consolidate.py:144-188`).
- **Status-aware claim reads** — `claims_for_moment(..., status=...)` (`store.py:360-369`) and `list_claims(status=...)` (`store.py:383-392`) already let callers ask for committed-only vs all. `get_claim` always returns even superseded (`store.py:337-339`).
- **Evolution read** — `claim_evolution` already orders an entity/topic's claims by `published_at` and flags `superseded` (`evolution.py:118-135`); surfaced via SDK `evolution()` (`sdk.py:94-103`) and CLI `evolution` (`cli.py:123-139`).
- **Fusion entry point** — `retrieve()` returns fused `(moment_id, rrf_score)` from `rrf_fuse` (`retrieve.py:30-65`); `answer.ask` consumes it (`answer.py:87-90`). `answer.ask` already does a `temporal`-strategy chronological re-sort using `video.published_at` (`answer.py:127-139`), so the publish-date plumbing into answers partly exists.

What is missing / partial:
- **`published_at` is never set for local sources.** `_acquire_local` builds `SourceMeta` without `published_at` (`acquire.py:76-83`, `acquire.py:88-96`); only `_acquire_url` sets it from yt-dlp's `upload_date` (`acquire.py:151`). The golden corpus is ingested via `mv.ingest(str(vtt))` (`harness.py:236`) with no date ⇒ every recency/temporal path computes the neutral `0.5` and every `evolution`/temporal sort degenerates to the `t_start_s`/`claim_id` tiebreak.
- **Retrieval ignores recency entirely** — `retrieve()`/`rrf_fuse` have no date input (`retrieve.py:18-65`).
- **No query-time supersede demotion** — superseded claims drop out of *committed* claim queries but a moment with only superseded claims is still indexed and retrievable.
- **No `claim_history` read API** — the `superseded_by` chain can only be walked by hand via repeated `get_claim`.
- **No `timeline`/evolution surface on REST or MCP** — `rest.py` do_GET routes (`rest.py:53-71`) and `mcp.py` `TOOLS` (`mcp.py:23-83`) have no evolution/timeline entry; only SDK + CLI do.
- **No `decay_*` settings** — `Settings` (`config.py:27-71`) has consensus/retrieval knobs but nothing for decay.
- **No `decay` eval block** — the harness report (`harness.py:587-631`) and gates (`harness.py:642-670`) have no decay metric; the golden corpus has no dated variant (`eval/golden/` holds undated `talk_a.en.vtt`, `talk_b.en.vtt`).

## Free-path guarantee
The stdlib-only deterministic default stays intact by construction:
- **`decay_enabled` defaults to `False`.** When off, `retrieve()` returns exactly today's `rrf_fuse` output (same ids, same float scores, same order) and no demotion is applied. This is the byte-identical contract; W1's Red test asserts it directly against a captured baseline.
- The recency multiplier reuses the **existing** pure-Python half-life arithmetic (`0.5 ** (age/halflife)`) and `util.parse_iso` (`util.py:26-44`) — no new dependency, fully deterministic.
- `published_at` injection only *adds* a value when the caller passes one (or a sidecar provides one); local ingest with no date keeps `published_at=None` exactly as today, so the free VTT corpus output is unchanged unless a test/harness explicitly opts into the dated variant.
- `claim_history` and the `timeline` surface are pure reads over already-persisted rows — no write, no schema change beyond what exists, no model needed.
- **Frozen eval-settings snapshot:** `decay_enabled=False`, `decay_halflife_days=365.0` are added to the pinned snapshot the harness asserts (alongside `_FREE_BACKENDS`), so the growing surface of default-OFF flags can't silently move gate numbers (review risk #2 / discipline (b)).
- **Provenance is sacred:** demotion only *reorders/filters* retrieval candidates; it never edits a claim's stored span or status, and the displayed citation span is unchanged. Nothing is deleted (§2).

## Workstreams
Ordered, TDD-sized, each independently committable. Build W1 first (the byte-identical scaffolding + flag), then the surfaces.

### W1 — Recency multiplier + decay flags (default OFF, byte-identical) · M
- **Files:** `src/memovox/config.py` (add `decay_enabled: bool = False`, `decay_halflife_days: float = 365.0`); `src/memovox/augur/retrieve.py` (add `reference_date`/decay-aware re-weight, default no-op); `src/memovox/loom/consensus.py` (expose a reusable `recency_multiplier(age_days, halflife)` or `recency_weight(published_at, reference_date, halflife)` extracted from `_recency_term`, so retrieval reuses it without forking the constant); `tests/test_decay.py` (new).
- **Red (failing test first):** `tests/test_decay.py::test_decay_off_is_byte_identical` ingests two dated moments, captures `retrieve(...)` output with `decay_enabled=False`, and asserts it equals the pre-change baseline (fails only if the refactor isn't a true no-op); `test_decay_on_reweights_recent_first` builds two moments tied on RRF but with different `published_at` and asserts the newer one ranks first **only** when `decay_enabled=True`. Fails today because `retrieve()` has no recency input and no flag.
- **Green (implement):** Extract the half-life math from `consensus._recency_term` into a small public helper (consumed by both consensus and retrieve — no second constant). In `retrieve()`, when `settings.decay_enabled`, compute `reference_date` = max `published_at` over the candidate moments' videos, multiply each fused score by `recency_weight(...)` (neutral `0.5`→treat as `1.0` multiplier when dates absent, or skip — choose the variant that keeps off==on for an all-undated corpus), then re-sort. Off ⇒ return early with today's path.
- **Verify:** `make test` (new `test_decay` passes; all existing green); `python -m eval.harness` report unchanged with default settings.
- **Commit:** `feat(augur): default-off recency decay multiplier on fused scores (spec §4.7)`

### W2 — Superseded-moment demotion + `store.claim_history` · M
- **Files:** `src/memovox/loom/store.py` (add `claim_history(claim_id) -> List[Claim]` walking `superseded_by` forward + inbound supersedes); `src/memovox/augur/retrieve.py` (demote/exclude all-superseded moments when `decay_enabled`); `tests/test_decay.py` (extend).
- **Red (failing test first):** `test_claim_history_preserves_all_versions` supersedes claim A→B→C via `supersede_claim`, asserts `claim_history(A_id)` returns all three in lineage order with statuses intact and nothing deleted (fails: no such method). `test_superseded_only_moment_demoted` creates a moment whose every claim is superseded and asserts it is absent from `retrieve()` results when `decay_enabled=True` and present when `False` (fails: no demotion).
- **Green (implement):** `claim_history` chases `superseded_by` from the given claim and also collects claims pointing at it (so a mid-chain id yields the full version set), deterministically ordered. In `retrieve()` (decay path only) drop moment ids whose `claims_for_moment(mid, status='committed')` is empty *and* which have at least one superseded claim — i.e. fully-superseded moments — before re-sort.
- **Verify:** `make test`; confirm `eval.harness` default report still byte-identical (demotion only fires under `decay_enabled`).
- **Commit:** `feat(loom): claim_history lineage read + decay-time supersede demotion (spec §4.7/§2)`

### W3 — `published_at` injection for local sources (coordinate M0.3 signature) · S
- **Files:** `src/memovox/stentor/acquire.py` (`_acquire_local` accepts + sets `published_at`); `src/memovox/stentor/__init__.py` (thread `published_at` through `run`); `src/memovox/pipeline.py` (add **keyword-only** `published_at=` to `ingest`, coordinated with M0.3's signature edit — see Open questions); optional sidecar: read `<file>.meta.json`'s `published_at` if present; `tests/test_decay.py` / `tests/test_stentor.py` (extend).
- **Red (failing test first):** `test_local_ingest_threads_published_at` ingests a local VTT with `published_at="2026-01-01"` and asserts `store.get_video(vid).published_at == "2026-01-01"` (fails: `_acquire_local` never sets it, `ingest` has no such kwarg). `test_local_ingest_default_published_at_unchanged` ingests with no date and asserts `published_at is None` (free-path unchanged).
- **Green (implement):** Add `published_at` to `SourceMeta` usage in `_acquire_local`, thread `published_at` through `stentor.run` and `pipeline.ingest` (as a keyword-only param appended to the M0.3 signature). `make_video_id`/`is_unchanged` must remain stable — only set the field on the `Video`. (Coordinate with M0.3 so the param is added once.)
- **Verify:** `make test`; default (no-date) golden ingest still produces `published_at=None` ⇒ harness default report unchanged.
- **Commit:** `feat(stentor): published_at injection for local sources (coordinate M0.3 ingest signature)`

### W4 — `timeline` surface on SDK/CLI/REST/MCP (reuse evolution) · S
- **Files:** `src/memovox/sdk.py` (add `timeline(...)` thin wrapper over `claim_evolution`, or document `evolution()` as the timeline read and add `claim_history` passthrough); `src/memovox/server/rest.py` (add `GET /timeline?entity=|topic=` and optionally `GET /claim/{id}/history` to do_GET; register in `/` endpoints list); `src/memovox/server/mcp.py` (add a `claim_timeline` tool to `TOOLS` + `_tool_claim_timeline`); `tests/test_decay.py` / `tests/test_mcp.py` / extend REST tests.
- **Red (failing test first):** `test_rest_timeline_endpoint` hits `GET /timeline?topic=...` and asserts an ordered list of `EvolutionStep.to_dict()` dicts (fails: route 404s today). `test_mcp_timeline_tool` lists tools and asserts `claim_timeline` is present and returns ordered steps (fails: not in `TOOLS`). Both reuse `claim_evolution`, asserting no new ordering logic is introduced.
- **Green (implement):** REST `do_GET` branch calls `mv.evolution(entity=, topic=)` and `_send`s the list; MCP tool wraps the same. SDK already exposes `evolution()`; add `claim_history(claim_id)` passthrough for parity with W2. Keep all surfaces reusing `loom/evolution` — no duplicated sort.
- **Verify:** `make test`; logging stays on stderr (MCP stdout = JSON-RPC only — review risk #5).
- **Commit:** `feat(server): expose claim timeline + history on REST/MCP (reuse loom/evolution)`

### W5 — `decay` eval block on a dated golden variant (ungated → gated) · M
- **Files:** `eval/golden/` (add a dated variant — e.g. `talk_a`/`talk_b` get `published_at` via a sidecar or a small dated 3rd item; do **not** perturb the existing undated gates — see risk below); `eval/harness.py` (add `_decay_metrics` + a `decay` block to the report; add `decay_enabled`/`decay_halflife_days` to the **frozen eval-settings snapshot**); `tests/test_eval.py` (extend).
- **Red (failing test first):** `tests/test_eval.py::test_decay_block_present` asserts the report has a `decay` block with `recent_first_ordering` (does the newer-source answer outrank the stale one when `decay_enabled`?) and `superseded_excluded` (a fully-superseded moment is absent). Fails: no decay block.
- **Green (implement):** A dedicated dated harness pass (separate `Memovox` configured with `decay_enabled=True` + dated ingest) computes the two decay sub-metrics; emit them in the report. Land **UNGATED** (discipline (a)). Add the `decay_*` flags to the pinned snapshot the harness asserts so default-OFF flag drift is caught.
- **Verify:** `python -m eval.harness` shows `decay` block; `python -m eval.harness --assert-thresholds` still passes (existing four gates green, decay not yet gated); `make test`.
- **Commit:** `feat(eval): ungated decay block (recent-first ordering, superseded excluded)` — and, once ≥3 stable dated items exist, a follow-up `feat(eval): gate decay block` flipping it into `_check_thresholds`.

## Eval gate
A new **`decay`** block in the harness report (`harness.py` `_compute_report`), measured on a **dated golden variant** with `decay_enabled=True`:
- `recent_first_ordering` — the answer to a recency-sensitive question ranks the newer source's moment above the older one.
- `superseded_excluded` — a moment whose claims are all superseded does not appear in retrieval/answer results.

Per global discipline (a) it lands **UNGATED first**, and is promoted to a gate (proposed threshold: both sub-metrics `== 1.0`, i.e. exact recent-first ordering + full superseded exclusion on the dated fixture) **only once ≥3 stable dated golden items exist** — a 2-item dated corpus is too thin to hard-gate (review risk #1). The frozen eval-settings snapshot is extended to pin `decay_enabled=False` / `decay_halflife_days=365.0` (discipline (b)).

**Existing gates must stay green:** with default settings the harness report is byte-identical to today — `retrieval.hit_rate ≥ 0.6`, `groundedness ≥ 0.8`, `contradiction.f1 ≥ 0.5`, `synthesis.groundedness ≥ 0.8` (`harness.py:642-670`) — because `decay_enabled=False` makes the recency multiplier and demotion no-ops and the default (undated) golden ingest leaves `published_at=None`.

## Risks & mitigations
- **Refactor breaks byte-identity (the dominant risk for this track).** Extracting the half-life helper and adding the decay branch must leave `decay_enabled=False` exactly equal to today. *Mitigation:* W1 Red test captures the current `retrieve()` output as a golden baseline and asserts equality; run `python -m eval.harness` before/after and diff the report.
- **Decay vs the consensus model forking constants** (review fold-in). *Mitigation:* extract `_recency_term`'s math into one shared helper consumed by both `consensus` and `retrieve`; never copy `_RECENCY_HALFLIFE_DAYS`.
- **Determinism erosion from new default-OFF flags** (review risk #2). *Mitigation:* add `decay_enabled`/`decay_halflife_days` to the frozen eval-settings snapshot, not just `_FREE_BACKENDS`.
- **Eval thinness / gate flakiness on a 2-item dated corpus** (review risk #1). *Mitigation:* land the `decay` block ungated; gate only after ≥3 stable dated golden items.
- **`talk_c` re-baseline blast radius** (review risk #7) — adding a dated 3rd video can perturb several existing metrics at once. *Mitigation:* prefer adding `published_at` to the *existing* talk_a/talk_b via a sidecar (which is a no-op for the undated default gates) over introducing a new video in this track; if a new dated item is needed, serialize that commit against M1.2's `talk_c` re-baseline and do not stack other harness edits in it.
- **`pipeline.ingest` signature collision with M0.3** (reconciliation #7). *Mitigation:* `published_at=` is a keyword-only addition coordinated with M0.3, which **owns** the signature; this track adds only that one param and rebases on M0.3's landing.
- **MCP stdout discipline** (review risk #5) — the new MCP timeline tool must log only to stderr; the JSON-RPC channel on stdout stays clean. *Mitigation:* reuse the existing `_tool_json`/`_send` plumbing, no stray `print`.
- **Temporal-answer interaction** — `answer.ask` already re-sorts on `published_at` for the `temporal` strategy (`answer.py:127-139`). Decay re-weighting at the `retrieve()` layer must compose with, not double-count, that re-sort. *Mitigation:* apply decay in `retrieve()` (candidate scoring) and leave the temporal answer re-sort as-is; assert in a test that a `temporal`-strategy answer with `decay_enabled=False` is byte-identical.

## Definition of done
- [ ] `Settings.decay_enabled=False` / `decay_halflife_days=365.0` added; default run byte-identical (W1 baseline test + harness report diff clean).
- [ ] Recency multiplier in `retrieve()` reuses the single shared half-life helper; recent-first ordering verified under `decay_enabled=True`.
- [ ] Fully-superseded moments demoted/excluded under `decay_enabled`; `store.claim_history` returns all versions, nothing deleted.
- [ ] `published_at` threads through `_acquire_local` → `stentor.run` → `pipeline.ingest` (keyword-only, coordinated with M0.3); default no-date ingest still yields `published_at=None`.
- [ ] `timeline` (+ `claim_history`) exposed on REST and MCP, reusing `loom/evolution`; SDK/CLI parity confirmed.
- [ ] `decay` block in the harness report (ungated), flags pinned in the frozen eval-settings snapshot.
- [ ] `make test` green; `python -m eval.harness --assert-thresholds` green (existing four gates unchanged).

## Open questions
- **M0.3 signature coordination:** does M0.3 land the full keyword-only superset (`published_at=`, `visual_result=`, `modality=`, ASR device knobs) in one commit, with M3.1 only *using* `published_at=`? Confirm the merge order so W3 rebases on M0.3 rather than editing the signature first (reconciliation #7).
- **Dated golden source:** add `published_at` to existing `talk_a`/`talk_b` via a `.meta.json` sidecar (lowest blast radius), or introduce a dated 3rd item? The sidecar keeps the existing undated gates untouched and is preferred; confirm a sidecar convention (e.g. `talk_a.meta.json` next to `talk_a.en.vtt`) is acceptable.
- **Recency multiplier shape:** multiply RRF scores by the raw `0.5**(age/halflife)` recency weight, or by a blended `(1 - w_rec) + w_rec * recency` so old-but-relevant items aren't crushed? Default-OFF makes this safe to tune, but the gated threshold depends on it — confirm the intended aggressiveness.
- **Undated-corpus behavior under `decay_enabled=True`:** when *all* sources are undated, should decay be a strict no-op (recommended, since `_recency_term` returns neutral `0.5` uniformly) or should it still apply the uniform `0.5`? The former keeps on==off for undated corpora; confirm.
- **Demotion = exclude vs down-rank:** should a fully-superseded moment be dropped entirely or just pushed below committed-bearing moments? Spec §2 forbids deletion of the *stored* claim, not its appearance in a ranked list — confirm exclude-from-results is acceptable.
