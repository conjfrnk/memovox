# M3.2 — Subscriptions & incremental sync

> **Wave:** 3 · **Effort:** L · **Status:** ✅ done (branch `phase4-subscriptions`, 7/7 + review fix; 449 pass / 2 skip; 14 gates). enumerate_source (no download) + sync_state cursor + resolve_corpus deferral + SyncReport engine + subscribe/unsubscribe CLI + incremental==batch gate.
> **Depends on:** M0.2 (incremental consolidation — *consume, do not reimplement*) · **Owns (single-owner concerns):** the sync rewrite (Serving / M3.3 consumes this) · **Blocks:** M3.3
> **Spec:** §8 (CLI: `memovox ingest <url | playlist | channel>`, `memovox sync`), §11 (Phase 4: "channel/playlist subscriptions + incremental sync")

## Goal
Turn `memovox sync` from a flat one-URL-per-entry re-ingest loop into a real subscription engine. A new `acquire.enumerate_source` expands a channel/playlist URL into its video ids via `yt-dlp --flat-playlist` (metadata only — **never** downloads). A persisted `sync_state` cursor records the video ids already seen per source so the second-and-later passes **skip seen ids before any download happens**. Each entry is ingested under its own try/except so one failure never aborts the batch. After the batch, sync runs **one** incremental consolidation through M0.2 (per-video ingest gets a `resolve_corpus=False` flag so the whole-corpus resolve happens once for the batch instead of once per video). `subscribe` / `subscriptions` / `unsubscribe` CLI commands manage the source list. When `yt-dlp` is absent the free/local path behaves **exactly** as today.

## Why it matters
Spec §11 names "channel/playlist subscriptions + incremental sync" as a literal Phase-4 deliverable and the differentiator that makes memovox a "production-grade library" rather than a one-shot ingest tool. The user-visible capability: `memovox subscribe <channel-url>` then a periodic `memovox sync` that pulls only *new* uploads, ingests them, and re-consolidates the corpus once — cheaply and idempotently (a second `sync` with no new uploads does zero work). Today `mv.sync()` (`src/memovox/sdk.py:43`) re-ingests every entry in `subscriptions.json` on every call and treats a playlist/channel URL as a single video (yt-dlp runs with `--no-playlist`, `src/memovox/stentor/acquire.py:127`), so it neither expands sources nor skips seen videos — it is a stub, not a subscription engine.

## Scope (reconciled)

In scope:
- **`acquire.enumerate_source(config, url) -> list[EnumeratedEntry]`** — `yt-dlp --flat-playlist --dump-single-json` (or `--print id,...`), metadata-only, **never downloads media**. Returns `(video_id, url, title?)` per child entry. A bare video URL enumerates to a single entry (so `sync` treats `ingest`-style URLs uniformly). Behind `shutil.which("yt-dlp")`; clear `AcquisitionError` when absent, mirroring `_acquire_url` (`acquire.py:116`).
- **Persisted `sync_state` cursor** — per-source set of already-seen video ids, stored in the existing `meta` table (`src/memovox/loom/store.py:37`, keyed e.g. `sync_state:<source_key>`) via `set_meta`/`get_meta` (`store.py:169`/`:175`). The cursor is consulted to **skip seen ids before download** — the whole point is to never invoke the per-video download/ASR path for a video already in the store.
- **Rewritten `Memovox.sync()`** (`sdk.py:43`) — for each subscribed source: enumerate → diff against cursor → for each unseen id, `ingest(url, resolve_corpus=False)` under per-entry try/except → record the id in the cursor on success → after the loop, run **one** incremental consolidation via M0.2. Returns a structured `SyncReport` (per-entry status + skipped count) so M3.3 (Serving) can drive it as a job and the CLI can print it.
- **`pipeline.ingest(..., resolve_corpus: bool = True)`** — a keyword-only flag (default `True` = today's behavior) that, when `False`, skips the *whole-corpus* `resolve_entities` / `resolve_speakers` / `link_claim_relations` passes (`pipeline.py:201`, `:224`, `:230`) on the per-video path; the batch's single incremental consolidate does the corpus-wide resolve once. **Default-True keeps single-video `ingest` byte-identical.**
- **CLI:** `memovox subscribe <url>`, `memovox subscriptions` (list), `memovox unsubscribe <url>` — edit `subscriptions.json` (`config.subscriptions_path`, `config.py:146`). `cmd_sync` (`cli.py:70`) reworked to print the structured `SyncReport` (new / skipped-seen / failed counts).
- **Eval:** a new `incremental` block in `eval/harness.py` asserting (a) **incremental == batch equivalence** (a corpus ingested incrementally-then-consolidated-once produces the same gated report as a full batch consolidate) and (b) **idempotent re-sync** (a 2nd `sync` over the same sources ingests 0 / all skipped-seen). All yt-dlp interaction behind `shutil.which` and **monkeypatched in tests** — `make test` does no network I/O.

Folded in from the completeness review:
- **`resolve_corpus=False` ingest flag is owned here** (planning fix #7 coordinates the `pipeline.ingest` signature in M0.3; this track adds `resolve_corpus` as a keyword-only addition, layered the same way).
- **M-X.2 fold-in:** ffprobe ASR-readiness gate and captions-as-fused-prior are *folded into M0.3/M3.2*; for this track that means enumeration must tolerate entries with no media (caption-only / unavailable) by isolating the failure per-entry, not by pre-validating media here.

Non-goals / deferrals:
- **No incremental-consolidation reimplementation.** The watermark / new-claims-vs-all scan / paging lives **once** in M0.2. This track *calls* it. If M0.2 is not yet merged, sync calls the existing full `consolidate()` (`sdk.py:78`) once per batch as the interim — still correct, just not yet incremental — and swaps to the M0.2 incremental entry point when it lands (see Open questions).
- **No background/async job runner, no scheduling, no cron.** That is M3.3 (Serving), which *consumes* this sync rewrite. `sync` here is synchronous and deterministic.
- **No new ranking, retrieval, or answer behavior.** Sync only changes *what gets ingested and when consolidation runs*.
- **No yt-dlp download of playlists.** Enumeration is `--flat-playlist` metadata only; actual download stays the per-video `_acquire_url` path with `--no-playlist`.

## Current state (grounded)
- `Memovox.sync()` — `src/memovox/sdk.py:43-59`. Reads `subscriptions.json`, and for each `entry` calls `self.ingest(url)` unconditionally. No enumeration, no cursor, no per-entry isolation (an `ingest` raising propagates and aborts the whole loop), no post-batch consolidation. This is the single owner of the sync path and the thing being rewritten.
- `cmd_sync` — `src/memovox/cli.py:70-77`. Prints one line per report; "No subscriptions found" when empty. There is **no** `subscribe`/`subscriptions`/`unsubscribe` command — `subscriptions.json` is hand-edited today (documented in `config.py:12`).
- `acquire()` / `_acquire_url()` — `src/memovox/stentor/acquire.py:45`, `:113`. URL download runs `yt-dlp` with `--no-playlist` (`acquire.py:127`) behind `shutil.which("yt-dlp")` (`acquire.py:116`), writing `info.json` + subs. There is **no** enumeration entry point. `SourceMeta` (`acquire.py:26`) is the per-video result; enumeration needs a new lightweight return type.
- `pipeline.ingest()` — `src/memovox/pipeline.py:90-103`. Signature is keyword-only after `source`; the whole-corpus resolve passes run **unconditionally** at `pipeline.py:201` (`resolve_entities`), `:224` (`resolve_speakers` — explicitly "Re-resolves the WHOLE corpus each ingest"), `:230` (`link_claim_relations`). There is no flag to skip them, so N-video sync today does N whole-corpus resolves.
- `store.is_unchanged(video)` — `src/memovox/loom/store.py:189-195`. Already gives idempotency *after* download (content_hash + pipeline_version match → `ingest` returns `unchanged`, `pipeline.py:130`). But it only fires *after* acquire/ASR have run; the cursor's job is to skip *before* download so re-sync is cheap.
- `store.set_meta` / `get_meta` — `src/memovox/loom/store.py:169`/`:175`, backed by `CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)` (`store.py:37`). This is the persistence slot for the cursor — no schema change needed.
- `config.subscriptions_path` — `src/memovox/config.py:146-147` → `<store>/subscriptions.json`. Exists; the format is `{"sources": [ {"url": ...} | "<url>" ]}` per `sdk.py:55-57`.
- `make_video_id` / `youtube_id` — `src/memovox/util.py:110` and `:` (youtube_id above it). YouTube URLs map to a stable `yt:<id>`; this is the natural cursor key for a flat-playlist entry whose id `yt-dlp` already reports, so the cursor can be populated *from enumeration metadata* without downloading.
- Eval harness — `eval/harness.py`. `run_eval` (`:552`) ingests the golden corpus via `_ingest_golden` (`:229`) with `_FREE_BACKENDS` pinned (`:62`), computes `_compute_report` (`:589`), and `_check_thresholds` (`:656`) gates `retrieval.hit_rate>=0.6`, `groundedness>=0.8`, `contradiction.f1>=0.5`, `synthesis.groundedness>=0.8` (`:642-649`). There is **no** `incremental` block yet; that is this track's new ungated-then-gated metric.
- No `tests/test_sync.py`, `test_subscriptions.py`, or `test_acquire.py` exist (confirmed: `tests/` has none). The only existing monkeypatch pattern for the visual/network-touching path is `mock.patch("memovox.tessera.run", ...)` in `tests/test_integration.py:84,100` — the same pattern (patch the module attribute) applies to `enumerate_source` / `yt-dlp`.

## Free-path guarantee
- **`yt-dlp` absent (the free/stdlib default):** `enumerate_source` raises the same `AcquisitionError`-with-install-hint shape as `_acquire_url` (`acquire.py:116-121`); `sync` over a local/transcript-only `subscriptions.json` still works because a bare local entry enumerates to itself. `make test` and the eval harness never have `yt-dlp` on PATH and never hit the network — all enumeration is monkeypatched.
- **`resolve_corpus` defaults to `True`** → a single `Memovox.ingest(url)` is **byte-identical** to today (same resolve passes at `pipeline.py:201/224/230`). The skip only engages when `sync` explicitly passes `resolve_corpus=False` and then runs the corpus resolve once via consolidation.
- **Consolidation ownership:** sync never reimplements the watermark/scan — it calls M0.2's incremental entry point (or, interim, the existing `consolidate()`), so the free deterministic consolidation behavior is unchanged byte-for-byte; sync only changes *how often* it runs (once per batch instead of never).
- **Frozen eval-settings snapshot:** the `resolve_corpus` default and any new sync default flags are added to the pinned snapshot the harness asserts (the discipline that pins more than `_FREE_BACKENDS`), so a future default flip cannot silently move gate numbers.
- **What must stay byte-identical:** (1) single-video `ingest` output and digests; (2) the existing eval report values (all four gates green, unchanged); (3) `sync` over a local-only `subscriptions.json` with no yt-dlp.

## Workstreams

### W1 — `enumerate_source` (flat-playlist expansion, never downloads) · M
- **Files:** `src/memovox/stentor/acquire.py` (add `EnumeratedEntry` dataclass + `enumerate_source`), `src/memovox/stentor/__init__.py` (re-export), `tests/test_acquire.py` (new).
- **Red (failing test first):** `tests/test_acquire.py` monkeypatches `subprocess.run` (and `shutil.which("yt-dlp")` → a path) to return a canned `--flat-playlist --dump-single-json` payload with 3 `entries`; asserts `enumerate_source` returns 3 `EnumeratedEntry(video_id, url, title)` and that the `yt-dlp` argv contains `--flat-playlist` and does **not** contain `-f`/`bestaudio`/`-o` (proves no download). A second test: `shutil.which` → `None` raises `AcquisitionError` with the install hint. A third: a bare `youtu.be/<id>` URL enumerates to exactly one entry. Fails today: `enumerate_source` does not exist.
- **Green (implement):** add `EnumeratedEntry`; `enumerate_source(config, url)` shells `yt-dlp --flat-playlist --dump-single-json --no-warnings <url>`, parses `entries` (falling back to a single-video shape), derives `video_id` via `make_video_id`/the reported `id`. No media dir writes. Reuse the `shutil.which` guard and error text from `_acquire_url`.
- **Verify:** `make test` (new tests pass; no network — `subprocess.run` patched).
- **Commit:** `feat(stentor): enumerate_source flat-playlist expansion (metadata-only, no download)`

### W2 — Persisted `sync_state` cursor · S
- **Files:** `src/memovox/stentor/acquire.py` or a small `src/memovox/sync_state.py` (cursor read/write helpers), `tests/test_sync.py` (new).
- **Red (failing test first):** `tests/test_sync.py` opens a `LoomStore`, writes a cursor of `{src_key: [id1, id2]}` via the helper, reopens the store, reads it back, asserts equality; asserts an unknown source returns an empty set; asserts `mark_seen(src, id3)` is additive and idempotent (marking `id1` again is a no-op). Fails today: no cursor helper exists.
- **Green (implement):** thin helpers over `store.set_meta`/`get_meta` (`store.py:169/175`) using key `sync_state:<source_key>` and a JSON-encoded sorted id list. `source_key` is a stable hash of the source URL (reuse `util.short_hash`/`sha1_hex`). Deterministic serialization (sorted) so the stored value is stable across runs.
- **Verify:** `make test`.
- **Commit:** `feat(sync): persisted per-source sync_state cursor in meta table`

### W3 — `resolve_corpus=False` ingest flag · S
- **Files:** `src/memovox/pipeline.py` (add keyword-only `resolve_corpus: bool = True`; guard the three corpus passes), `tests/test_integration.py` (extend).
- **Red (failing test first):** ingest two contradicting VTTs with `resolve_corpus=False`, assert the per-video claims still commit but the cross-corpus CONTRADICTS edge is **absent** until a consolidate runs; then a separate test asserts `ingest(..., resolve_corpus=True)` (default) is unchanged from today (same committed counts, same edges, same digest). Fails today: `ingest` has no `resolve_corpus` parameter (`pipeline.py:90-103`).
- **Green (implement):** add `resolve_corpus: bool = True` (keyword-only). Wrap `resolve_entities` (`pipeline.py:201`), `resolve_speakers` (`:224`), and `link_claim_relations` (`:230`) in `if resolve_corpus:`. Note these are the *whole-corpus* passes; the per-video moment/claim/edge writes above stay unconditional. Default `True` → byte-identical.
- **Verify:** `make test` + `python -m eval.harness --assert-thresholds` (all four gates still green, since the harness ingests with the default `resolve_corpus=True`).
- **Commit:** `feat(pipeline): resolve_corpus flag to defer whole-corpus resolve (default on)`

### W4 — `Memovox.sync()` rewrite (enumerate → cursor-skip → isolate → one consolidate) · M
- **Files:** `src/memovox/sdk.py` (rewrite `sync`, add `SyncReport`), `tests/test_sync.py` (extend).
- **Red (failing test first):** monkeypatch `acquire.enumerate_source` to return 2 entries for a fake channel URL and `pipeline.ingest` / `Memovox.ingest` to a stub that records calls and writes minimal claims. Assert: first `sync()` ingests both, records both in the cursor, and calls consolidate **exactly once** (not twice); a forced-raise on entry 2 still ingests entry 1 and reports entry 2 as failed (per-entry isolation); a **second** `sync()` with the same enumeration ingests **0** (all skipped-seen) and the `SyncReport` shows `skipped==2, new==0`. Fails today: `sync` re-ingests everything, has no cursor, no isolation, no consolidate (`sdk.py:43-59`).
- **Green (implement):** rewrite `sync` to: load sources → for each, `enumerate_source` → diff vs cursor → for each unseen, `try: self.ingest(url, resolve_corpus=False)` `except` (catch `MemovoxError`/`Exception`, record failed) → on success `mark_seen` → after all sources, if anything new, run one incremental consolidation via M0.2 (interim: `self.consolidate()`). Return `SyncReport` (list of per-entry `(video_id, status)` + `n_skipped`). Errors log to **stderr** (never stdout).
- **Verify:** `make test`.
- **Commit:** `feat(sdk): subscription sync engine — enumerate, cursor-skip, isolate, single batch consolidate`

### W5 — `subscribe` / `subscriptions` / `unsubscribe` CLI · S
- **Files:** `src/memovox/cli.py` (3 new subparsers + handlers; rework `cmd_sync` to print `SyncReport`), `src/memovox/sdk.py` (small `subscribe`/`unsubscribe`/`list_subscriptions` helpers that edit `subscriptions.json`), `tests/test_cli.py` (extend).
- **Red (failing test first):** `test_cli.py` runs `main(["--store", tmp, "subscribe", "https://www.youtube.com/@chan"])`, asserts `subscriptions.json` now contains the URL once; a second `subscribe` of the same URL is idempotent (no dup); `subscriptions` lists it; `unsubscribe` removes it; `sync` with a monkeypatched `enumerate_source` prints the new/skipped/failed summary. Fails today: no such subcommands (`cli.py:254-321`).
- **Green (implement):** `Memovox.subscribe(url)` / `unsubscribe(url)` / `list_subscriptions()` read-modify-write `subscriptions.json` (dedup by normalized URL), then the three `cmd_*` handlers + subparsers wired into `build_parser` (`cli.py:242`). Rework `cmd_sync` (`cli.py:70`) to print the `SyncReport` (counts + per-entry lines).
- **Verify:** `make test`.
- **Commit:** `feat(cli): subscribe/subscriptions/unsubscribe commands + structured sync output`

### W6 — `incremental` eval block (ungated → gated) · M
- **Files:** `eval/harness.py` (add `_incremental_metrics` + an `incremental` report block; extend the frozen settings snapshot), `tests/test_eval.py` (extend).
- **Red (failing test first):** a harness-level test builds the golden corpus two ways — (A) batch: ingest all videos then one `consolidate`; (B) incremental: ingest each with `resolve_corpus=False`, then one consolidate — and asserts the **gated** report values (hit_rate, groundedness, contradiction.f1, synthesis.groundedness) are equal across A and B (`incremental.equivalent == True`). A second assertion: re-running the incremental sync over the same sources ingests 0 (`incremental.idempotent_resync == True`). Fails today: no `incremental` block in the report (`_compute_report`, `:589`).
- **Green (implement):** add `_incremental_metrics(...)` producing `{"equivalent": bool, "idempotent_resync": bool, "delta": {...}}`; surface it as `report["incremental"]`. Land it **ungated** first (computed and printed, not in `_check_thresholds`), per the thin-fixture discipline. Add the `resolve_corpus` default and sync flags to the pinned eval-settings snapshot.
- **Verify:** `python -m eval.harness` prints `incremental`; `python -m eval.harness --assert-thresholds` still passes the existing four gates unchanged.
- **Commit:** `feat(eval): incremental block — batch==incremental equivalence + idempotent re-sync (ungated)`

### W7 — Gate the `incremental` block (once stable) · S
- **Files:** `eval/harness.py` (`_check_thresholds`, `:656`), `tests/test_eval.py`.
- **Red (failing test first):** assert `--assert-thresholds` fails if `incremental.equivalent` is `False` (simulate by feeding a perturbed report); assert it passes on the real corpus.
- **Green (implement):** add `incremental.equivalent` and `incremental.idempotent_resync` (both must be `True`) to `_check_thresholds`. Do this **only after** the equivalence holds stably across the existing golden corpus *and* at least one synthetic-scale check (see Risks #3) — equivalence is a boolean invariant, so it can gate as soon as it is reliably true, but validate at >2-video scale first.
- **Verify:** `make test` + `python -m eval.harness --assert-thresholds` (now five gate-checks; all green).
- **Commit:** `feat(eval): gate incremental equivalence + idempotent re-sync`

## Eval gate
A new **`incremental`** block in `eval/harness.py`:
- `incremental.equivalent` — the gated report (hit_rate, groundedness, contradiction.f1, synthesis.groundedness) computed over a corpus ingested incrementally (`resolve_corpus=False` per video) + one consolidation equals the report from a full batch consolidate. Boolean invariant.
- `incremental.idempotent_resync` — a 2nd `sync` over the same sources ingests **0** videos (all skipped-seen via the cursor). Boolean invariant.

**Lands ungated then gated** (W6 ungated, W7 gated). Threshold: both booleans must be `True`. Because these are equivalence/idempotency invariants (not noisy scalar metrics), they can be hard-gated as soon as they hold stably across the golden corpus and one synthetic-scale check — the "≥3 golden items" rule is about scalar-metric stability and does not apply to a boolean invariant, but the synthetic-scale validation (Risk #3) is still required before gating. **The existing four gates (`hit_rate>=0.6`, `groundedness>=0.8`, `contradiction.f1>=0.5`, `synthesis.groundedness>=0.8`) must stay green** — verified by `python -m eval.harness --assert-thresholds` at every workstream and confirmed unchanged because the harness ingests with the default `resolve_corpus=True`.

## Risks & mitigations
- **Incremental == batch equivalence is genuinely hard (review Risk #3).** The hub-token quadratic in `find_contradictions` and the `dedup_claims` / `find_contradictions` `max_claims=600` cap disagreement (`consolidate.py:67`) mean a 2–3-video corpus won't exercise the failure mode. *Mitigation:* the W6/W7 equivalence test must run at **synthetic scale** (e.g. duplicate golden claims to push past where ordering/cap effects appear) before gating; consolidation incrementality itself is M0.2's responsibility — if M0.2's incremental scan isn't order-invariant, equivalence fails loudly here and surfaces an M0.2 bug, which is the intended early-warning.
- **Cursor vs `is_unchanged` double-skip drift.** The cursor skips before download; `is_unchanged` (`store.py:189`) skips after. *Mitigation:* the cursor is an *optimization* — even if a video is marked seen but absent from the store (e.g. a prior failed ingest), the next sync must still re-attempt it. Mark-seen **only on ingest success**, and treat the cursor as advisory: a `--force` / re-subscribe path clears the source's cursor.
- **MCP stdout discipline (review Risk #5).** Sync runs subprocesses and loops; any stray `print` would corrupt MCP JSON-RPC. *Mitigation:* all sync/enumeration diagnostics go to **stderr**; the only stdout writer is the CLI `cmd_sync` (never the SDK). Asserted by routing through `logging`/`sys.stderr` and a test that captures stdout during `sync()` and asserts it is empty.
- **yt-dlp `--flat-playlist` JSON shape drift across versions.** *Mitigation:* parse defensively (handle both `entries` and single-video shapes), tolerate missing fields, and keep enumeration isolated in `acquire.py` so breakage stays contained (the spec §4 stage-0 modularity intent); all parsing is unit-tested against canned payloads.
- **Network/non-determinism leaking into `make test`.** *Mitigation:* every yt-dlp call is behind `shutil.which` and **monkeypatched** in tests (patch `subprocess.run` / `acquire.enumerate_source`), mirroring the existing `mock.patch("memovox.tessera.run", ...)` pattern (`test_integration.py:100`). No test puts `yt-dlp` on PATH.
- **Per-entry isolation hiding real bugs.** A blanket `except` could swallow a genuine pipeline regression. *Mitigation:* catch and record per-entry, but include the exception type/message in the `SyncReport` and log full detail to stderr; the eval equivalence path ingests directly (no try/except) so real failures still surface in CI.
- **M0.2 not merged when this starts.** *Mitigation:* the interim path calls the existing full `consolidate()` once per batch (correct, just not incremental); the swap to M0.2's incremental entry point is a one-line change behind a clear seam (see Open questions).

## Definition of done
- [ ] `acquire.enumerate_source` expands channel/playlist URLs via `yt-dlp --flat-playlist`, never downloads, errors clearly when yt-dlp absent.
- [ ] Persisted `sync_state` cursor in the `meta` table skips already-seen ids **before** download.
- [ ] `pipeline.ingest(resolve_corpus=...)` added (default `True` → byte-identical single-video ingest).
- [ ] `Memovox.sync()` enumerates, cursor-skips, isolates per-entry failures, and runs **one** consolidation per batch via M0.2 (or interim `consolidate()`).
- [ ] `subscribe` / `subscriptions` / `unsubscribe` CLI commands manage `subscriptions.json`; `cmd_sync` prints a structured `SyncReport`.
- [ ] `incremental` eval block: `equivalent` and `idempotent_resync` both `True`; landed ungated then gated after a synthetic-scale validation.
- [ ] `make test` green (new `test_acquire.py` / `test_sync.py` + extended `test_cli.py` / `test_eval.py` / `test_integration.py`), no network.
- [ ] `python -m eval.harness --assert-thresholds` green — existing four gates unchanged, new `incremental` gate passing.
- [ ] All sync diagnostics on stderr; SDK `sync()` writes nothing to stdout.
- [ ] Free path (no yt-dlp, local `subscriptions.json`) behaves exactly as before.

## Open questions
- **M0.2 incremental entry point signature.** This track must call M0.2's incremental consolidation rather than the full `consolidate()`. Confirm the exact function/flag M0.2 exposes (e.g. `consolidate(store, ..., incremental=True)` vs a new `consolidate_incremental(...)`) so W4 wires to the right seam. Interim fallback is the existing `sdk.consolidate()`.
- **Cursor key granularity.** Key the cursor by source URL, by canonical channel id, or by both? A channel re-subscribed under a different URL form (handle vs `/channel/UC...`) should not re-ingest its back-catalog. Proposal: key by the channel/playlist id `yt-dlp` reports when available, falling back to a normalized URL hash.
- **`subscriptions.json` schema evolution.** Today entries may be a bare string or `{"url": ...}` (`sdk.py:55-57`). Should `subscribe` normalize all to objects (room for per-source options like `since`/`max`)? Proposal: write objects, keep reading both.
- **Should `sync` honor a per-source recency bound** (e.g. only the N most recent uploads, or uploads after a date)? Out of scope for the gate, but the enumeration metadata makes it cheap — confirm whether to stub the option now.
