# M2.3 — Answer-with-video clip stitching

> **Wave:** 2 · **Effort:** M · **Status:** ✅ done (branch `phase4-clips`, 7/7 + review fix; 420 pass / 2 skip; 10 gates incl. clip.coverage>=0.3). Pure stitch (provenance-safe: widens to union of verified spans), ranged deep links, REST/CLI/MCP surfacing, opt-in ffmpeg render.
> **Depends on:** M0.3 (word-precise spans), M2.2 (citation-build coordination on `answer.ask`) · **Owns (single-owner concerns):** none · **Blocks:** none
> **Spec:** §5 ("Answer-with-video: return stitched, deep-linked clip references"), §8 (`GET /clip`, SDK, MCP)

## Goal
Deliver the §5 "answer-with-video" promise: turn a question's scattered cited `Moment` spans into a small set of minimal, deep-linked **clips** — *"the 92 seconds where this is explained"* — instead of leaving the user a list of point timestamps. A pure, deterministic `augur/stitch.py` merges adjacent/overlapping cited spans **per video** into the fewest non-overlapping `(t_start, t_end)` windows, mints **ranged** YouTube deep links (`watch?v=…&start=&end=`), and attaches the result to `Answer.clips` (a new `Clip` type). The clips surface uniformly through the SDK (`Answer.clips` / `Answer.to_dict()`), REST (`GET /clip` becomes a stitched superset of today's per-span moment list), the CLI `ask` rendering, and the MCP `search_knowledge` payload. An **optional** `render_clip` ffmpeg-concat helper materializes a local `.mp4` only when local media + ffmpeg are present; it is a no-op (returns `None`) otherwise. The arithmetic operates purely on spans retrieval already produces, and tightens onto the word-precise span windows M0.3 introduces.

## Why it matters
Point timestamps make the user scrub. Stitched clips make the answer *playable*: every claim in a grounded answer becomes one click into the exact 90 seconds of video that supports it, and a multi-citation answer collapses to a handful of distinct watch windows rather than a wall of near-duplicate links. This is the literal Phase-4 spec bullet (§11 "answer-with-video clip stitching") and the headline differentiator over text-only RAG. Because stitching is pure arithmetic over spans the retrieval layer already returns, the capability ships free, deterministic, and offline; the ffmpeg render is the only upgrade and it is strictly opt-in on local media.

## Scope (reconciled)
In scope:
- **`augur/stitch.py`** — a pure, dependency-free module. Core function `stitch_clips(citations, *, videos, merge_gap_s=...) -> List[Clip]`:
  - groups citations by `video_id`;
  - sorts each group by `t_start_s`;
  - merges any two adjacent windows whose gap `<= merge_gap_s` (and all overlapping windows) into one span — the union `[min(t_start), max(t_end)]`;
  - returns minimal **non-overlapping, sorted** clips per video, each carrying the union span, the set of contributing citation indices, a ranged deep link, and a duration.
- **`Clip` dataclass + `Answer.clips`** in `augur/types.py`. `Clip` carries `video_id`, `t_start_s`, `t_end_s`, `duration_s`, `title`, `deep_link` (ranged), `citation_indices: List[int]`, and `to_dict()`. `Answer` gains `clips: List[Clip] = field(default_factory=list)` and `to_dict()` emits `"clips"`.
- **Ranged deep links.** Add a `deep_link_range(source_url, t_start, t_end)` helper in `util.py` alongside the existing `deep_link` (which stays untouched): YouTube → `https://www.youtube.com/watch?v=<id>&start=<int t0>&end=<int t1>`; non-YouTube → reuse the existing `#t=`/`&t=` start-only form (no standard ranged fragment exists, so the start link is the honest fallback).
- **Wire into `answer.ask`.** After citations are built (and after the temporal re-index that already runs), call `stitch_clips(...)` and set `Answer.clips`. This is the M2.2 coordination point — all of M2.1/M2.2/M2.3 touch the post-retrieval section of `answer.ask`; stitching is a strictly additive final step that reads the finalized `citations` list.
- **REST `GET /clip`** becomes a **stitched superset** of today. Keep the current response keys (`video_id`, `t_start_s`, `t_end_s`, `deep_link`, `moments`) so existing consumers don't break, and **add** a `clips` array (the stitched windows over the moments overlapping the requested range, with ranged deep links).
- **CLI `ask`** renders a "Clips:" section under the existing "Citations:" block (formatted span via `format_span`, ranged deep link), and `--json` already serializes `Answer.to_dict()` so clips appear automatically.
- **MCP `search_knowledge`** payload includes clips automatically (it returns `answer.to_dict()`), with no schema change required.
- **Optional `render_clip(video, clip, *, out_dir) -> Optional[Path]`** in `augur/stitch.py` (or a thin `augur/render.py`): only when a local media file exists for the video AND `audio.which_ffmpeg()` returns a path, shell out to ffmpeg to cut/concat the clip's span to an `.mp4`; otherwise return `None`. Pure no-op on the free path.
- **Eval:** a `clip` metric block — `clip.coverage` (IoU >= 0.3 of the best stitched clip vs a gold clip span) plus structural **invariants** (clips are non-overlapping per video; stitching is **idempotent** — `stitch(stitch(x)) == stitch(x)`). Lands **ungated**, gated only once >= 3 stable golden clip items exist.

Folded in from the completeness review / reconciliation:
- Feeds on **M0.3 word-precise spans**: clip coverage must target the tightened word-window `(t0,t1)` on each Moment/citation, not the cue-granular ceiling. Stitching reads whatever span the citation carries, so this is automatic once M0.3 lands — but the gold clip spans and the coverage test must be authored against word-precise expectations.
- Feeds on **M2.2** (and M2.1 rerank / M1.1 visual-aware retrieval): the citation set stitching consumes is already reranked/plan-fused; stitching is order-independent (it sorts by time), so it composes cleanly.
- The new default-OFF flag (clip merge enabled? render enabled?) must be added to the **frozen eval-settings snapshot**, not just `_FREE_BACKENDS`, per global discipline (b).

Non-goals / deferrals:
- **No video transcoding/serving** in the free path — `render_clip` is opt-in, local-media-only, never invoked by the gate or default `ask`.
- **No new retrieval** — stitching adds zero retrieval; it only post-processes citations.
- **No ranged-fragment invention for non-YouTube** sources — start-only deep link is the documented fallback.
- **No change to the existing `deep_link`** (start-only) helper or the per-citation `Citation.deep_link` — clips add a *parallel* ranged link; citations keep their start link byte-for-byte.

## Current state (grounded)
- **No clip type, no stitch module.** `augur/types.py` (`/Users/connor/projects/memovox/src/memovox/augur/types.py:9-40`) defines only `Citation` and `Answer`; `Answer.to_dict()` (`:34-40`) emits `text`/`strategy`/`low_evidence`/`citations` — no `clips`. `augur/__init__.py` (`/Users/connor/projects/memovox/src/memovox/augur/__init__.py:1-10`) exports `Answer`/`Citation` but nothing clip-related. No `stitch.py` exists in `src/memovox/augur/`.
- **`answer.ask` builds the citation list this track consumes.** `/Users/connor/projects/memovox/src/memovox/augur/answer.py:62-152`: retrieval → per-moment `Citation` construction (`:100-125`) → temporal re-index (`:127-139`) → synthesis (`:141-147`) → `return Answer(...)` (`:152`). The clip step inserts right before the synthesis/return, reading the finalized `citations`. Each `Citation` already carries `video_id`, `t_start_s`, `t_end_s`, `deep_link` (start-only, from `prov.deep_link`), `title`, `index`.
- **Deep links are start-only today.** `util.deep_link` (`/Users/connor/projects/memovox/src/memovox/util.py:128-137`) returns `https://youtu.be/<id>?t=<int>` for YouTube and `<url>{#|&}t=<int>` otherwise — start only, no `end`. `make_provenance` (`/Users/connor/projects/memovox/src/memovox/loom/models.py:157-174`) calls it; `Provenance.deep_link` flows into `Citation.deep_link`. `youtube_id` (`util.py:88-107`) already extracts the id we need for the ranged `watch?v=` form.
- **REST `/clip` already does a primitive, un-stitched version.** `/Users/connor/projects/memovox/src/memovox/server/rest.py:64-90` (`_clip`): given `video`, `t_start`, `t_end`, it returns every `Moment` overlapping `[t_start, t_end]` plus a single start-only `deep_link`. It does **not** merge spans or emit clips. The endpoint is already documented in the file header (`:8`) and the `/` index (`:59-61`). This is the "superset of today" baseline.
- **CLI `ask` renders citations only.** `/Users/connor/projects/memovox/src/memovox/cli.py:47-67` (`cmd_ask`): prints `answer.text`, then a "Citations:" block (`:55-66`) with `seconds_to_hms` timestamps and per-citation deep links; `--json` dumps `answer.to_dict()` (`:49-51`). No clip rendering.
- **MCP returns the answer dict verbatim.** `/Users/connor/projects/memovox/src/memovox/server/mcp.py:137-139` (`_tool_search_knowledge`) returns `_tool_json(answer.to_dict())` — clips will appear for free once `to_dict()` includes them. (Note: the `modality` param in `search_knowledge`'s schema, `mcp.py:43`, is the dead param M1.1 owns; do not touch it here.)
- **ffmpeg helper exists and is the right gate for `render_clip`.** `audio.which_ffmpeg()` (`/Users/connor/projects/memovox/src/memovox/audio.py:26-27`) returns `shutil.which("ffmpeg")` or `None`; `tessera/frames.py:48-49` already uses the `which_ffmpeg() or not path.exists()` → graceful-degradation pattern (`tessera/frames.py` header documents the spec §9 graceful-degradation contract). `render_clip` mirrors this exactly.
- **No local-media pointer on `Video` today.** `Video` (`/Users/connor/projects/memovox/src/memovox/loom/models.py:37-52`) has no persisted local-media path; it has `source_url` + `content_hash`. So `render_clip` needs an explicit `media_path=` argument (or a best-effort lookup the caller supplies) — it cannot assume the store knows where bytes live. This keeps `render_clip` out of the default path cleanly.
- **`moments_for_video` returns moments ordered by `idx`** (`/Users/connor/projects/memovox/src/memovox/loom/store.py:286-290`), so REST `_clip` already has a deterministic order to stitch over.
- **Eval harness shape to extend.** `/Users/connor/projects/memovox/eval/harness.py`: report assembled in `_compute_report` (`:587-631`); gates in `_check_thresholds` (`:656-670`) read `report["retrieval"]["hit_rate"]`, `report["groundedness"]`, `report["contradiction"]["f1"]`, `report["synthesis"]["groundedness"]`; gate constants at `:642-649`. Golden corpus in `/Users/connor/projects/memovox/eval/golden/` (`qa.json` shape at `qa.json:1-...` uses `q`/`relevant_moment_substrings`/`answer_substrings`). A new `clip` block is added to the report dict and a new gold file (e.g. `clips.json`) added to the corpus. The harness already documents the **ungated-then-gated** discipline (`harness.py:638-649`).

## Free-path guarantee
- **Stitching is the new default and it is pure stdlib + deterministic.** `stitch_clips` is arithmetic over `(t_start, t_end)` tuples — no models, no I/O, no randomness — so it cannot regress determinism. Its output ordering is fully specified (group by `video_id` in first-seen citation order, sort spans by `t_start` then `t_end`). It runs every `ask`, but it only *adds* `Answer.clips`; the `text`, `citations`, `strategy`, and `low_evidence` fields stay byte-identical to today.
- **What defaults OFF / what must stay byte-identical:**
  - `render_clip` defaults to **no-op**: returns `None` unless an explicit local `media_path` exists AND `which_ffmpeg()` is truthy. Never called from `ask`/REST/MCP/CLI default paths. Never touched by the eval gate.
  - The existing **`Citation.deep_link` stays start-only** and byte-identical (it keeps calling `util.deep_link`). Ranged links live only on `Clip.deep_link` via the new `deep_link_range` helper — `deep_link` itself is not modified.
  - REST `/clip` keeps **every existing response key** unchanged; `clips` is purely additive.
  - The eval gates that exist today (`hit_rate>=0.6`, `groundedness>=0.8`, `contradiction.f1>=0.5`, `synthesis.groundedness>=0.8`) must stay green — verified by running the harness after each workstream. Because stitching never alters `text`/`citations`, groundedness and retrieval are mathematically untouched.
- **Frozen eval-settings snapshot:** any new default-OFF flag introduced (e.g. a `clip_render_enabled` or a `clip_merge_gap_s` knob) is recorded in the frozen settings snapshot the harness pins (global discipline (b)), so a future default flip can't silently move gate numbers. If `merge_gap_s` is a `Settings` field, its default is pinned there.
- **Logging to stderr only** (MCP speaks JSON-RPC on stdout): `render_clip`'s ffmpeg invocation must never `print` to stdout; any diagnostic goes to stderr (follow the `tessera/frames.py` pattern of silent graceful degradation — preferred is to emit nothing on the no-op path).

## Workstreams

### W1 — `Clip` type + `Answer.clips` + ranged deep-link helper · S
- **Files:** `src/memovox/augur/types.py` (add `Clip`, extend `Answer`), `src/memovox/util.py` (add `deep_link_range`), `src/memovox/augur/__init__.py` (export `Clip`), `tests/test_augur.py` (new tests) or a new `tests/test_stitch.py`.
- **Red (failing test first):**
  - `test_deep_link_range_youtube` asserts `deep_link_range("https://youtu.be/abc123", 750, 845) == "https://www.youtube.com/watch?v=abc123&start=750&end=845"`, and the non-YouTube case falls back to the start-only `deep_link` form. Fails: `deep_link_range` does not exist.
  - `test_clip_to_dict_and_answer_clips_serialized` asserts a `Clip(...)` round-trips through `to_dict()` with `citation_indices`/`duration_s`, and that `Answer(text="x").to_dict()` now contains a `"clips": []` key. Fails: `Clip` and `Answer.clips` do not exist; current `Answer.to_dict()` (`types.py:34-40`) has no `clips`.
- **Green (implement):** add the `Clip` dataclass + `to_dict()`; add `Answer.clips: List[Clip] = field(default_factory=list)` and `"clips": [c.to_dict() for c in self.clips]` in `Answer.to_dict()`; add `deep_link_range` to `util.py` reusing `youtube_id`; export `Clip` from `augur/__init__.py`.
- **Verify:** `python -m unittest tests.test_augur tests.test_stitch` green; `make test` still passes (existing `Answer.to_dict()` consumers unaffected — additive key only).
- **Commit:** `feat(augur): Clip type, Answer.clips, ranged deep-link helper (spec §5/§8)`

### W2 — Pure `stitch_clips` merge with invariants · M
- **Files:** `src/memovox/augur/stitch.py` (new), `src/memovox/augur/__init__.py` (export `stitch_clips`), `tests/test_stitch.py`.
- **Red (failing test first):** in `tests/test_stitch.py`:
  - `test_merges_adjacent_and_overlapping_per_video` — given citations on one video at `(0,30),(28,60),(200,240)` with `merge_gap_s=2.5`, expects exactly **two** clips `(0,60)` and `(200,240)`, each listing the contributing `citation_indices`.
  - `test_does_not_merge_across_videos` — same spans on two different `video_id`s never merge.
  - `test_clips_non_overlapping_and_sorted` — output clips per video are sorted by `t_start` and pairwise non-overlapping.
  - `test_idempotent` — re-stitching the spans implied by `stitch_clips` output reproduces identical clips (`stitch(stitch(x)) == stitch(x)`).
  - `test_ranged_deep_link_on_clip` — each clip's `deep_link` is the ranged form for a YouTube video.
  These fail because `stitch_clips` does not exist.
- **Green (implement):** `stitch_clips(citations, *, videos: dict[str, Video], merge_gap_s: float = 2.5) -> List[Clip]`. Group by `video_id` (preserve first-seen video order), sort each group by `(t_start, t_end)`, sweep-merge with the gap rule, build `Clip` with union span, `duration_s = t_end - t_start`, accumulated `citation_indices`, `title` from the `Video`, and `deep_link = deep_link_range(video.source_url, t0, t1)`. Pure function, no store access (caller passes the `videos` map it already built). Export from `__init__.py`.
- **Verify:** `python -m unittest tests.test_stitch` green; `make test` green.
- **Commit:** `feat(augur): pure span-stitching into minimal deep-linked clips (spec §5)`

### W3 — Wire stitching into `answer.ask` · S
- **Files:** `src/memovox/augur/answer.py`, `tests/test_augur.py`.
- **Red (failing test first):** `test_ask_returns_stitched_clips` — ingest a tiny store whose top citations for a query land on two adjacent moments of one video; assert the returned `Answer.clips` has exactly one clip spanning both, with a ranged deep link, and that `Answer.text`/`Answer.citations` are **unchanged** vs the pre-clip behavior (compare against the existing `test_answer_has_citation_and_deeplink` expectations at `tests/test_augur.py:53`). Fails: `ask` does not populate `clips` (currently returns `Answer(text=..., citations=..., ...)` at `answer.py:152` with no clips arg).
- **Green (implement):** in `answer.ask`, after the temporal re-index block (`answer.py:127-139`) and before/at the `return`, build the `videos` map from the existing `video_cache` (`answer.py:99`) and call `stitch_clips(citations, videos=video_cache, merge_gap_s=settings.<knob>)`; pass `clips=...` into `Answer(...)`. The `video_cache` already holds the `Video` objects keyed by `video_id`, so no extra store reads.
- **Verify:** `make test` green; **run `python -m eval.harness` and confirm `retrieval.hit_rate`, `groundedness`, `contradiction.f1`, `synthesis.groundedness` are byte-identical to the pre-change report** (stitching adds a field, never mutates citations).
- **Commit:** `feat(augur): populate Answer.clips from cited spans on every ask (spec §5)`

### W4 — Surface clips through REST / CLI / MCP · M
- **Files:** `src/memovox/server/rest.py` (extend `_clip`), `src/memovox/cli.py` (extend `cmd_ask` rendering), `tests/test_cli.py`, `tests/test_loom.py` (or a new `tests/test_rest.py` for the handler), `tests/test_mcp.py`.
- **Red (failing test first):**
  - REST: `test_clip_endpoint_returns_stitched_superset` — call the `make_handler` flow (or `_clip` directly) for a range covering two adjacent moments; assert the response still has the legacy keys (`video_id`, `t_start_s`, `t_end_s`, `deep_link`, `moments`) AND a new `clips` array with the merged, ranged-linked window. Fails: `_clip` (`rest.py:75-90`) emits no `clips`.
  - CLI: extend the `ask` test (`tests/test_cli.py:47-65`) to assert that when citations exist, stdout contains a "Clips:" line and a ranged link, and `--json` output contains `"clips"`. Fails today.
  - MCP: `test_search_knowledge_includes_clips` — drive `search_knowledge` and assert the returned JSON text contains `"clips"`. Fails until `to_dict()` (W1) + `ask` (W3) land; this test guards the MCP wire stays correct (no schema change needed).
- **Green (implement):**
  - REST `_clip`: after collecting overlapping moments, build `Citation`-shaped or lightweight span tuples for those moments and call `stitch_clips` (passing the single `video` in a `{video_id: video}` map); add `"clips": [c.to_dict() for c in clips]` to the response. Keep all existing keys.
  - CLI `cmd_ask`: after the "Citations:" loop (`cli.py:55-66`), add a "Clips:" block iterating `answer.clips` with `format_span(c.t_start_s, c.t_end_s)` and `c.deep_link`.
  - MCP: no code change (returns `to_dict()`); the test just locks the contract.
- **Verify:** `make test` green (all REST/CLI/MCP suites); eval gates still green.
- **Commit:** `feat(serving): surface stitched clips via /clip, CLI ask, and MCP (spec §8)`

### W5 — Optional ffmpeg `render_clip` (no-op on free path) · S
- **Files:** `src/memovox/augur/stitch.py` (add `render_clip`) or `src/memovox/augur/render.py`, `tests/test_stitch.py`.
- **Red (failing test first):**
  - `test_render_clip_noop_without_media_or_ffmpeg` — calling `render_clip(video, clip, media_path=None)` (or with a nonexistent path) returns `None` and writes no file. Must pass on CI (no ffmpeg/local media). Fails: `render_clip` does not exist.
  - `test_render_clip_builds_ffmpeg_cmd` — with `which_ffmpeg` monkeypatched to a fake path and `subprocess.run` patched, assert the constructed argv cuts the clip span (`-ss <t0> -to <t1>`) to an `.mp4` under `out_dir`, mirroring the `tessera/frames.py` invocation style. Asserts command shape without executing ffmpeg.
- **Green (implement):** `render_clip(video, clip, *, media_path=None, out_dir) -> Optional[Path]`: return `None` immediately unless `media_path` exists and `audio.which_ffmpeg()` is truthy; otherwise shell out (`subprocess.run`, captured output, timeout, no stdout) to extract `[clip.t_start_s, clip.t_end_s]` to a deterministic output filename (e.g. `<video_id-slug>_<int t0>-<int t1>.mp4`). Graceful-degrade (return `None`) on `OSError`/`SubprocessError`, exactly like `tessera/frames.py:59`.
- **Verify:** `make test` green on a machine with no ffmpeg (the no-op path is the CI path); the command-shape test passes via mocking.
- **Commit:** `feat(augur): optional ffmpeg clip render, no-op without local media (spec §5)`

### W6 — `clip.coverage` eval metric + invariants (ungated) · M
- **Files:** `eval/golden/clips.json` (new gold file), `eval/harness.py` (compute `clip` block; **do not** gate yet), `tests/test_eval.py` (or wherever harness metric tests live) for the IoU + invariant unit tests.
- **Red (failing test first):**
  - Unit: `test_clip_coverage_iou` — given a gold clip span and a stitched clip, the IoU computation matches a hand-calculated value; `test_stitch_invariants_hold_on_golden` — clips are non-overlapping per video and stitching is idempotent. Fails until the metric function exists.
  - Harness: assert `run_eval(...)["clip"]` exists with `coverage`, `non_overlap` (bool), `idempotent` (bool) keys. Fails: `_compute_report` (`harness.py:587-631`) emits no `clip` block.
- **Green (implement):** add a `clip_coverage(found_clips, gold_clips)` function (best-match IoU per gold clip, averaged) and an invariants check; author `eval/golden/clips.json` with the gold clip spans for existing QA items (authored against **word-precise** M0.3 spans). Add a `"clip"` block to the report dict in `_compute_report`. **Do not** add a threshold to `_check_thresholds` (`harness.py:656-670`) yet — land **ungated** per discipline (a)/(b). Record the new clip flag/knob in the frozen eval-settings snapshot.
- **Verify:** `python -m eval.harness` prints the `clip` block with `coverage>=0.3` on the gold items (validated by hand) and `non_overlap/idempotent == true`; **the four existing gates stay green**. `make test` green.
- **Commit:** `feat(eval): clip.coverage (IoU) + stitch invariants, ungated (spec §10)`

### W7 — Gate `clip.coverage` once >= 3 stable golden clip items exist · S
- **Files:** `eval/harness.py` (`_check_thresholds` + a `_CLIP_COVERAGE_GATE` constant), `eval/golden/clips.json` (grow to >= 3 items), `tests/test_eval.py`.
- **Red (failing test first):** `test_clip_coverage_gate_enforced` — a forced low-coverage report makes `_check_thresholds` return a `clip.coverage … < 0.3` failure. Fails until the gate is wired.
- **Green (implement):** add `_CLIP_COVERAGE_GATE = 0.3` and the check in `_check_thresholds`; update the harness docstring/`--assert-thresholds` help (`harness.py:688-692`) to mention the clip gate. Only do this **after** >= 3 stable clip golden items are verified across runs (talk_c from M1.2 supplies more material).
- **Verify:** `python -m eval.harness --assert-thresholds` exits 0 with all five gates green; two consecutive runs identical (determinism).
- **Commit:** `feat(eval): gate clip.coverage >= 0.3 IoU now that >=3 golden clips exist`

## Eval gate
- **New metric:** `clip.coverage` = mean best-match IoU between each gold clip span and the stitched clips for the same video; **threshold 0.3** (IoU >= 0.3), matching the reconciled scope. Accompanied by two structural invariants asserted in the report and in unit tests: **per-video non-overlap** and **idempotency** (`stitch(stitch(x)) == stitch(x)`).
- **Ungated-then-gated (global discipline (a)):** lands **ungated** in W6 (computed and printed, no `--assert-thresholds` failure), and is **gated** only in W7 once **>= 3 stable golden clip items** exist (the corpus grows via M1.2's `talk_c`). Until then the invariants are enforced by `tests/test_stitch.py` (unit tests, not the golden gate).
- **Frozen eval-settings snapshot (global discipline (b)):** any new default-OFF flag (`clip_render_enabled`, `merge_gap_s` default) is pinned in the snapshot the harness freezes, not just `_FREE_BACKENDS`, so a future default flip can't move numbers silently.
- **Regression guard:** the four existing gates — `retrieval.hit_rate>=0.6`, `groundedness>=0.8`, `contradiction.f1>=0.5`, `synthesis.groundedness>=0.8` (`harness.py:642-649`) — **must stay green** after every workstream. Because stitching only *adds* `Answer.clips` and never mutates `text`/`citations`, these are mathematically unaffected; W3's verify step asserts byte-identical pre/post reports for these four.

## Risks & mitigations
- **Span drift vs. the verification gate (review risk #4 — provenance is sacred).** The displayed clip span must not drift from the span the NLI groundedness gate actually verified. *Mitigation:* stitching only ever takes the **union of citation spans** (it widens to a superset of already-verified windows, never narrows or invents); the citation premise stays segment-granular as M0.3 specifies; the coverage test asserts clips cover (not replace) cited spans, and W3 asserts `citations` are byte-identical pre/post stitch.
- **Word-precise spans not yet landed (M0.3 dependency).** If M0.3 hasn't merged, citation spans are cue-granular and gold clip IoU targets would be authored against the wrong granularity. *Mitigation:* this track is sequenced after M0.3; author `clips.json` only once word-precise spans exist, and note in the gold file which span granularity it targets.
- **Ranged YouTube link semantics.** `watch?v=&start=&end=` requires integer seconds and only works on the `youtube.com/watch` host (not the `youtu.be` short form used by `deep_link`). *Mitigation:* `deep_link_range` normalizes to the `watch?v=` form and `int()`-truncates; a unit test pins the exact string.
- **Merge-gap tuning over a 2-talk corpus (review risk #1 — eval thinness).** A too-large `merge_gap_s` over-merges; too small under-merges; a tiny corpus won't expose either. *Mitigation:* keep the metric ungated until >= 3 items; default `merge_gap_s` conservatively (reuse the moment-gap intuition, e.g. ~2.5s); pin it in the frozen snapshot.
- **MCP stdout discipline (review risk #5).** `render_clip` shelling to ffmpeg must not leak to stdout (MCP JSON-RPC). *Mitigation:* capture output, no `print`; prefer the silent graceful-degradation pattern of `tessera/frames.py`; W5's test asserts no stdout write.
- **REST `/clip` backward compatibility.** Existing consumers depend on the current key set. *Mitigation:* `clips` is strictly additive; the REST test asserts all legacy keys remain.
- **Determinism of clip ordering / IDs.** Non-deterministic grouping or output filenames would erode reproducibility. *Mitigation:* group by first-seen `video_id` order, sort spans by `(t_start, t_end)`, derive render filenames from `slugify`+integer span; two-run determinism asserted in W7.

## Definition of done
- [ ] `Clip` dataclass + `Answer.clips` exist; `Answer.to_dict()` emits `clips`; `Clip` exported from `augur`.
- [ ] `util.deep_link_range` mints ranged YouTube links; non-YouTube falls back to start-only; `deep_link` itself unchanged.
- [ ] `augur/stitch.py` merges adjacent/overlapping cited spans per video into minimal, non-overlapping, sorted clips; pure, deterministic, idempotent (unit-tested).
- [ ] `answer.ask` populates `Answer.clips`; `text` and `citations` are byte-identical to pre-change.
- [ ] `GET /clip` returns the legacy response **plus** a `clips` array; CLI `ask` renders a "Clips:" block and `--json` includes clips; MCP `search_knowledge` includes clips (contract test).
- [ ] `render_clip` is a no-op (returns `None`) without local media + ffmpeg; builds the correct ffmpeg argv when both present (mock-tested); writes nothing to stdout.
- [ ] `eval/golden/clips.json` added; `run_eval` emits a `clip` block with `coverage`, `non_overlap`, `idempotent`; lands **ungated**.
- [ ] `clip.coverage >= 0.3` gated only after >= 3 stable golden clip items exist; new flags pinned in the frozen eval-settings snapshot.
- [ ] `make test` green; `python -m eval.harness --assert-thresholds` green with the four existing gates unchanged.

## Open questions
- **`merge_gap_s` home & default.** Should the merge gap be a `Settings` field (then it must be pinned in the frozen snapshot) or a `stitch_clips` keyword default only? Proposed: a `Settings` field defaulting to ~2.5s (aligns with `moment_gap_sec`), pinned in the snapshot. Confirm the exact default.
- **Local-media discovery for `render_clip`.** `Video` has no persisted local-media path today (`models.py:37-52`). Should M2.3 add a media-path lookup (e.g. a convention under the store dir keyed by `content_hash`), or require callers to pass `media_path=` explicitly? Proposed: explicit `media_path=` argument now; a persisted pointer is deferred to the Scale/Serving tracks. Confirm.
- **REST `/clip` clip construction without `Citation` objects.** The endpoint operates on raw `Moment`s, not `Citation`s. Confirm it is acceptable to stitch over lightweight `(video_id, t_start, t_end)` span tuples there (reusing the same `stitch_clips` core via a small adapter) rather than constructing throwaway `Citation`s.
- **Gold clip span granularity.** Confirm `clips.json` gold spans are authored against M0.3 word-precise windows (not cue-granular), so coverage is meaningful — this depends on M0.3 being merged first.
