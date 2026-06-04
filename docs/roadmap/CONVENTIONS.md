# How we execute the roadmap

The conventions every track follows. They are the same ones that built Phases 0–3
(see the committed Phase 2/3 plans), written down so any session — human or agent —
can pick up a track and proceed identically.

## The loop (per track)

1. **Branch.** From `main`: `git checkout -b phase4-<track-slug>` (e.g.
   `phase4-observability`). One branch per track keeps reviews scoped and lets
   tracks land independently.
2. **Resolve open questions.** Read the track doc's *Open questions* and the
   cross-track decisions in [`README.md`](README.md); confirm anything that changes
   the design before writing code.
3. **Execute workstreams in order, TDD.** Each workstream is one commit. Follow
   red → green → verify (below). Never start a workstream before the previous one
   is green and committed.
4. **Keep the gates green at every commit.** `make test` and
   `python -m eval.harness --assert-thresholds` must pass before each commit.
5. **Track progress.** Tick the track doc's *Definition of done*; update its
   `Status:` line; update [`PROGRESS.md`](PROGRESS.md).
6. **Review before merge.** Request a code review of the branch diff (the Phase-3
   pattern caught a real bug this way). Fix findings TDD-style.
7. **Merge.** Fast-forward to `main` once the review is clean and gates are green;
   delete the branch. Push only when asked.

## TDD (non-negotiable — same as Phases 0–3)

Red → Green → Refactor. **No production code without a failing test first.**

- **Red:** write the smallest test that asserts the new behavior; run it; confirm
  it fails for the *right reason* (feature missing, not a typo/import error).
- **Green:** minimal code to pass. Don't add scope the test doesn't demand.
- **Verify:** the new test passes, the **full** suite stays green
  (`make test` → 247+ pass / 2 skip), and the eval gates hold.
- **Refactor:** clean up while staying green.

Each track doc spells out the Red/Green/Verify/Commit for every workstream.

## Commits

- One commit per workstream, with the message given in the doc:
  `type(scope): summary` (e.g. `feat(loom): …`, `fix(augur): …`, `docs: …`,
  `feat(eval): …`). Body explains the *why* when non-obvious.
- Do **not** add "authored by Claude" trailers (per repo/user preference).
- Commit only when the gates are green. If a workstream can't go green, keep it
  in-progress — don't commit a red state or weaken a gate to pass.

## The free-path guarantee (the project's core invariant)

Every track must preserve this — it is what makes memovox runnable for free:

- **Stdlib-only, deterministic, idempotent core.** No new hard dependency. Every
  model *and* storage slot has a deterministic fallback behind a common interface;
  real backends are optional upgrades selected by `auto`/explicit and gated by
  `is_available()`.
- **New capabilities default OFF** when they could move a gate number. The default
  (free) path must stay **byte-identical** — prove it with an off==today assertion.
- **Provenance is sacred.** Every fact/edge/citation resolves to
  `(video, span, modality, confidence)` + a deep link. A change that lets the
  displayed span drift from the span actually verified by the NLI gate is a bug.
- **Logging goes to stderr** (the MCP server speaks JSON-RPC on stdout). Never
  `print()` or attach a stdout `StreamHandler` on a path the MCP server can reach.
- **Idempotency:** deterministic ids, `UNIQUE`-guarded edges, status-gated
  supersession; re-running any operation over unchanged input is a no-op.

## The two global disciplines (owned by M-X, apply everywhere)

1. **Thin-fixture gating.** A 2–3 video golden corpus is too small to hard-gate a
   new metric stably. Land every new metric **ungated** (compute + report it), with
   an *equivalence* assertion that the free default == today. Promote it to a
   `_check_thresholds` gate only once ≥3 stable golden items exercise it. This is
   exactly why `topic_f1`/`entity_f1`/`der` shipped informational first.
2. **Frozen eval-settings snapshot.** The harness pins the free stack
   (`_FREE_BACKENDS`); as default-OFF flags proliferate it must *also* pin them, so
   a future default flip or a leaked `MEMOVOX_*` env var cannot silently change a
   gate. When a track adds a flag, add it to the snapshot in the same commit.

## Verifying

```bash
make test                                   # stdlib unittest; 247+ pass / 2 skip
PYTHONPATH=src python -m eval.harness        # print the metric report
PYTHONPATH=src python -m eval.harness --assert-thresholds   # CI gate (exit 0)
make lint                                    # ruff (advisory; typing.List is house style)
```

For a behavior change to a gated metric, capture the harness JSON **before and
after** and confirm byte-identity on the free path (the Phase-3 `find_contradictions`
edge fix and every default-OFF feature were verified this way).

## Definition of done — per wave

- **Wave 0:** measurement spine live; free-path scale parity + incremental==full
  consolidation; word-precise spans with the free VTT corpus unchanged; the M-X
  disciplines (snapshot + thin-fixture rule) encoded in the harness.
- **Wave 1:** a measurable multimodal lift on on-screen-only knowledge; the §5 graph
  leg exercised end-to-end by `talk_c`; entity_f1/der promoted to gates.
- **Wave 2:** multi-part questions fully answered, reranked, and returned as stitched
  deep-linked clips.
- **Wave 3:** subscriptions sync incrementally; consolidation runs async/resumable;
  optional FastAPI/worker deploy; backends ranked, not assumed.

## When the plan changes

These docs are living. If a track's reality diverges (a workstream splits, an open
question resolves a different way), edit the track doc in the same branch and note
it in the commit. Keep [`README.md`](README.md)'s table and the single-owner
reconciliations authoritative — if ownership of a shared concern moves, update it
there so no two tracks duplicate work.
