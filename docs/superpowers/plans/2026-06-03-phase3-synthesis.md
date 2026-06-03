# Phase 3 — Synthesis (Loom, async) — design + plan

**Status:** in progress (branch `phase3-synthesis`)
**Date:** 2026-06-03
**Differentiator unlocked (spec §11):** *Cross-corpus reasoning* — the corpus
stops being a bag of islands and starts agreeing, disagreeing, and tracking how
claims change over time.

Builds on Phase 2 (`main` @ `8dd285b`): a verified temporal knowledge graph with
cross-corpus entities, speakers, `MENTIONS`/`ELABORATES`/`CORRECTS`/`CONTRADICTS`
edges, graph-expansion retrieval, and a hermetic eval harness (199/2-skip green,
all golden metrics 1.0).

## Non-negotiables (carried from Phase 2)

- **Free / stdlib-only path stays intact.** Every new capability runs end-to-end
  with the hashing embedder + lexical NLI + no LLM. Real backends are optional
  upgrades behind the existing interfaces.
- **Deterministic + idempotent.** Deterministic ids (`topic:<slug>`,
  `spk:<slug>`-style), `UNIQUE`-guarded edges, greedy clustering in a stable
  sorted order. Re-running over an unchanged corpus is a no-op.
- **Provenance is sacred.** Everything new resolves to `(video, span, modality)`
  with a deep link. No orphan facts.
- **Eval-driven.** New capabilities get golden labels + a metric; the robust ones
  become CI gates. Existing green gates must not regress.
- **Additive.** The existing ingest path and the existing contradiction path are
  unchanged. Phase 3 is new modules + new store helpers + read-time computation +
  one explicit background job. Cross-corpus consolidation is moved OUT of ingest
  into an explicit `consolidate()` pass (spec §4 stage 7: "runs as a background
  job as the library grows").

## Spec coverage map

| Spec requirement | Workstream |
|---|---|
| §4.7 topic induction + `(Moment)-[:ABOUT]->(Topic)` (spec §6) | W1 |
| §4.7 consensus scoring (source count × recency × authority) | W2 |
| §4.7 contradiction & **agreement** detection (cluster equivalent claims) | W2 (agreement) + existing `find_contradictions` (W5 orchestration) |
| §5 claim-evolution ordering by `published_at` as a first-class query | W3 |
| §5 corpus-level "literature review" synthesis across sources | W4 |
| §4.7 dedup & decay using the supersede lifecycle (wire `supersede_claim`) | W5 |
| §4 stage 7 "runs as a background job" — move consolidation off ingest | W5 |
| §2 eval-driven; §10 metrics | W6 |

Known Phase-2 follow-ups folded in: wire the unused `Settings.salience_floor`
(W6); optionally add a 3rd golden video reachable only via a graph edge for the
§5 graph leg (W6, only if it does not destabilise existing gates).

---

## W1 — Topic induction + `ABOUT` edges

**New:** `src/memovox/loom/topics.py`. **Modify:** `loom/store.py` (helpers),
`config.py` (settings), `loom/__init__.py`/exports as needed.
**Test:** `tests/test_topics.py`.

- `induce_topics(store, *, embedder, settings) -> list[Topic]`:
  - gather every moment that has a stored text vector;
  - **greedy cosine clustering** in sorted `moment_id` order (mirrors
    `cluster_by_voiceprint`): each moment joins the first cluster whose
    representative vector is `>= settings.topic_similarity`, else opens a new
    cluster. Deterministic; dimension-guarded.
  - label each cluster from its top content tokens (stopword-stripped, frequency
    then alphabetical) → `topic:<slug>` deterministic id + human label;
  - `store.set_moment_topic(moment_id, topic_id)` for each member; upsert the
    `Topic` (with `moment_count`); emit a provenanced
    `(Moment)-[:ABOUT]->(Topic)` edge (`UNIQUE`-guarded → idempotent).
- **Store helpers:** `set_moment_topic`, `list_topics`, `moments_for_topic`.
- **Settings:** `topic_similarity: float = 0.5`, `topic_min_size: int = 1`.
- **Tests:** similar moments group / dissimilar split; `topic_id` persisted; ABOUT
  edges present + provenanced; deterministic ids; idempotent re-run; label is
  content-word based.

## W2 — Claim clustering + consensus scoring

**New:** `src/memovox/loom/consensus.py`. **Test:** `tests/test_consensus.py`.

- `ClaimCluster` dataclass: `claims`, `videos` (distinct sources),
  `published` (min/max date), `representative` text, `support_count`,
  `consensus`.
- `cluster_claims(store, *, min_shared=2, jaccard=0.5, write_edges=True) -> list[ClaimCluster]`:
  cluster committed claims that are **equivalent** across videos using the
  inverted-index content-token prefilter from `consolidate.py` + a Jaccard floor
  (free, deterministic). Union groups greedily. For each cross-video equivalent
  pair, emit a provenanced `SUPPORTS` edge (agreement; `UNIQUE`-guarded).
- `score_consensus(cluster, store) -> float`: combine, into `[0,1]`,
  - **source term** — distinct video count (saturating);
  - **recency term** — newest `published_at` mapped to `[0,1]` (neutral 0.5 when
    dates absent — the free golden path);
  - **authority term** — mean claim salience (the available authority proxy).
  Weighted sum, documented constants; read-time (not persisted → schema stable).
- **Tests:** equivalent claims cluster across videos; distinct don't; consensus
  rises with more sources / higher salience / newer date; `SUPPORTS` emitted;
  deterministic; idempotent.

## W3 — Claim-evolution tracking (first-class)

**New:** `src/memovox/loom/evolution.py`. **Modify:** `sdk.py`, `cli.py`.
**Test:** `tests/test_evolution.py`.

- `EvolutionStep` dataclass: claim, video, `published_at`, deep link, `relation`
  (`None|CONTRADICTS|CORRECTS|SUPPORTS` vs the prior step), `superseded` flag.
- `claim_evolution(store, *, entity_id=None, topic=None) -> list[EvolutionStep]`:
  collect committed claims mentioning the entity (`entity_mentions`) or matching
  the topic tokens; order by `(published_at is None, published_at, t_start_s)`;
  annotate each step's relation to its predecessor by reading the graph edges.
- SDK `evolution(entity=None, topic=None)`; CLI `memovox evolution`.
- **Tests:** date ordering (None last); contradiction/correction transitions
  flagged; entity- and topic-scoped variants.

## W4 — Corpus-level synthesis ("literature review")

**New:** `src/memovox/augur/synthesize.py`. **Modify:** `augur/__init__.py`,
`sdk.py`, `server/mcp.py` (real `synthesize_topic`), `cli.py`, `server/rest.py`.
**Test:** `tests/test_synthesize.py`.

- `Synthesis` value type (text, citations, `consensus_points`, `contradictions`,
  `low_evidence`) — reuses `Citation`.
- `synthesize(store, topic, *, embedder, nli, llm=None, settings) -> Synthesis`:
  retrieve topic moments (ABOUT topic if matched, else `retrieve`); cluster
  claims (W2) for consensus points; surface in-topic contradictions
  (`find_contradictions`); compose a grounded, **every-sentence-cited** synthesis
  (extractive free path; LLM optional). Low-evidence flagged, never confabulated.
- Wire SDK `synthesize(topic)`, replace the MCP `synthesize_topic` `ask()`-shim
  with the real synthesis, add `memovox synthesize <topic>` + REST `/synthesize`.
- **Tests:** surfaces consensus + the golden cross-talk contradiction; grounded;
  cited; low-evidence path.

## W5 — Consolidation background job + dedup/decay + supersede wiring

**Modify:** `src/memovox/loom/consolidate.py` (add orchestrator + dedup),
`sdk.py`, `cli.py`, `server/mcp.py`. **Test:** `tests/test_consolidate.py`.

- `consolidate(store, *, embedder, nli, settings) -> ConsolidationReport`:
  run `induce_topics` + `find_contradictions(include_supports=True)` +
  `cluster_claims` + `dedup_claims`; return counts.
- `dedup_claims(store) -> int` — **wires `supersede_claim` (live caller)**,
  conservatively:
  - **within-video exact duplicates** (same normalized text) → keep earliest,
    supersede the rest;
  - **`CORRECTS` edges** → supersede the corrected (older) claim by the correction.
  - **Never** supersede a cross-video equivalent (that is consensus evidence, not
    a duplicate).
- SDK `consolidate()`; CLI `memovox consolidate`; MCP `consolidate` tool.
- Kept **separate from ingest** (spec: background job). Ingest stays fast +
  idempotent.
- **Tests:** orchestrator runs every leg + report counts; within-video exact dup
  superseded; `CORRECTS` supersedes corrected; cross-video equivalents untouched;
  idempotent.

## W6 — Eval coverage + gates + docs + memory

**Modify:** `eval/harness.py`, `eval/golden/` (+ `topics.json`),
`tests/test_eval.py`, `README.md`, `docs/DESIGN.md`, `assay/run` (salience_floor).

- Add `topic_f1` (cluster F1 over a new small `topics.json`) and a synthesis
  groundedness metric; compute best-effort, gate the robust ones (mirrors how
  entity_f1/der landed ungated then gated).
- Wire `Settings.salience_floor` into `assay.run` (drop claims below the floor;
  default `0.0` = no-op) — closes the unused-field follow-up.
- Optionally add a 3rd golden video reachable only via a graph edge (the §5
  graph-leg follow-up) **iff** it keeps all gates green.
- Update README (CLI: `consolidate`, `synthesize`, `evolution`), DESIGN, MCP tool
  list. Final whole-branch integration review. Update auto-memory.

---

## Risks & mitigations

- **Perturbing green gates.** Mitigation: all new work is additive; the existing
  ingest + contradiction paths are byte-unchanged; `consolidate()` is NOT run by
  the existing eval, so it cannot move existing metrics. New metrics are added
  best-effort first.
- **Over-superseding (dedup).** Mitigation: supersede only within-video exact
  duplicates and `CORRECTS` targets; never cross-video equivalents; conservative,
  unit-tested, idempotent. Golden corpus has no CORRECTION claims and no exact
  within-video dups, so the golden eval is unaffected (verified by re-running).
- **Recency without dates.** Golden `.vtt` ingests have `published_at = None`;
  the recency term degrades to a neutral 0.5 so consensus still ranks by sources
  + authority. Date-sensitive behaviour is unit-tested with explicit dates.
- **Topic granularity.** `topic_similarity` is tunable; default chosen so the two
  scaling-laws talks share a topic in the golden eval.

## Execution

Inline TDD, one commit per workstream, `make test` + `python -m eval.harness`
green after each. Merge to `main` after a final integration review.
