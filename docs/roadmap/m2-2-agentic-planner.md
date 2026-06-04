# M2.2 — Agentic multi-step query planner

> **Wave:** 2 · **Effort:** M · **Status:** ✅ done (branch `phase4-agentic-planner`, 5/5 + review fix; 403 pass / 2 skip; 9 gates). Single-clause byte-identical; deterministic decompose default (LLM decomposer opt-in via planner_agentic); round-robin merge; plan.subquery_recall ungated (1.0).
> **Depends on:** M2.1 (coordinate citation-build changes) · **Owns (single-owner concerns):** none · **Blocks:** M2.3
> **Spec:** §5 ("Query planner (agentic): decompose the question, then choose the retrieval strategy and modality"; "Multi-hop traversal … for synthesis questions"; "every sentence carries a citation").

## Goal
Upgrade `augur/planner` from a single-pass keyword intent classifier into an agentic planner that **decomposes a multi-part question into ordered sub-queries**, assigns a `strategy` + `modality` to each sub-query, executes each one through the existing dense/lexical/graph legs (multi-hop where its strategy calls for it), fuses results, and composes **one** grounded, every-sentence-cited answer that covers all parts. The deterministic, rule-based decomposer is the free default; an LLM decomposer is an optional upgrade with the deterministic one as a guaranteed fallback. A single-clause question must degrade **exactly** to today's behavior (byte-identical citations and answer text). The resulting plan is surfaced in `Answer.to_dict()`.

## Why it matters
Today `augur.ask` answers a compound question — "What was the optimal context length, **and** which model family reused the Chinchilla token ratio?" — with one fused retrieval over the whole query string. When the two answers live in **disjoint moments** (different videos, no shared terms), a single RRF pass over the concatenated query under-retrieves the weaker half: the dominant clause's terms crowd the top-k and the second clause's moment never enters the citation set. Decomposing into per-clause sub-queries, retrieving each independently, and merging the citations is the spec's §5 "agentic planner" bet and is what makes multi-part Q&A actually answerable. It is also the prerequisite for M2.3 clip stitching, which stitches the cited Moments this track surfaces.

## Scope (reconciled)
In scope:
- **Deterministic rule-based decomposer** (`augur/planner.py`): split a query into ordered sub-queries on conservative, deterministic boundaries (coordinating "and"/"; "/", " between clauses, "what … and what …", multiple "?"). Single-clause input yields exactly one sub-query that is the original string verbatim.
- **Per-sub-query plan**: each sub-query gets its own `strategy`/`modality`/`contradiction`/`temporal` via the existing keyword classifier (today's `plan()` logic, reused per clause).
- **A `QueryPlan` that carries the sub-queries** (a structured plan object; the legacy single-`QueryPlan` shape is preserved as the degenerate one-sub-query case so existing call sites and tests do not break).
- **Multi-sub-query execution in `augur.ask`**: run `retrieve(...)` once per sub-query (graph leg on for the sub-queries whose strategy is `contradiction`, exactly as today), then **merge** the per-sub-query fused results into one citation list, de-duplicating moments and preserving per-sub-query coverage so every part contributes at least its top moment(s) when available.
- **One composed answer** covering all parts, every sentence cited, reusing the existing extractive free synthesizer and the optional LLM synthesizer (unchanged contracts).
- **Surface the plan in `Answer.to_dict()`**: add a `plan` key (the ordered sub-queries + their per-clause strategy/modality) so REST `/query`, MCP `search_knowledge`, and the SDK expose it. This is **additive** to the dict.
- **Optional LLM decomposer** behind the existing `llm_backend` selection, guarded by `is_generative` and a deterministic try/except fallback to the rule-based decomposer (mirrors the synthesizer's `_synthesize_llm` fallback pattern in `answer.py:141-147`).
- **Eval**: a 2-part golden QA item whose two answers live in **disjoint moments** (talk_a's "optimal context length … 512 tokens" + talk_b's "Llama family … reused the Chinchilla token ratio"); a new `plan.subquery_recall` metric in `eval/harness.py`, landed **ungated** first, gated only once ≥3 stable golden multi-part items exist.

Folded in from the completeness review / non-negotiables:
- **Frozen eval-settings snapshot extension (global discipline (b)):** this track adds a new default-OFF surface (the planner mode / LLM decomposer). The reconciled program requires pinning the growing surface of default-OFF flags, not just `_FREE_BACKENDS`. This track adds the planner's default to the pinned eval-settings snapshot (or, if M0.1/M1.2 has not yet created a dedicated snapshot file, asserts the planner default in a snapshot test alongside `_FREE_BACKENDS`). See W4.
- **Ungated-then-gated metric discipline (global discipline (a)):** `plan.subquery_recall` lands ungated; the existing four gates (hit_rate≥0.6, groundedness≥0.8, contradiction.f1≥0.5, synthesis.groundedness≥0.8) must stay green.
- **Provenance-is-sacred / verification alignment (review risk #4):** the merged citation list must keep premise construction segment/moment-granular so the displayed citation span never drifts from what the groundedness gate verifies. The composed answer must remain every-sentence-cited.

Non-goals / deferrals:
- **Cross-encoder rerank** is M2.1 — this track consumes whatever fused/reranked retrieval M2.1 produces; it does **not** add a rerank stage. Coordinate the citation-build code path with M2.1 (see Coordination note).
- **Answer-with-video clip stitching** is M2.3 (this track blocks it but does not implement it).
- **New retrieval legs / visual leg** are M1.1 — the planner routes `modality='visual'` sub-queries through whatever leg exists; it does not add one.
- **Timeline-anchored retrieval** ("what did the speaker say right after the chain-rule slide?", spec §5) is out of scope; the temporal strategy stays the existing published_at ordering.
- Arbitrarily deep agentic loops / tool-use planning are out of scope: decomposition is one pass (a fixed list of sub-queries), not an open-ended ReAct loop.

## Current state (grounded)
- `augur/planner.py:30-40` — `plan(query)` is a single-pass keyword classifier returning one `QueryPlan(strategy, modality, contradiction, temporal)` (`planner.py:13-18`). It checks four keyword tuples (`_CONTRADICTION`, `_TEMPORAL`, `_PROCEDURE`, `_VISUAL`, `planner.py:21-27`) in priority order and falls through to `hybrid`. There is **no** decomposition — the whole query string is classified as one unit.
- `augur/answer.py:62-152` — `ask(...)` calls `plan_query(query)` once (`answer.py:72`), turns the graph leg on only for `strategy == "contradiction"` (`answer.py:81-86`), calls `retrieve(...)` **once** over the full query (`answer.py:87-90`), builds a citation per fused moment (`answer.py:98-125`), applies temporal re-ordering for the temporal strategy (`answer.py:127-139`), and synthesizes extractively or via LLM with a try/except fallback (`answer.py:141-147`). This is the exact code path the single-clause case must remain byte-identical to.
- `augur/retrieve.py:30-65` — `retrieve(...)` already does dense + lexical (+ optional graph) → `rrf_fuse` and is the per-sub-query execution primitive this track reuses unchanged. `rrf_fuse` (`retrieve.py:18-27`) and `expand` (graph leg, `traverse.py:32-78`) need no changes.
- `augur/types.py:27-40` — `Answer` is a dataclass with `text/citations/strategy/low_evidence`; `to_dict()` (`types.py:34-40`) emits exactly those four keys. There is **no** `plan` field today — adding one is the surfacing change.
- `tests/test_augur.py:17-23` — `TestPlanner.test_intents` asserts `plan(...).strategy`/`.contradiction` on single-clause queries; the new `QueryPlan` shape must keep these passing (the per-clause classifier is the same logic). `TestAsk` (`test_augur.py:34-70`) and `TestStrategyDrivenRetrieval` (`test_augur.py:73-203`) pin the single-clause `ask` behavior (citations, deep-link, contradiction graph routing, temporal ordering) — these are the byte-identical guardrail.
- Call sites of `ask`: SDK `sdk.py:63-69`, CLI `cli.py:48`, REST `server/rest.py:123`, MCP `server/mcp.py:138`. All call `mv.ask(query, video_id=...)` and serialize via `answer.to_dict()` (`rest.py:124`, `mcp.py:138`). The `plan` key flows out for free once `to_dict()` carries it; no call-site signature changes are required.
- `config.py:26-71` — `Settings` has retrieval knobs (`rrf_k`, `top_k`, `contradiction_threshold`) but **no** planner/decomposer field yet. `llm_backend` ("auto") already exists and gates the optional LLM decomposer.
- `eval/harness.py:62-72` — `_FREE_BACKENDS` is the **only** pinned default surface today; there is no dedicated frozen eval-settings snapshot file yet (grep for `snapshot`/`frozen` in `eval/` and `tests/` returns nothing). `_retrieval_and_groundedness` (`harness.py:319-341`) drives QA via `ing.mv.ask(item["q"])`; `qa.json` (`eval/golden/qa.json`) holds 5 single-clause items. Gates live in `harness.py:642-670` (`_HIT_RATE_GATE=0.6`, `_GROUNDEDNESS_GATE=0.8`, `_CONTRADICTION_F1_GATE=0.5`, `_SYNTHESIS_GROUNDEDNESS_GATE=0.8`).
- Golden corpus facts that make a disjoint-moment 2-part item trivial to author: talk_a says "the optimal context length for retrieval was 512 tokens" (talk_a moment), and talk_b says "The Llama family of models reused the Chinchilla token ratio with strong results" (talk_b moment). These are different videos with no shared content terms — a single fused pass biases toward one clause; decomposition recovers both.

## Free-path guarantee
- **Decomposer default is deterministic.** The rule-based decomposer is the free path. The LLM decomposer fires **only** when an LLM backend is configured generative (`llm is not None and getattr(llm, "is_generative", False)`) AND a planner-mode flag is enabled; on any error it falls back to the deterministic decomposer (same pattern as `answer.py:141-147`). Under `_FREE_BACKENDS` (`llm_backend="none"`) the LLM path is never taken.
- **Single-clause = exactly today.** The deterministic decomposer returns `[query]` (the verbatim original string, one sub-query) for any query with no decomposition boundary. In that case `ask` must take the identical code path it takes today: one `plan_query`, one `retrieve(...)` over the full string, the same citation-build loop, the same temporal re-order, the same synthesizer. The contract is **byte-identical** `Answer.text` and `Answer.citations` (every field: index, moment_id, t_start/t_end, modality, speaker, title, deep_link, snippet, score). The W1 red test asserts this byte-for-byte against the golden corpus.
- **`Answer.to_dict()` addition is additive and stable.** The new `plan` key is always present; for a single-clause query it is a one-element list whose sole sub-query equals the original query with the strategy/modality the legacy `plan()` would have returned. Existing consumers that ignore unknown keys are unaffected; the four existing eval gates do not read `plan`.
- **Pinned default.** The planner mode defaults OFF/deterministic and is added to the frozen eval-settings snapshot (W4) so a future default flip cannot silently move gate numbers.

## Workstreams

### W1 — Deterministic decomposer + structured plan, single-clause byte-identical · M
- **Files:** `src/memovox/augur/planner.py` (decomposer + new plan shape), `tests/test_augur.py` (extend `TestPlanner`).
- **Red (failing test first):** in `tests/test_augur.py`, add `TestDecompose`:
  - `test_single_clause_yields_one_subquery`: `decompose("what is attention")` returns one sub-query equal to `"what is attention"` with `strategy == "hybrid"`. Fails today because `decompose` does not exist.
  - `test_multipart_splits_on_and`: `decompose("What was the optimal context length, and which model family reused the Chinchilla token ratio?")` returns **two** sub-queries, the first about context length, the second about the model family, each independently classified. Fails because there is no decomposition.
  - `test_legacy_plan_intents_unchanged`: keep the existing `test_intents` assertions green against the new shape (single-clause classification identical to `planner.py:30-40`).
- **Green (implement):** add `decompose(query) -> QueryPlan` where `QueryPlan` grows a `subqueries: List[SubQuery]` field (each `SubQuery` carries `text`, `strategy`, `modality`, `contradiction`, `temporal`); the top-level `QueryPlan.strategy`/`modality` mirror the first/primary sub-query so legacy reads keep working. Decomposition rule (deterministic, conservative): split only on clear coordinating boundaries — sentence-final `?` between clauses, top-level `;`, `, and `/` and ` joining two interrogative or imperative clauses — never inside quoted spans or obvious noun-phrase "X and Y" lists (guard: only split when **both** resulting fragments contain a content token / a wh-/imperative cue, else keep as one). Reuse the existing keyword tuples per fragment. Single fragment ⇒ `subqueries == [SubQuery(text=query, ...legacy classification...)]`.
- **Verify:** `python -m unittest tests.test_augur -v`; the new decompose tests pass and `test_intents` stays green.
- **Commit:** `feat(augur): deterministic multi-clause query decomposer (spec §5)`

### W2 — Multi-sub-query execution + merged single-answer in `ask` · M
- **Files:** `src/memovox/augur/answer.py`, `tests/test_augur.py` (extend `TestStrategyDrivenRetrieval` / new `TestAgenticAsk`).
- **Red (failing test first):** add `test_multipart_cites_both_disjoint_moments`: build a corpus where clause A's answer moment shares only A's terms and clause B's answer moment shares only B's terms and would fall outside a single fused top-k for the combined query; assert that `ask(...)` over the combined 2-part query cites **both** moments. Add `test_single_clause_byte_identical`: run `ask` over a single-clause query before and after the change is conceptually identical — implement as a golden-fixture assertion that, for each `qa.json` item, the `Answer.text` and the full `[c.to_dict() for c in citations]` list equal a frozen expected snapshot (captured from `main`). Fails today because `ask` runs one fused pass and under-retrieves the disjoint second clause.
- **Green (implement):** in `ask`, call `decompose(query)`. If one sub-query → unchanged code path (guarantee byte-identical). If multiple → for each sub-query call `retrieve(store, sq.text, ..., use_graph=(sq.strategy=='contradiction'), graph_rels=...)`, then **merge** the per-sub-query fused `(moment_id, score)` lists into one ordered, de-duplicated moment list that guarantees per-clause coverage (e.g. interleave each sub-query's top results round-robin, or union with a per-clause minimum, keeping the existing RRF score for ordering; document the deterministic merge rule). Build citations from the merged list with the **same** citation-build loop (`answer.py:98-125`). Apply temporal re-order only if the (single) sub-query is temporal, preserving today's behavior. Synthesize one answer over the merged citations (extractive free path; LLM path unchanged). Keep premise/citation spans moment-granular (review risk #4).
- **Verify:** `python -m unittest tests.test_augur -v`; then `make test` (expect the same total + the new tests). Confirm `TestAsk`, `TestStrategyDrivenRetrieval` stay green (single-clause byte-identical).
- **Commit:** `feat(augur): execute decomposed sub-queries and compose one cited answer (spec §5)`

### W3 — Surface the plan in `Answer.to_dict()` · S
- **Files:** `src/memovox/augur/types.py`, `src/memovox/augur/answer.py` (populate `Answer.plan`), `tests/test_augur.py`, and a serialization assertion in `tests/test_server.py` (or wherever REST/MCP `/query` is tested) if such a test exists.
- **Red (failing test first):** `test_answer_to_dict_carries_plan`: after `ask(...)` on a 2-part query, `ans.to_dict()["plan"]` is a list of two entries, each with `text`/`strategy`/`modality`; for a single-clause query it is a one-element list whose entry matches the legacy classification. Fails because `Answer` has no `plan` field and `to_dict()` (`types.py:34-40`) omits it.
- **Green (implement):** add `plan: List[dict] = field(default_factory=list)` to `Answer` (`types.py:27-32`); add `"plan": self.plan` to `to_dict()` (`types.py:34-40`); populate it in `ask` from the decomposed `QueryPlan.subqueries`. Verify REST `/query` (`rest.py:123-124`) and MCP `search_knowledge` (`mcp.py:138`) now emit `plan` (they serialize `to_dict()` verbatim — no change needed there).
- **Verify:** `python -m unittest tests.test_augur -v` plus any server test; `make test`.
- **Commit:** `feat(augur): surface decomposed query plan in Answer.to_dict (spec §5)`

### W4 — Optional LLM decomposer + frozen eval-settings snapshot pin · S
- **Files:** `src/memovox/augur/planner.py` (LLM decomposer + fallback), `src/memovox/config.py` (planner-mode default, if a flag is needed), `eval/harness.py` (extend the pinned default surface), `tests/test_augur.py`, `tests/` (snapshot/settings test).
- **Red (failing test first):**
  - `test_llm_decomposer_falls_back_on_error`: a fake generative LLM whose decompose call raises must yield the deterministic decomposition (no exception escapes). Fails until the try/except fallback exists.
  - `test_eval_settings_snapshot_pins_planner_default`: assert the frozen eval-settings snapshot includes the planner default (e.g. deterministic / LLM-decomposer OFF) so a default flip is caught. Fails because no snapshot pins it today (only `_FREE_BACKENDS` exists, `harness.py:62-72`).
- **Green (implement):** add an `llm_decompose(llm, query)` that prompts for a JSON list of sub-queries, parses defensively, and on any parse/transport error returns `decompose(query)` (deterministic). Gate it on `is_generative` + a planner-mode flag; under `_FREE_BACKENDS` it is never reached. Extend the pinned eval-settings surface: either add the planner default to `_FREE_BACKENDS` (if expressed as a Settings field) or add a tiny frozen snapshot dict + test that pins it (and is the seed of the program-wide frozen snapshot the reconciled plan calls for).
- **Verify:** `python -m unittest tests.test_augur -v`; `python -m eval.harness --assert-thresholds` (free path numbers unchanged); `make test`.
- **Commit:** `feat(augur): optional LLM query decomposer with deterministic fallback + pin planner default (spec §5)`

### W5 — Golden 2-part item + `plan.subquery_recall` metric (ungated) · M
- **Files:** `eval/golden/qa.json` (add the multi-part item), `eval/harness.py` (the metric + report key), `eval/golden/README.md` (document the new item shape), `tests/test_eval_harness.py` (or the existing harness test) for the metric's pure-function behavior.
- **Red (failing test first):** unit-test `subquery_recall` as a pure function in the harness test (parallel to `hit_rate`): given per-sub-query `(retrieved_ids, relevant_set)` tuples it returns the fraction of sub-queries whose relevant moment is in the retrieved citation set; returns 0.0 on empty input. Then add the 2-part item to `qa.json` with a `subqueries` field, each carrying its own `relevant_moment_substrings`. Fails until the metric and the harness wiring exist.
- **Green (implement):** add the multi-part golden item:
  ```json
  {
    "q": "What was the optimal context length for retrieval, and which model family reused the Chinchilla token ratio?",
    "subqueries": [
      {"q": "optimal context length for retrieval",
       "relevant_moment_substrings": ["the optimal context length for retrieval was 512 tokens"]},
      {"q": "which model family reused the Chinchilla token ratio",
       "relevant_moment_substrings": ["The Llama family of models reused the Chinchilla token ratio"]}
    ],
    "relevant_moment_substrings": ["the optimal context length for retrieval was 512 tokens",
                                   "The Llama family of models reused the Chinchilla token ratio"],
    "answer_substrings": ["512 tokens", "Llama family"]
  }
  ```
  In `harness.py`, for items carrying `subqueries`, compute `subquery_recall`: for each sub-query resolve its gold moment ids via `_relevant_moment_ids` (`harness.py:246-258`) and check whether ANY is in the single composed answer's citation moment ids (`ans.citations`). Average across sub-queries, then across multi-part items, into a new top-level `report["plan"] = {"subquery_recall": ...}`. Keep `_retrieval_and_groundedness` (`harness.py:319-341`) working for legacy single-clause items: items without `subqueries` are scored exactly as today (use the existing `relevant_moment_substrings` for hit_rate; the new union field must not perturb existing items — only the new item is multi-part, and its top-level `relevant_moment_substrings` is the union so hit_rate still credits it on either moment).
- **Verify:** `python -m eval.harness` prints a `plan.subquery_recall` value; `python -m eval.harness --assert-thresholds` still passes the four existing gates (subquery_recall is **not** asserted yet); `make test`.
- **Commit:** `feat(eval): 2-part disjoint-moment golden item + ungated plan.subquery_recall (spec §5/§10)`

> **Coordination note (M2.1 dependency):** M2.1 inserts a cross-encoder rerank stage between fused retrieval and synthesis and touches the same citation-build path. Land M2.2 on top of M2.1's `ask` shape: per-sub-query retrieval should feed the **same** rerank-then-cite path M2.1 establishes (rerank each sub-query's fused results, then merge), not a parallel copy. Confirm with the M2.1 owner whether rerank runs per-sub-query (preferred — rerank sees a focused clause) or once over the merged list; either is acceptable but must be decided jointly so the byte-identical single-clause guarantee holds for both tracks.

## Eval gate
- **New metric:** `plan.subquery_recall` — for each multi-part golden item, the fraction of its sub-queries whose gold moment is present in the single composed answer's citations, averaged over items. Lands **UNGATED** (computed and printed, not asserted) per global discipline (a), because it starts as exactly one hand-authored item. It is promoted to a gate only once **≥3 stable multi-part golden items** exist; the eventual threshold target is `subquery_recall ≥ 1.0` (every part covered) once the corpus is large enough to be a stable signal — confirm the threshold at promotion time.
- **Byte-identical guard (this track's real proof today):** the W2 single-clause snapshot test asserts `Answer.text` + full citation dicts are byte-identical to the pre-change `main` output for every `qa.json` single-clause item.
- **Existing gates stay green:** `python -m eval.harness --assert-thresholds` must still pass `retrieval.hit_rate ≥ 0.6`, `groundedness ≥ 0.8`, `contradiction.f1 ≥ 0.5`, `synthesis.groundedness ≥ 0.8` (`harness.py:642-670`). The new union-`relevant_moment_substrings` on the 2-part item keeps `hit_rate` creditable; the multi-part item must not regress the four single-clause items.

## Risks & mitigations
- **Over-eager decomposition splits noun-phrase "X and Y" lists** (e.g. "scaling laws and compute") into junk sub-queries, regressing single-intent queries. *Mitigation:* conservative split rule (both fragments must independently look like queries — wh-word / imperative / content token); the byte-identical golden snapshot test (W2) catches any single-clause regression immediately; default to one sub-query on ambiguity.
- **Merge logic re-orders/duplicates citations and breaks the byte-identical guarantee** for single-clause. *Mitigation:* the single-sub-query branch takes the **literal** unchanged code path (early return before any merge), proven by the snapshot test.
- **Citation-span drift vs the verification gate (review risk #4).** Merging multiple sub-query result sets must not change how premise spans are built. *Mitigation:* reuse the existing per-moment citation-build loop unchanged; keep premise = the cited moment(s)' text exactly as `_answer_groundedness` (`harness.py:276-316`) expects; assert groundedness gate stays ≥0.8.
- **Eval thinness (review risk #1).** One multi-part item is not a stable gate. *Mitigation:* ungated-then-gated; grow to ≥3 items before gating; cover decompose logic by unit tests, not the golden gate.
- **Determinism erosion from the new default-OFF planner flag (review risk #2).** *Mitigation:* pin the planner default in the frozen eval-settings snapshot (W4), not just `_FREE_BACKENDS`.
- **Collision with M2.1 on `ask` / citation build.** *Mitigation:* the Coordination note — land on M2.1's rerank-then-cite path; decide per-sub-query vs merged rerank jointly.
- **LLM decomposer non-determinism / bad JSON.** *Mitigation:* default OFF, `is_generative`-gated, defensive parse, guaranteed deterministic fallback (W4 red test asserts the fallback).

## Definition of done
- [ ] `decompose(query)` exists; single-clause ⇒ one verbatim sub-query; multi-part ⇒ ordered per-clause sub-queries with per-clause strategy/modality.
- [ ] `ask` executes each sub-query through the existing legs (graph leg on for contradiction sub-queries) and composes ONE every-sentence-cited answer covering all parts.
- [ ] Single-clause `ask` output is **byte-identical** to `main` (snapshot test green over all `qa.json` items).
- [ ] `Answer.to_dict()` carries an additive `plan` key; REST/MCP/SDK expose it without call-site changes.
- [ ] Optional LLM decomposer is `is_generative`-gated with a guaranteed deterministic fallback (fallback test green).
- [ ] A 2-part disjoint-moment golden item exists; `plan.subquery_recall` is computed and printed, **ungated**.
- [ ] Planner default pinned in the frozen eval-settings snapshot.
- [ ] `python -m eval.harness --assert-thresholds` passes all four existing gates; `make test` passes (prior 247 pass / 2 skip plus the new tests).

## Open questions
- **Merge policy for disjoint sub-query results:** round-robin interleave vs union-with-per-clause-minimum vs re-RRF over the per-sub-query fused lists. Recommend round-robin top-k per sub-query (simplest deterministic per-clause coverage); confirm before W2.
- **Per-sub-query vs merged rerank with M2.1** (see Coordination note) — needs a joint decision with the M2.1 owner.
- **Planner-mode flag surface:** is the LLM decomposer toggled by a new `Settings` field (e.g. `planner_mode`) or implicitly by `llm_backend` being generative? A dedicated default-OFF flag is cleaner to pin in the snapshot; confirm whether to add a `Settings` field vs piggy-back on `llm_backend`.
- **`subquery_recall` promotion threshold** (1.0 vs ≥ first-relevant-in-top-k) at the ≥3-item milestone — confirm at gate-promotion time.
