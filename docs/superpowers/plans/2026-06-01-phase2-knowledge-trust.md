# Phase 2 — Knowledge + Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn memovox's knowledge layer from a per-video claim store into a *verified, cross-corpus knowledge graph* — entities and speakers resolved library-wide, claims linked by typed edges, retrievable by multi-hop graph traversal, and measurable by an eval harness.

**Architecture:** Build on the existing SQLite store (relational + FTS5 + vectors + edge-table graph) and the existing Assay claim pipeline. Add a new `loom/resolve.py` (entity + speaker resolution), extend Assay to bind claims to their *exact* source span and emit entity mentions, add typed claim→claim edges (`ELABORATES`/`CORRECTS`) and a graph leg + multi-hop traversal to Augur, and stand up a minimal `eval/` harness with a golden set so resolution quality is measured (spec §10), not asserted. Every new model slot keeps a deterministic stdlib fallback with an optional upgrade, per house style.

**Tech Stack:** Python stdlib (sqlite3, urllib, difflib, dataclasses); existing memovox backends; optional upgrades behind interfaces — `pyannote.audio` (voiceprints), Wikidata REST (entity grounding), an LLM backend (richer extraction). No new required dependencies.

---

## Current state (what this plan completes)

Phase 2 is **partially scaffolded** by earlier commits. The audit (adversarially verified) established:

| Area | State today | Evidence |
|------|-------------|----------|
| Claim extraction + epistemic typing + salience | ✅ works (rule-based + optional LLM) | `assay/claims.py` |
| NLI verification gate | ⚠️ wired but **near-vacuous on the default path** — claims are verbatim sentences and the premise is the whole Moment, so overlap ≈ 1.0 and nothing is ever rejected | `assay/__init__.py:28-31`, `backends/nli.py:46` |
| Claim provenance span | ⚠️ bound to the **whole Moment**, not the "exact source span" the spec requires | `assay/claims.py:104-105` |
| Graph edge table + provenance | ✅ table real; **only 4 of 9 edge types emitted** (`PRECEDES`, `STATES`, `ATTRIBUTED_TO`, `CONTRADICTS`); `SUPPORTS` dead-by-default; `ELABORATES`/`CORRECTS`/`MENTIONS`/`ABOUT` never created | `store.py:376-401`, `pipeline.py:122-141` |
| Entity resolution | ❌ **dead code** — `upsert_entity`/`link_mention`/`entity_mentions` have zero callers; no `Entity` is ever constructed; no `MENTIONS` edge | `store.py:351-365` (grep: no callers) |
| Cross-video speaker resolution | ❌ **the inverse of the spec** — speakers are namespaced **per-video** (`{video_id}:{speaker}`); `voiceprint_ref` always null | `pipeline.py:39-43,105-110` |
| Graph retrieval / multi-hop | ❌ **missing** — `store.neighbors()`/`store.edges()` never called from `augur/`; planner's `temporal`/`contradiction`/`visual` strategies are dead metadata never consumed by `retrieve()` | `augur/retrieve.py` (grep: no `neighbors`/`strategy`) |
| Eval harness | ❌ **absent (0%)** — no golden set, no metrics, no CI; yet "eval-driven" is a non-negotiable (§2) and §10's Phase-2 metrics are entity-resolution F1 + DER | repo has no `eval/`, no `.github/` |

**Spec sections covered:** §4.5 (Assay), §4.6 (Loom resolution + indexing), §5 (graph retrieval / multi-hop), §6 (Entity/Speaker/edge data model), §10 (eval metrics), §11 (Phase 2 deliverable), §12 (cross-video diarization risk).

## Design principles to honor (spec §2)

- **Provenance is sacred** — every new edge/entity/mention carries `(video_id, t_start, t_end, modality, confidence)`; entity/speaker resolution must never produce an orphan or silently merge across a confidence floor.
- **Verify before commit** — the trust pass must be able to *actually reject*; prove it with a test that an unsupported (hallucinated) claim is flagged.
- **Idempotent ingestion** — resolution is re-runnable: deterministic canonical IDs (`ent:<slug>`, `spk:<slug>`); re-ingest/re-resolve never duplicates.
- **Model-agnostic / local-first** — voiceprints (pyannote), Wikidata grounding, and LLM extraction are optional upgrades behind interfaces; the free path resolves by name/lexical similarity.
- **Eval-driven** — nothing in this phase is "done" until a metric moves; W0 builds the measuring stick first.

## File structure

```
src/memovox/
  loom/
    resolve.py        CREATE  entity + speaker resolution (canonicalization, clustering)
    models.py         MODIFY  add Mention dataclass + canonical_speaker_id field on Speaker
    store.py          MODIFY  add mentions readback, canonical-speaker queries, MENTIONS/ABOUT edge helpers, supersede writer
  assay/
    claims.py         MODIFY  exact source-span binding; entity-mention extraction
    spans.py          CREATE  map a claim sentence back to its word/segment-level (t_start,t_end)
    verify.py         MODIFY  verify against the claim's own span, not the whole Moment
  backends/
    entity_link.py    CREATE  EntityLinker interface: NullLinker (slug) | WikidataLinker (optional, urllib)
    diarize_voiceprint.py CREATE optional pyannote voiceprint backend (graceful absence)
  augur/
    retrieve.py       MODIFY  add graph-expansion leg fused into RRF
    traverse.py       CREATE  multi-hop traversal over SUPPORTS/CONTRADICTS/ELABORATES
    answer.py         MODIFY  consume plan.strategy; use graph traversal for synthesis questions
  escapement/
    fusion.py         MODIFY  retain per-segment offsets on the Moment for exact-span lookup
  pipeline.py         MODIFY  call resolve_entities + resolve_speakers; emit MENTIONS/ELABORATES/CORRECTS; replace per-video speaker namespacing with canonical resolution
eval/
  __init__.py         CREATE
  harness.py          CREATE  metric runners: retrieval (hit/MRR/nDCG), groundedness, entity-F1, DER-lite, contradiction P/R
  golden/             CREATE  small committed golden corpus + labels (transcripts + QA + entity/speaker keys)
    *.en.vtt, qa.json, entities.json, speakers.json, contradictions.json
tests/
  test_resolve.py     CREATE
  test_spans.py       CREATE
  test_graph_retrieval.py CREATE
  test_eval.py        CREATE
  (extend test_assay.py, test_loom.py, test_augur.py, test_integration.py)
.github/workflows/ci.yml CREATE  run `pytest` (hermetic) + eval thresholds
```

## Build order & rationale

1. **W0 — Eval scaffold + hermetic tests.** Build the measuring stick first so every later workstream is verified, not asserted. Also fixes the non-hermetic `test_auto_*` failures so CI is trustworthy.
2. **W1 — Trust hardening.** Exact source-span binding + a gate that can reject. This is the "trust" half of "Knowledge + trust" and must precede piling a graph on top.
3. **W2 — Entity resolution.** Wire the dead code → real `Entity` nodes + `MENTIONS` edges + cross-corpus canonicalization. Measured by entity-F1 from W0.
4. **W3 — Graph edges + retrieval + multi-hop.** Create `ELABORATES`/`CORRECTS`; add the graph leg to retrieval and traversal to synthesis; wire the dead planner routes. Measured by retrieval metrics from W0.
5. **W4 — Cross-video speaker resolution.** Name/metadata clustering (free) + optional voiceprints, behind the DER eval (§12 risk).

Each workstream ships independently and leaves the suite green. Commit after every task.

---

## W0 — Eval scaffold + hermetic tests

### Task 0.1: Pin the test suite to deterministic fallbacks (hermetic)

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_backends.py:` the `test_auto_embedder_is_available` / `test_auto_nli_is_available` cases

- [ ] **Step 1: Write `tests/conftest.py` forcing fallbacks for the whole suite**

```python
import os
# Hermetic tests: never resolve to network-backed model downloads.
os.environ.setdefault("MEMOVOX_EMBED_BACKEND", "hashing")
os.environ.setdefault("MEMOVOX_NLI_BACKEND", "lexical")
os.environ.setdefault("MEMOVOX_ASR_BACKEND", "captions")
os.environ.setdefault("MEMOVOX_LLM_BACKEND", "none")
os.environ.setdefault("MEMOVOX_VLM_BACKEND", "none")
os.environ.setdefault("MEMOVOX_OCR_BACKEND", "none")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
```

- [ ] **Step 2: Make the two `auto` tests assert behavior, not a network model**

Rewrite them to assert that `get_embedder("auto")`/`get_nli("auto")` return a backend whose `name` is in the known set and whose `is_available()` is True for the *resolved* backend — without instantiating a model that downloads. Example:

```python
def test_auto_embedder_resolves_to_available_backend(self):
    emb = get_embedder("auto")
    self.assertIn(emb.name, {"hashing", "sentence-transformers"})
    # the free fallback must always be importable & usable offline
    self.assertEqual(get_embedder("hashing").embed_one("x").__len__(), 256)
```

- [ ] **Step 3: Run** `python -m pytest tests/test_backends.py -q` → Expected: PASS (no network).
- [ ] **Step 4: Run the full suite** `python -m pytest -q` → Expected: all pass (0 network failures).
- [ ] **Step 5: Commit** `git commit -m "test: make suite hermetic (pin fallbacks, no network in CI)"`

### Task 0.2: Golden corpus

**Files:**
- Create: `eval/golden/talk_a.en.vtt`, `eval/golden/talk_b.en.vtt` (2–3 short transcripts; `talk_b` deliberately contradicts `talk_a` on one claim and reuses an entity/speaker name)
- Create: `eval/golden/qa.json`, `entities.json`, `speakers.json`, `contradictions.json`

- [ ] **Step 1: Author the transcripts and labels.** `qa.json` items: `{ "q": str, "relevant_moment_substrings": [str], "answer_substrings": [str] }`. `entities.json`: `{ "canonical": [str], "mentions": {video_id: [str]} }`. `speakers.json`: `{ "same_person_across": [[video_id, raw_label], ...] }`. `contradictions.json`: `[{ "topic": str, "video_a": id, "video_b": id }]`.
- [ ] **Step 2:** No test yet — this is data. Commit. `git commit -m "test(eval): add golden corpus + labels"`

### Task 0.3: Eval harness with metric runners

**Files:**
- Create: `eval/__init__.py`, `eval/harness.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test** (`tests/test_eval.py`)

```python
def test_retrieval_metrics_on_golden_corpus():
    report = run_eval(golden_dir=GOLDEN, store=fresh_store_ingested_with_golden())
    assert report["retrieval"]["hit_rate"] >= 0.6      # baseline gate
    assert report["groundedness"] >= 0.8
    assert "entity_f1" in report and "der" in report and "contradiction" in report
```

- [ ] **Step 2: Run** `pytest tests/test_eval.py -q` → FAIL: `run_eval` undefined.

- [ ] **Step 3: Implement `eval/harness.py`** — pure-stdlib metrics:

```python
def hit_rate(retrieved_ids, relevant_ids): ...     # @k
def mrr(retrieved, relevant): ...
def ndcg(retrieved, relevant, k): ...
def groundedness(answer_sentences, cited_spans, nli): ...  # % entailed
def clustering_f1(pred_clusters, gold_clusters): ...       # pairwise F1 for entity/speaker res
def contradiction_pr(found_pairs, gold_pairs): ...

def run_eval(golden_dir, store) -> dict:
    # ingest golden vtts into a temp store, run ask() over qa.json,
    # compare to labels, compute the table above. Return a dict report.
```

- [ ] **Step 4: Run** `pytest tests/test_eval.py -q` → PASS.
- [ ] **Step 5: Add `make eval`** target (`$(PY) -m eval.harness`) printing the report.
- [ ] **Step 6: Commit** `git commit -m "feat(eval): minimal eval harness + golden-corpus metrics (spec §10)"`

### Task 0.4: CI gate

**Files:** Create `.github/workflows/ci.yml`

- [ ] **Step 1:** Workflow: checkout → `pip install -e '.[dev]'` → `pytest -q` → `python -m eval.harness --assert-thresholds`. Hermetic env from conftest applies.
- [ ] **Step 2: Commit** `git commit -m "ci: gate on hermetic tests + eval thresholds"`

---

## W1 — Trust hardening (exact source span + a gate that can reject)

### Task 1.1: Retain per-segment offsets on Moments

**Files:**
- Modify: `src/memovox/loom/models.py` (add `segments: List[SegmentRef]` to `Moment`, default `[]`, `SegmentRef = (t_start_s, t_end_s, text)`)
- Modify: `src/memovox/escapement/fusion.py` (`_make_moment` records the spans it merged)
- Test: `tests/test_escapement.py`

- [ ] **Step 1: Failing test** — a Moment built from 3 segments exposes 3 `segments` with their original `(t_start,t_end,text)`.

```python
def test_moment_retains_segment_offsets():
    segs = [seg(0,5,"alpha"), seg(5,10,"beta"), seg(10,15,"gamma")]
    m = build_moments(VID, segs)[0]
    assert [(s.t_start_s, s.t_end_s) for s in m.segments] == [(0,5),(5,10),(10,15)]
```

- [ ] **Step 2: Run** → FAIL (`Moment` has no `segments`).
- [ ] **Step 3: Implement** — add `segments` field to `Moment`; in `_make_moment`, populate it from `segs`. (Not persisted to the moments table — it is a build-time artifact consumed by Assay in the same pipeline run.)
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(escapement): retain per-segment offsets on Moments for exact-span claims"`

### Task 1.2: Bind claims to their exact source span

**Files:**
- Create: `src/memovox/assay/spans.py`
- Modify: `src/memovox/assay/claims.py` (`_extract_rule_based` and `_extract_with_llm` set `t_start_s/t_end_s` from the matched span)
- Test: `tests/test_spans.py`, extend `tests/test_assay.py`

- [ ] **Step 1: Failing test** (`tests/test_spans.py`)

```python
def test_locate_span_returns_segment_window_for_a_sentence():
    segments = [(0.0, 5.0, "Neural nets learn by backprop."),
                (5.0, 12.0, "The chain rule is central.")]
    assert locate_span("The chain rule is central.", segments) == (5.0, 12.0)

def test_locate_span_falls_back_to_full_when_unmatched():
    assert locate_span("unrelated", [(0.0, 9.0, "alpha beta")], default=(0.0, 9.0)) == (0.0, 9.0)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement `spans.py`** — token-overlap match of a claim sentence to the segment(s) whose text best contains it; return `(min t_start, max t_end)` of the matched segment span; fall back to `default` (the whole-Moment span) when no segment clears a 0.5 overlap floor.

```python
from ..util import tokenize
def locate_span(sentence, segments, *, default=None):
    s = set(tokenize(sentence))
    if not s or not segments:
        return default
    best, best_ov = None, 0.0
    for (t0, t1, text) in segments:
        ov = sum(1 for w in tokenize(text) if w in s) / max(1, len(s))
        if ov > best_ov:
            best, best_ov = (t0, t1), ov
    return best if best_ov >= 0.5 else default
```

- [ ] **Step 4: Wire into `claims.py`** — both extractors call `locate_span(sentence, moment.segments, default=(moment.t_start_s, moment.t_end_s))` and set the claim span from it.
- [ ] **Step 5: Failing test in `test_assay.py`** — a Moment spanning 0–30 with the claim sentence living in segment 20–30 yields a claim with `t_start_s==20, t_end_s==30` (not 0–30).
- [ ] **Step 6: Run** → PASS.
- [ ] **Step 7: Commit** `git commit -m "feat(assay): bind claims to exact source span (spec §4.5 provenance)"`

### Task 1.3: Verify against the claim's own span + prove rejection

**Files:**
- Modify: `src/memovox/assay/verify.py`, `src/memovox/assay/__init__.py` (pass the per-claim source text, not the whole Moment)
- Test: extend `tests/test_assay.py`

- [ ] **Step 1: Failing test — the gate must reject a hallucinated LLM claim.** Use a stub `LLMBackend` whose extraction emits one verbatim claim and one fabricated claim absent from the span; assert the fabricated one is `status == "unsupported"` and the real one is `committed`.

```python
class _HallucinatingLLM(LLMBackend):
    is_generative = True
    def complete(self, prompt, **kw):
        return '[{"text":"The chain rule is central.","type":"FACT"},'\
               ' {"text":"Quantum tunneling powers the optimizer.","type":"FACT"}]'

def test_gate_rejects_unsupported_llm_claim():
    m = Moment("v#m0","v",0,12,"The chain rule is central.", segments=[(0,12,"The chain rule is central.")])
    claims = assay.run(m, nli=get_nli("lexical"), llm=_HallucinatingLLM())
    by_text = {c.text: c.status for c in claims}
    assert by_text["The chain rule is central."] == "committed"
    assert by_text["Quantum tunneling powers the optimizer."] == "unsupported"
```

- [ ] **Step 2: Run** → FAIL (today the premise is the whole Moment text and the lexical gate is lenient; verify per-claim source span instead).
- [ ] **Step 3: Implement** — in `assay.run`, build per-claim premise from the claim's own span text (look up the segment text for `[claim.t_start_s, claim.t_end_s]`), and verify against that. Keep the threshold in `Settings.entailment_threshold`.
- [ ] **Step 4: Run** → PASS (real claim entailed by its span; fabricated claim not).
- [ ] **Step 5: Commit** `git commit -m "feat(assay): verify claims against their own span; gate now rejects hallucinations"`

### Task 1.4: Implement the supersede lifecycle (close the dead `superseded_by`)

**Files:**
- Modify: `src/memovox/loom/store.py` (add `supersede_claim(old_id, new_id)` writing `status='superseded'` + `superseded_by`)
- Test: extend `tests/test_loom.py`

- [ ] **Step 1: Failing test** — `supersede_claim("c_old","c_new")` sets `c_old.status=="superseded"` and `c_old.superseded_by=="c_new"`, and `c_old` is excluded from default committed queries.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `supersede_claim`. (Callers wired in W2/Phase 3; this closes the §2 "never silently deleted / versioned supersede" invariant the audit flagged.)
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(loom): implement claim supersede lifecycle (versioned, never deleted)"`

---

## W2 — Entity resolution (wire the dead code → real graph)

### Task 2.1: Entity-mention extraction

**Files:**
- Modify: `src/memovox/assay/claims.py` (add `extract_mentions(claim) -> List[str]`)
- Test: extend `tests/test_assay.py`

- [ ] **Step 1: Failing test** — `extract_mentions` on a claim about "BERT and the Transformer architecture by Vaswani" returns `["BERT","Transformer","Vaswani"]` (proper-noun phrases + capitalized runs; lower-cased stopword/common words excluded).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — regex over capitalized token runs in `claim.text`/`subject`/`object`, plus all-caps acronyms; drop sentence-initial-only capitals that are common words (a small stoplist). Return distinct surface forms.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(assay): extract entity mentions from claims"`

### Task 2.2: Entity linker backend (NullLinker + optional Wikidata)

**Files:**
- Create: `src/memovox/backends/entity_link.py`
- Modify: `src/memovox/backends/__init__.py` (`get_entity_linker`, status)
- Test: `tests/test_resolve.py`

- [ ] **Step 1: Failing test** — `get_entity_linker("none").canonicalize("Transformer")` returns `Canonical(entity_id="ent:transformer", name="Transformer", wikidata_qid=None)`; same surface form → same id across calls (deterministic/idempotent).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — `EntityLinker` ABC with `canonicalize(surface) -> Canonical`. `NullLinker`: normalized slug id, no QID. `WikidataLinker` (optional, `is_available()` checks online): query `https://www.wikidata.org/w/api.php?action=wbsearchentities` via urllib, attach `wikidata_qid` + canonical label, **graceful fallback to slug on any network/parse error**.
- [ ] **Step 4: Run** → PASS (Null path; Wikidata path covered by a skipUnless-online test).
- [ ] **Step 5: Commit** `git commit -m "feat(backends): entity-linker interface (slug fallback, optional Wikidata)"`

### Task 2.3: Cross-corpus entity resolution

**Files:**
- Create: `src/memovox/loom/resolve.py` (`resolve_entities(store, claims, *, linker)`)
- Modify: `src/memovox/loom/store.py` (add `mentions_for_entity`, `get_entity`, `MENTIONS` edge in `link_mention`)
- Test: extend `tests/test_resolve.py`, `tests/test_loom.py`

- [ ] **Step 1: Failing test** — resolving claims from two videos that both mention "Transformer" produces ONE `Entity("ent:transformer")` with a `MENTIONS` edge from each claim, and `store.neighbors(claim_id, rel="MENTIONS")` returns it.

```python
def test_same_entity_across_videos_is_one_node():
    # ingest two golden videos mentioning "Transformer"
    ents = store.list_entities()
    transformer = [e for e in ents if e.entity_id == "ent:transformer"]
    assert len(transformer) == 1
    mids = store.mentions_for_entity("ent:transformer")
    assert len({m.video_id for m in store.get_claims(mids)}) == 2
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement `resolve_entities`** — for each committed claim: `extract_mentions` → `linker.canonicalize` → `store.upsert_entity(Entity(...))` (idempotent), `store.link_mention(claim_id, entity_id)`, and `store.add_edge(claim_id, "MENTIONS", entity_id, src_type="Claim", dst_type="Entity", video_id=claim.video_id, t_start_s=claim.t_start_s, t_end_s=claim.t_end_s)`. Same normalized surface → same `entity_id` across the corpus → cross-video unification for free.
- [ ] **Step 4: Wire into `pipeline.py`** after claims are committed: `resolve_entities(store, committed_claims, linker=get_entity_linker(settings.entity_backend))`. Add `entity_backend="auto"` to Settings (auto → Wikidata if online else Null).
- [ ] **Step 5: Run** → PASS; add an idempotency test (re-ingest → entity count stable).
- [ ] **Step 6: Eval** — `run_eval` `entity_f1` must clear the W0 gate; record the number.
- [ ] **Step 7: Commit** `git commit -m "feat(loom): cross-corpus entity resolution + MENTIONS edges (spec §4.6)"`

---

## W3 — Typed edges + graph retrieval + multi-hop

### Task 3.1: Create `ELABORATES` and `CORRECTS` edges

**Files:**
- Modify: `src/memovox/pipeline.py` (after claims committed, link claim→claim within a video)
- Create helper: `src/memovox/loom/resolve.py::link_claim_relations(store, claims)`
- Test: extend `tests/test_loom.py`

- [ ] **Step 1: Failing test** — consecutive claims in the same Moment/topic get an `ELABORATES` edge; a claim typed `CORRECTION` gets a `CORRECTS` edge to the most recent prior claim sharing an entity/subject.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — `ELABORATES`: claim N → claim N+1 when same speaker and adjacent in time within a Moment. `CORRECTS`: a `CORRECTION`-typed claim → the nearest prior claim sharing a subject/entity. Provenance-stamped via `add_edge`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(loom): emit ELABORATES/CORRECTS claim edges (spec §6)"`

### Task 3.2: Graph-expansion retrieval leg

**Files:**
- Create: `src/memovox/augur/traverse.py`
- Modify: `src/memovox/augur/retrieve.py` (fuse a graph leg into RRF)
- Test: `tests/test_graph_retrieval.py`

- [ ] **Step 1: Failing test** — given seed moments from dense+lexical, `retrieve(..., use_graph=True)` also returns moments reachable via `SUPPORTS`/`CONTRADICTS`/`ELABORATES` from claims in the seed moments (a moment that shares no query terms but is graph-linked is surfaced).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement `traverse.py`** — `expand(store, seed_moment_ids, *, rels, hops=1) -> List[(moment_id, score)]`: for each seed moment, find its claims, follow `neighbors(claim_id, rel)` for `rel in rels`, map neighbor claims back to their moments, score by hop distance. In `retrieve()`, add `graph = expand(store, [mid for mid,_ in rrf_fuse([dense,lexical])], ...)` and fuse `[dense, lexical, graph]`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(augur): graph-expansion retrieval leg fused into RRF (spec §5)"`

### Task 3.3: Consume the planner; multi-hop synthesis answers

**Files:**
- Modify: `src/memovox/augur/answer.py` (use `qp.strategy` to choose retrieval mode + traversal depth)
- Test: extend `tests/test_augur.py`

- [ ] **Step 1: Failing test** — a `contradiction` query routes through the graph leg and returns citations spanning both sides of a `CONTRADICTS` edge; a `temporal` query orders claim citations by `published_at`; assert the planner's strategy is actually reflected in `Answer.strategy` *and* in which moments come back (today it is decorative).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — branch on `qp`: `contradiction`/`temporal`/`hybrid` set `use_graph` and traversal `rels`/ordering; `temporal` sorts citations by the video's `published_at`. Keep the extractive synthesizer; every sentence still cites.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Eval** — retrieval `hit_rate`/`mrr` must not regress; contradiction P/R recorded.
- [ ] **Step 6: Commit** `git commit -m "feat(augur): planner-driven graph retrieval + multi-hop synthesis (spec §5)"`

---

## W4 — Cross-video speaker resolution (behind the DER eval, spec §12)

### Task 4.1: Canonical speaker model + name-based resolution (free path)

**Files:**
- Modify: `src/memovox/loom/models.py` (`Speaker.canonical_id`), `src/memovox/loom/store.py` (`canonical_speaker`, `SAME_AS` edge), `src/memovox/loom/resolve.py` (`resolve_speakers`)
- Modify: `src/memovox/pipeline.py` (call `resolve_speakers` after upserting per-video speakers; stop relying on per-video namespacing as the final identity)
- Test: extend `tests/test_resolve.py`

- [ ] **Step 1: Failing test** — two videos whose speakers resolve to the name "Dr. Lee" map to one canonical `spk:dr-lee`; `store.canonical_speaker("vidA:Dr. Lee") == store.canonical_speaker("vidB:Dr. Lee")`; per-video provenance is preserved (the original per-video speaker rows still exist, linked by `SAME_AS`).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — `resolve_speakers(store)`: group per-video speakers by normalized `resolved_name` (fuzzy via `difflib.SequenceMatcher` above a threshold); create canonical `spk:<slug>`; add `(per_video_speaker)-[:SAME_AS]->(canonical)`; never merge anonymous `spk_0` across videos (§12 conservatism). Set `Speaker.canonical_id`.
- [ ] **Step 4: Run** → PASS; add a negative test (two different names never merge; anonymous speakers never merge).
- [ ] **Step 5: Eval** — DER-lite on `speakers.json` recorded; gate that name-based resolution ≥ baseline.
- [ ] **Step 6: Commit** `git commit -m "feat(loom): cross-video speaker resolution by name (conservative, spec §4.6/§12)"`

### Task 4.2: Optional voiceprint clustering

**Files:**
- Create: `src/memovox/backends/diarize_voiceprint.py` (pyannote-backed; `is_available()` guards)
- Modify: `src/memovox/loom/resolve.py` (use voiceprints to merge same-voice speakers when names are absent/ambiguous)
- Test: `tests/test_resolve.py` (skipUnless pyannote installed)

- [ ] **Step 1: Failing test** (skipUnless) — with synthetic embeddings injected, two anonymous speakers with near-identical voiceprints cluster; far ones do not.
- [ ] **Step 2: Run** → FAIL/skip.
- [ ] **Step 3: Implement** — cosine clustering of voiceprints above a threshold, gated to run only when the optional backend is present; the free path is untouched.
- [ ] **Step 4: Run** → PASS/skip.
- [ ] **Step 5: Commit** `git commit -m "feat(backends): optional voiceprint clustering for speaker resolution"`

---

## Self-review (spec coverage)

| Spec requirement | Task(s) | Covered? |
|---|---|---|
| §4.5 atomic claims tied to **exact source span** | 1.1, 1.2 | ✅ |
| §4.5 NLI gate that actually rejects | 1.3 | ✅ |
| §4.5 salience + epistemic typing | (already built) | ✅ |
| §4.6 entity resolution + canonicalization + Wikidata (optional) | 2.1–2.3 | ✅ |
| §4.6 cross-video speaker resolution + voiceprints (optional) | 4.1, 4.2 | ✅ |
| §6 `MENTIONS`/`ELABORATES`/`CORRECTS` edges | 2.3, 3.1 | ✅ |
| §6 supersede lifecycle (versioned, never deleted) | 1.4 | ✅ |
| §5 graph retrieval (dense+sparse+**graph** in RRF) | 3.2 | ✅ |
| §5 multi-hop traversal of SUPPORTS/CONTRADICTS/ELABORATES | 3.2, 3.3 | ✅ |
| §5 planner strategy actually consumed | 3.3 | ✅ |
| §10 retrieval (hit/MRR/nDCG), groundedness, entity-F1, DER, contradiction P/R | 0.2, 0.3 | ✅ |
| §2 eval-driven (CI gate) | 0.1, 0.4 | ✅ |
| §12 cross-video diarization risk (conservative, eval-gated) | 4.1 | ✅ |

**Known deferrals (NOT in this plan):** `ABOUT`/topic edges + topic induction (Phase 3); consensus scoring + claim-evolution ordering as a first-class feature (Phase 3); ColPali visual-vector retrieval leg (Phase 4); the agentic multi-step planner (Phase 4). These are listed in "Beyond Phase 2".

## Risks & mitigations

- **Cross-video diarization (§12).** Voiceprint drift + name collisions. Mitigation: name/metadata merge first; voiceprints optional and threshold-tuned; never merge anonymous speakers; gate every change on DER-lite from W0.
- **Entity over/under-merging.** Slug normalization can collide ("Mercury" the planet vs element) or fragment ("GPT-4" vs "GPT 4"). Mitigation: Wikidata QID disambiguation when online; eval `entity_f1` gate; keep surface forms as `aliases`.
- **Exact-span matching is heuristic** without word timestamps. Mitigation: 0.5 overlap floor with whole-Moment fallback; precision improves automatically on the WhisperX/faster-whisper word-timestamp path.
- **Graph leg latency.** Multi-hop expansion adds queries. Mitigation: `hops=1` default, bounded fan-out, indexed `edges(src/dst/rel)` (already indexed in `store.py`).

## Beyond Phase 2 (documented for completeness)

- **Phase 3 — Synthesis (Loom, async):** topic induction + `ABOUT` edges; consensus scoring (source count × recency × authority); claim-evolution ordering by `published_at` as a first-class query; dedup/decay using the supersede lifecycle from Task 1.4; move consolidation to a background job.
- **Phase 4 — Scale & polish:** channel/playlist subscriptions + incremental sync (beyond today's flat re-ingest); answer-with-video clip stitching; ColPali multi-vector visual embeddings + a visual late-interaction retrieval leg in Augur (Tessera already stores per-keyframe visual vectors — wire them as a retrieval modality here); decay/versioning dashboards; the named production backends (Qdrant/Kùzu/Tantivy-SPLADE) behind the existing interfaces.
- **Cross-cutting:** wire the real ASR stack (WhisperX word alignment + pyannote) so exact spans and voiceprints reach full fidelity; expand the golden set; add per-stage observability/metrics (§9).

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-01-phase2-knowledge-trust.md`. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints for review.

Which approach? (Or stop here — this is the requested forward-work document; no code has been written.)
