# M1.1 — Visual retrieval leg + named visual defaults

> **Wave:** 1 · **Effort:** L · **Status:** not started
> **Depends on:** M0.2 (storage-backend interface — `visual_search` is wrapped once, behind the `VectorIndex` ABC) · **Owns (single-owner concerns):** the visual golden fixtures + the `visual_result` injection seam on `pipeline.ingest` (M1.2 consumes both — do NOT duplicate) · **Blocks:** M1.2
> **Spec:** §4.3 (Stage 3 visual track / Stage 6 triple write), §5 (hybrid retrieval + planner), §7 (named backends), §9 (throughput — the visual track is THE bottleneck), §11 (Phase 4: ColPali visual retrieval), §12 (ColPali storage cost; ToS posture)

## Goal
Light up the **fourth retrieval leg**. Per-keyframe visual vectors are already computed at ingest and persisted to the `visual_vectors` table (`pipeline.py:168-169`, `store.py:90-93`), but nothing ever reads them back — there is no `visual_search` and no VISUAL leg in `augur.retrieve`. This track adds `LoomStore.visual_search` over that table (wrapped once behind M0.2's `VectorIndex` ABC), fuses a **default-OFF** VISUAL leg into the RRF that fires only when the plan routes to `visual`, wires the currently-dead MCP `modality` param through to `ask`, adds a deterministic **frame-type classifier** (slide/document/diagram vs talking-head), and introduces a `VisualEmbedder` backend interface with a free `SignatureVisualEmbedder` fallback plus an opt-in ColPali multi-vector (MaxSim) upgrade gated to slide/doc/diagram frames. It also folds in the spec §7 named visual defaults (Surya OCR, Qwen2.5-VL VLM, free path unchanged) and the §9 visual-track parallelism (the serial per-keyframe loop in `tessera.run` is the named throughput bottleneck). A `multimodal` eval block proves a measurable lift on on-screen-only QA, landing ungated then gated.

## Why it matters
This is the literal Phase-4 spec bullet ("ColPali visual retrieval", §11) and the project's core thesis bet (§4.3, "the differentiating core"): knowledge that exists **nowhere in the audio** — a diagram, a chart, a slide of code — must be retrievable *directly*, not only through whatever OCR text happens to linearize. Today on-screen text already leaks into retrieval through the *text* embedding (OCR/caption are concatenated into `Moment.text_for_embedding()`, `models.py:78-84`), but a chart or diagram that OCR cannot transcribe is invisible to retrieval. The VISUAL leg over the grayscale signature (and, opt-in, a ColPali page embedding) makes the frame itself a retrievable unit. The user-visible capability: *"show me the slide with the loss-curve diagram"* surfaces the right Moment even when no spoken or OCR'd word matches.

## Scope (reconciled)

In scope:
- **`LoomStore.visual_search(query_vec, top_k, *, video_id=None, space=...)`** over `visual_vectors`, mirroring `vector_search` (`store.py:554-575`) but reading the visual table. Wrapped **once** behind M0.2's `VectorIndex` ABC so there is a single visual vector index, not a second brute-force cosine to maintain.
- **A 4th VISUAL leg in `augur.retrieve`**, fused into the existing RRF (`retrieve.py:30-65`). **Default OFF.** Fires only when `plan.modality == "visual"` or `plan.strategy == "visual"` (`planner.py:38-39`).
- **Wire the dead MCP `modality` param.** `search_knowledge` declares `modality` in its inputSchema (`mcp.py:42`) but `_tool_search_knowledge` (`mcp.py:137-139`) never passes it. Thread `modality` → `Memovox.ask` (`sdk.py:63`) → `augur.ask` (`answer.py:62`) → `retrieve`.
- **Deterministic frame-type classifier** — `slide` / `document` / `diagram` vs `talking_head`, from OCR density (text length / token count per frame) + the grayscale signature (e.g. edge/variance heuristics on the `frame_side**2` vector). No model; pure stdlib. Gates the ColPali path and is the modality tag the visual leg uses.
- **`VisualEmbedder` backend interface** + free **`SignatureVisualEmbedder`** fallback (wraps the existing grayscale signature already produced by `tessera.frames.bytes_to_signature`, `frames.py:30-32`). Registered in `backends/__init__.py` alongside the other `get_*` factories.
- **Opt-in ColPali** multi-vector late-interaction (MaxSim) embedder, **gated to slide/doc/diagram frames** (never talking-head), behind the `VisualEmbedder` interface. Default OFF; never in the CI gate.
- **The `multimodal` eval block** — transcript-only vs tri-modal `hit_rate` on an on-screen-only QA item. Ungated-then-gated.

Folded in (from the completeness review):
- **§7 named visual defaults.** Add a **Surya OCR** backend (`backends/ocr.py`) and a **Qwen2.5-VL** VLM backend (`backends/vlm.py`) as named options behind the existing `OCRBackend`/`VLMBackend` interfaces. The free path keeps `NullVLM`/`NullOCR` and the signature embedder. Note: today `_OCR_ALIASES = {"surya": "tesseract"}` (`backends/__init__.py:49`) is a *placeholder* alias — replace it with a real `SuryaOCR` class (still `is_available()`-gated, falling back to tesseract→Null).
- **§9 visual-track parallelism.** The per-keyframe loop in `tessera.run` (`__init__.py:112-141`) — extract frame image → OCR → VLM caption, once per kept keyframe, **serially** — is THE named throughput bottleneck. Parallelize the per-frame work (thread or process pool), preserving deterministic output ordering.

Folded-in guard (review-flagged, non-negotiable):
- **Vector-space tagging.** The free **text** vector and the free **visual** signature are *both* 256-d by coincidence (`embed_dim: 256` default, `config.py:42`; `frame_side: 16` ⇒ `16*16 = 256`, `config.py:48`). Cosining a text query vector against a grayscale signature would be a silent, meaningless comparison. Add a **space tag** (e.g. `space="text"` / `space="visual_sig"` / `space="colpali"`) to the stored vectors and to every search call, and **raise** on a space mismatch — a comment is not enough.

Non-goals / deferrals:
- **ANN backends** (Qdrant/LanceDB) are opt-in and **out of the gate** — they belong to M0.2's opt-in column, not here. The free path stays brute-force cosine (optionally batched per M0.2).
- ColPali itself is opt-in and never gated; the gate runs the signature embedder only.
- Clip stitching / answer-with-video is **M2.3**, not here.
- Re-baselining the talk_c golden video and promoting `entity_f1`/`der` is **M1.2** — this track only *adds* the visual fixture + the `visual_result` seam M1.2 consumes.
- Reranking the fused list is **M2.1**.

## Current state (grounded)

- **Visual vectors are written but never read.** `add_moment(..., visual_embedding=...)` packs the per-Moment mean visual signature into `visual_vectors` (`store.py:259-263`); `pipeline.py:168` computes it via `escapement.moment_visual_embedding` (`fusion.py:84-102`). The only reader is `get_visual_vector(moment_id)` (`store.py:266-270`) — a single-row lookup, never used by retrieval. There is **no** `visual_search`.
- **Retrieval has exactly two legs + an optional graph leg.** `retrieve()` builds `legs = [dense, lexical]` and conditionally appends `graph` (`retrieve.py:52-64`). No visual leg exists.
- **The planner already routes to visual but nothing consumes it.** `plan()` returns `QueryPlan(strategy="visual", modality="visual")` for queries containing slide/diagram/chart/"on screen"/etc. (`planner.py:26-39`), but `answer.ask` only checks `qp.strategy == "contradiction"` to toggle the graph leg (`answer.py:81`) — `qp.modality` is read **nowhere**.
- **The MCP `modality` param is dead.** Declared at `mcp.py:42`; `_tool_search_knowledge` calls `self.mv.ask(args["query"], video_id=args.get("video_id"))` with no modality (`mcp.py:138`). `Memovox.ask` has no `modality` parameter (`sdk.py:63`), nor does `augur.ask` (`answer.py:62-70`).
- **The per-keyframe loop is serial.** `tessera.run` iterates kept keyframes one at a time, doing ffmpeg frame extraction → `ocr.extract` → `vlm.caption` inline (`__init__.py:112-141`). On the free path OCR/VLM are no-ops so it is cheap, but with Surya/Qwen wired this is the §9 bottleneck.
- **OCR/VLM backends exist with free fallbacks; Surya/Qwen are not real.** `NullOCR`/`TesseractOCR` (`ocr.py`), `NullVLM`/`OllamaVLM` (`vlm.py`). `_OCRS = {"none", "tesseract"}`, `_VLMS = {"none", "ollama"}` (`backends/__init__.py:47-48`); `"surya"` is aliased to tesseract as a placeholder (`:49`). There is **no** `VisualEmbedder` interface — the only "visual embedding" is the raw grayscale signature carried on `VisualEvent.embedding` (`tessera/__init__.py:49`).
- **The grayscale signature is the de-facto visual embedding.** `bytes_to_signature` (`frames.py:30-32`) normalizes 8-bit grayscale to `[0,1]`; the module docstring (`frames.py:9`) explicitly says it "double[s] as a free, deterministic visual embedding for retrieval until SigLIP/ColPali is wired." That "until" is now.
- **Vectors are stored untagged.** `vectors` and `visual_vectors` schemas (`store.py:85-93`) carry only `(moment_id, dim, vec)` — no space column. `vector_search` filters only on `len(vec) != qlen` (`store.py:571-572`), which does NOT distinguish a 256-d text vector from a 256-d visual signature.
- **A multimodal-lift test already exists — but tests the TEXT path, via a fragile mock.** `test_multimodal_lift_onscreen_text_becomes_retrievable` (`tests/test_integration.py:83-112`) injects a `VisualResult` by `mock.patch("memovox.tessera.run", ...)` and asserts on-screen OCR text becomes retrievable. It works because OCR text flows into the *text* embedding (`models.py:78-84`), **not** through any visual vector leg. This is the seam we promote to a first-class `visual_result=` keyword on `pipeline.ingest` so it is no longer monkeypatch-only — and the proof that visual *vector* retrieval is genuinely untested today.
- **Golden corpus has no video stream.** The VTT fixtures (`eval/golden/talk_a.en.vtt` etc.) ingest transcript-only, so `tessera.run` returns `available=False` and `visual_vectors` stays empty during eval. A visual golden fixture therefore needs the `visual_result=` injection seam — there is no real video to decode.
- **Baseline:** `make test` ⇒ **247 pass / 2 skip**; `python -m eval.harness --assert-thresholds` ⇒ hit_rate≥0.6, groundedness≥0.8, contradiction.f1≥0.5, synthesis.groundedness≥0.8 (gates in `eval/harness.py:642-649`).

## Free-path guarantee

- **The VISUAL leg defaults OFF.** `retrieve()` keeps `legs = [dense, lexical]` unless the plan routes to visual. Every existing retrieval/answer/eval query routes to `hybrid`/`contradiction`/`temporal`/`procedure` (factual eval QA → `hybrid`), so the fused output for the golden corpus is **byte-identical** to today. This mirrors how the graph leg was added default-OFF (`retrieve.py:42-48`).
- **The free visual embedder is the existing signature.** `SignatureVisualEmbedder` is a thin wrapper over `bytes_to_signature` — no model download, no new dependency, deterministic. ColPali/Surya/Qwen are all `is_available()`-gated and yield the Null/signature fallback when absent, exactly like `get_vlm`/`get_ocr`/`get_embedder` (`backends/__init__.py:96-124`).
- **Eval harness pins the free stack and stays default-OFF for visual.** `_FREE_BACKENDS` (`harness.py:62-70`) already pins `vlm_backend="none"`, `ocr_backend="none"`. Per the frozen-eval-settings discipline, **add an explicit `visual_retrieval=False` (default) and `visual_embed_backend="signature"` to `_FREE_BACKENDS`** so a future default flip can't silently move gate numbers. The existing `hit_rate`/`groundedness`/`contradiction`/`synthesis` gates run with the visual leg OFF and must remain unchanged.
- **Space tagging cannot regress the text path.** Existing rows are migrated with `space="text"` (text `vectors`) / `space="visual_sig"` (`visual_vectors`) via an idempotent `ALTER TABLE … ADD COLUMN space TEXT` with a backfill (the same idempotent-migration pattern as `speakers.canonical_id`, `store.py:155-158`). `vector_search` continues to query the text space; nothing in the free path ever crosses spaces.
- **Parallelism defaults to deterministic.** Per-keyframe parallelism uses an ordered map (results reassembled in `kept`-index order), and worker count defaults to `1` (or is keyed off a `visual_workers` setting defaulting to serial behavior) so the free-path output — and any golden ingest — is byte-identical regardless of pool. Concurrency >1 is opt-in and ungated, matching the M3.3 "deterministic by default" honesty note.

## Workstreams

Ordered, TDD-sized, each independently committable. W1–W3 are the retrieval core (depends on M0.2's `VectorIndex` ABC landing first); W4–W6 are the backend/throughput fold-ins; W7 is the eval.

### W1 — Space-tag the vector tables (guard) · S
- **Files:** `src/memovox/loom/store.py` (schema + `_migrate` + `add_moment` + `vector_search`), `tests/test_loom.py`
- **Red (failing test first):** in `tests/test_loom.py`, add `test_text_and_visual_vectors_are_space_tagged` — store a 256-d text embedding and a 256-d visual embedding for the same moment, then assert a new `vector_search(query_vec, space="text")` never returns the visual row and that `vector_search` raising/`assert` on a `space` it has no index for. Fails today: no `space` column, no `space` kwarg.
- **Green (implement):** add a `space TEXT` column to `vectors` and `visual_vectors` (idempotent `ALTER TABLE … ADD COLUMN`, backfilling `'text'`/`'visual_sig'` per the `canonical_id` migration pattern); `add_moment` writes `space='text'`/`space='visual_sig'`; `vector_search(..., space='text')` filters on it. Raise `ValueError` (or a small `VectorSpaceError` in `errors.py`) on a space/dim mismatch rather than silently scoring.
- **Verify:** `make test` (new test + all existing loom tests green); `python -m eval.harness --assert-thresholds` unchanged.
- **Commit:** `feat(loom): space-tag text vs visual vectors; guard cross-space cosine`

### W2 — `LoomStore.visual_search` behind the VectorIndex ABC · M
- **Files:** `src/memovox/loom/store.py` (or the M0.2 `VectorIndex` impl module), `tests/test_loom.py`
- **Red (failing test first):** `test_visual_search_ranks_by_visual_signature` — add three moments with distinct visual signatures, query with a vector close to one, assert it ranks first and that `visual_search` ignores rows whose `space != 'visual_sig'`. Fails today: `AttributeError: 'LoomStore' has no attribute 'visual_search'`.
- **Green (implement):** `visual_search(query_vec, top_k=20, *, video_id=None)` mirroring `vector_search` (`store.py:554-575`) but reading `visual_vectors WHERE space='visual_sig'`, with the same `video_id` join and `dim` guard. Route it through M0.2's `VectorIndex` ABC so it is the single visual index (no second ad-hoc cosine). Reuse `cosine`/`unpack_floats` from `vectormath`.
- **Verify:** `make test`; gates unchanged.
- **Commit:** `feat(loom): visual_search over the visual_vectors index (spec §4.3/§6)`

### W3 — Fuse the default-OFF VISUAL leg + wire `modality` end-to-end · M
- **Files:** `src/memovox/augur/retrieve.py`, `src/memovox/augur/answer.py`, `src/memovox/augur/planner.py` (only if a strategy tweak is needed), `src/memovox/sdk.py`, `src/memovox/server/mcp.py`, `tests/test_augur.py`, `tests/test_mcp.py`
- **Red (failing test first):**
  - `tests/test_augur.py::test_visual_leg_off_by_default_byte_identical` — `retrieve(..., use_visual=False)` returns exactly today's fused list (regression guard). Plus `test_visual_query_fuses_visual_leg` — for a query the planner routes to `visual`, a moment retrievable ONLY by its visual signature (no lexical/text overlap) appears in the fused result; it does NOT appear when the leg is off. Fails today: no visual leg, no `use_visual` param.
  - `tests/test_mcp.py::test_search_knowledge_threads_modality` — calling `search_knowledge` with `arguments={"query": ..., "modality": "visual"}` reaches `ask` with `modality="visual"`. Fails today: `_tool_search_knowledge` drops `modality` (`mcp.py:138`).
- **Green (implement):**
  - `retrieve(..., use_visual=False, visual_query_vec=None)`: when `use_visual`, append `store.visual_search(visual_query_vec or <signature of query>, pool, video_id=...)` as a 4th leg into the RRF (`retrieve.py:52-64`). The visual query vector is produced by the `VisualEmbedder` (W4) — for a text query with no image this may be empty, in which case the leg is skipped (graceful, like the empty-graph case).
  - `answer.ask(..., modality="any")`: set `use_visual = (qp.modality == "visual" or qp.strategy == "visual" or modality == "visual")`; pass through to `retrieve`. Keep all other strategies untouched so the hybrid/contradiction/temporal paths are byte-identical.
  - `Memovox.ask(query, *, video_id=None, modality="any")` (`sdk.py:63`) forwards `modality`.
  - `_tool_search_knowledge` passes `modality=args.get("modality")` (`mcp.py:137-139`).
- **Verify:** `make test` (augur + mcp + integration green); `python -m eval.harness --assert-thresholds` — gates unchanged (every golden QA routes non-visual).
- **Commit:** `feat(augur): default-OFF VISUAL RRF leg + live MCP modality param (spec §5)`

### W4 — `VisualEmbedder` interface + free `SignatureVisualEmbedder` + frame-type classifier · M
- **Files:** `src/memovox/backends/base.py` (new `VisualEmbedder` ABC), `src/memovox/backends/visual_embed.py` (new), `src/memovox/backends/__init__.py` (`get_visual_embedder`, registry, `backend_status`), `src/memovox/tessera/frames.py` or a new `tessera/classify.py` (frame-type classifier), `src/memovox/config.py` (`visual_embed_backend` setting), `tests/test_backends.py`, `tests/test_tessera.py`
- **Red (failing test first):**
  - `tests/test_backends.py::test_signature_visual_embedder_is_free_and_deterministic` — `get_visual_embedder("auto")` returns a backend whose `embed_image(sig_or_path)` is deterministic and tagged `space="visual_sig"`. Fails: no interface/registry.
  - `tests/test_tessera.py::test_frame_type_classifier` — a high-OCR-density / high-edge signature classifies as `slide`/`document`/`diagram`; a low-text, low-variance signature classifies as `talking_head`. Fails: classifier does not exist.
- **Green (implement):** `class VisualEmbedder(Backend)` with `dim`, `space`, `embed_image(...)`. `SignatureVisualEmbedder` wraps `bytes_to_signature` (the current free path). `get_visual_embedder(name="auto", config=...)` mirrors `get_ocr`/`get_vlm` (`backends/__init__.py:96-124`), defaulting to signature. Add the deterministic `classify_frame(signature, ocr_text) -> "slide"|"document"|"diagram"|"talking_head"` (OCR token-density threshold + signature variance/edge heuristic; pure stdlib). Add `visual_embed_backend: str = "signature"` to `Settings`.
- **Verify:** `make test`; gates unchanged.
- **Commit:** `feat(backends): VisualEmbedder interface + signature fallback + frame-type classifier (spec §4.3)`

### W5 — Named visual defaults: Surya OCR + Qwen2.5-VL (free path unchanged) · M
- **Files:** `src/memovox/backends/ocr.py` (`SuryaOCR`), `src/memovox/backends/vlm.py` (`Qwen25VL`), `src/memovox/backends/__init__.py` (registries + drop the placeholder `surya→tesseract` alias at `:49`), `pyproject.toml` (optional extras), `tests/test_backends.py`
- **Red (failing test first):** `tests/test_backends.py::test_surya_and_qwen_fall_back_to_free_when_absent` — `get_ocr("surya")`/`get_vlm("qwen2.5-vl")` raise `BackendUnavailable` when the dep is missing (consistent with the other `get_*` factories), and `"auto"` still yields the free Null fallback. Fails today: `"surya"` is silently aliased to tesseract (`backends/__init__.py:49`); no Qwen backend exists.
- **Green (implement):** add `SuryaOCR(OCRBackend)` and `Qwen25VL(VLMBackend)`, each `is_available()`-gated on the import, with the existing graceful-fallback semantics. Register in `_OCRS`/`_VLMS`; replace the `_OCR_ALIASES["surya"]` placeholder with the real class. Add `[ocr]`/`[vlm]` (or `[visual]`) extras to `pyproject.toml`. **No free-path change:** `auto` on a bare machine still returns `NullOCR`/`NullVLM`.
- **Verify:** `make test`; `python -m eval.harness --assert-thresholds` unchanged (`_FREE_BACKENDS` pins `none`).
- **Commit:** `feat(backends): named Surya OCR + Qwen2.5-VL visual defaults (spec §7); free path unchanged`

### W6 — Parallelize the per-keyframe loop (§9 bottleneck) · M
- **Files:** `src/memovox/tessera/__init__.py` (`run`), `src/memovox/config.py` (`visual_workers: int = 1`), `tests/test_tessera.py`
- **Red (failing test first):** `tests/test_tessera.py::test_keyframe_work_is_order_deterministic_under_workers` — run with `visual_workers=1` and `visual_workers=4` over the same injected frames + a deterministic fake OCR/VLM that records call order; assert the resulting `events` list (order, timestamps, captions, embeddings) is **identical** across worker counts. Fails today: the loop is serial-only with no parallel seam to exercise (and any naive pool would reorder events).
- **Green (implement):** refactor the per-keyframe body (`__init__.py:112-141`) into a pure `_process_keyframe(pos, idx, ...) -> VisualEvent` and map it over `kept` via a `ThreadPoolExecutor` (I/O-bound: ffmpeg + OCR/VLM subprocess/HTTP), reassembling results in `kept` order. Default `visual_workers=1` (byte-identical serial behavior). Log worker count to **stderr** only.
- **Verify:** `make test` (tessera tests green, output order stable); gates unchanged.
- **Commit:** `perf(tessera): parallelize per-keyframe OCR/VLM work, deterministic ordering (spec §9)`

### W7 — `visual_result` injection seam + visual golden fixture + `multimodal` eval block · L
- **Files:** `src/memovox/pipeline.py` (`ingest(..., visual_result=None)`), `eval/golden/talk_vis.en.vtt` (new) + `eval/golden/visual.json` (new, the visual fixture + injected `VisualResult`), `eval/harness.py` (`multimodal` block + `_FREE_BACKENDS` additions), `tests/test_integration.py`, `tests/test_eval.py`
- **Red (failing test first):**
  - `tests/test_integration.py::test_ingest_accepts_injected_visual_result` — replace the `mock.patch("memovox.tessera.run", ...)` seam (`test_integration.py:100`) with `mv.ingest(vtt, visual_result=fake)`; assert visual_vectors are written and a visual-only query (routed to `visual`) retrieves the slide-only Moment **via the visual leg** (not just the text path). Fails today: `ingest` has no `visual_result` param (`pipeline.py:90-103`); the visual leg test would only pass through OCR text.
  - `tests/test_eval.py::test_multimodal_block_present` — `run_eval` report has a `multimodal` key with `transcript_only` and `tri_modal` hit_rate, and `tri_modal >= transcript_only`. Fails: no such block.
- **Green (implement):**
  - Add `visual_result: Optional[VisualResult] = None` as a **keyword-only** addition to `pipeline.ingest` (coordinated with M0.3, which owns the signature change — layer this in as keyword-only). When supplied, skip `tessera.run` and use it directly (`pipeline.py:144`). This is the single `visual_result` injection seam M1.2 consumes — **do not** add a second one.
  - Author the visual golden fixture: a short VTT (`talk_vis`) whose transcript deliberately does NOT contain the on-screen term, plus a `visual.json` carrying the injected `VisualResult` (an OCR/caption + a distinctive visual signature) and an on-screen-only QA item whose gold answer is reachable only through the visual content. Keep it thin (the §10 thin-fixture discipline).
  - In `harness.py`: ingest the visual fixture once **without** the visual leg (transcript-only `hit_rate`) and once **with** it (tri-modal), emit a `multimodal` block. Add `visual_retrieval=False` + `visual_embed_backend="signature"` to `_FREE_BACKENDS` (`harness.py:62-70`) to pin the frozen eval-settings snapshot.
- **Verify:** `make test`; `python -m eval.harness` shows the `multimodal` block; `--assert-thresholds` still green (block is **ungated** at first).
- **Commit:** `feat(eval): visual_result injection seam + visual golden fixture + multimodal lift block (spec §10)`

## Eval gate

- **New metric:** `multimodal` block — `transcript_only` vs `tri_modal` `hit_rate@k` on the on-screen-only QA item(s) in the visual golden fixture. The bet it proves: with the VISUAL leg ON, a Moment whose answer lives only on-screen is retrieved; with it OFF (transcript-only), it is missed.
- **Lands UNGATED first** (per the global thin-fixture discipline: a new metric is gated only once ≥3 stable golden items exist). With one visual fixture item it ships ungated, reported in the harness output but not in `_check_thresholds`.
- **Promote to a gate in M1.2** (which owns the talk_c re-baseline and golden growth) once ≥3 on-screen-only items are stable. Proposed gated threshold: `multimodal.tri_modal >= multimodal.transcript_only + δ` (a real lift), with `tri_modal` hit_rate ≥ the existing `_HIT_RATE_GATE` (0.6). Wire it alongside the other gates in `_check_thresholds` (`harness.py:656-670`) at that time.
- **Existing gates stay green:** because the visual leg is default-OFF and every current golden QA routes non-visual, `retrieval.hit_rate ≥ 0.6`, `groundedness ≥ 0.8`, `contradiction.f1 ≥ 0.5`, `synthesis.groundedness ≥ 0.8` are unchanged. Confirm by running `python -m eval.harness --assert-thresholds` after each workstream and asserting the report dict for the non-multimodal keys is byte-identical to the pre-track baseline.

## Risks & mitigations

- **Silent cross-space cosine (review-flagged).** Text (256-d hashing) and visual (256-d grayscale signature) vectors are the same length by coincidence (`config.py:42` vs `:48`). *Mitigation:* W1 adds a real `space` column + a hard raise on mismatch, not a comment; `visual_search` filters `space='visual_sig'` and never touches the text rows.
- **Eval thinness / gate flakiness (dominant program risk).** A single visual fixture item is not a stable gate. *Mitigation:* land the `multimodal` block ungated; gating waits on M1.2 growing ≥3 items; pin `visual_retrieval=False`/`visual_embed_backend` in `_FREE_BACKENDS` so the frozen snapshot can't drift.
- **Visual signature is a weak embedding.** A 16×16 grayscale mean is coarse; the lift on a tiny fixture may be marginal. *Mitigation:* the fixture is engineered so the answer term is absent from transcript AND OCR-linearizable text but present as a distinctive signature, so the signature leg is the *only* path — making even a coarse vector sufficient to demonstrate the mechanism. The opt-in ColPali path is the real-quality upgrade, benchmarked later (M3.4).
- **Determinism erosion from parallelism.** A thread/process pool can reorder `events`. *Mitigation:* reassemble strictly by `kept` index; default `visual_workers=1`; the W6 red test asserts identical output across worker counts.
- **ColPali storage cost (§12).** Multi-vector-per-frame is expensive at scale. *Mitigation:* gate ColPali strictly to `slide`/`document`/`diagram` frames via the W4 classifier (never talking-head), exactly as the spec warns (`spec.md:367`); ColPali stays opt-in and ungated.
- **`pipeline.ingest` signature collision.** `published_at=`, `visual_result=`, `modality=`, and ASR device knobs all touch the parameter list across tracks. *Mitigation:* M0.3 owns the signature change; W7 layers `visual_result=` in as a **keyword-only** addition, coordinated, not a competing rewrite.
- **MCP stdout discipline (§ non-negotiable).** Any new logging in the visual leg, the parallel pool, or the backends must go to **stderr** (MCP speaks JSON-RPC on stdout). *Mitigation:* no `print`; route through the M0.1 stderr logging hook; assert no stray stdout in the MCP test.
- **Re-baseline blast radius.** Adding a golden video perturbs several metrics. *Mitigation:* the visual fixture is a *separate* on-screen-only item feeding only the `multimodal` block; it does not enter the `retrieval`/`entity`/`speaker`/`contradiction` scorers, so it does not move existing gates. (Growing/re-baselining the *core* golden corpus is M1.2's serialized commit.)

## Definition of done

- [ ] `vectors`/`visual_vectors` carry a `space` tag; cross-space cosine raises (W1).
- [ ] `LoomStore.visual_search` exists, reads only `space='visual_sig'`, wrapped behind M0.2's `VectorIndex` ABC (W2).
- [ ] A default-OFF VISUAL RRF leg fires only on `plan.modality/strategy == "visual"`; `retrieve(use_visual=False)` is byte-identical to today (W3).
- [ ] MCP `search_knowledge` `modality` param reaches `augur.ask` (W3).
- [ ] `VisualEmbedder` ABC + free `SignatureVisualEmbedder` + `get_visual_embedder`; deterministic frame-type classifier (W4).
- [ ] Named `SuryaOCR` + `Qwen25VL` backends, `is_available()`-gated, free path unchanged; placeholder surya→tesseract alias removed (W5).
- [ ] Per-keyframe loop parallelized with order-deterministic output; `visual_workers=1` default (W6).
- [ ] `pipeline.ingest(..., visual_result=)` keyword-only seam; visual golden fixture; `multimodal` lift block in the harness, **ungated** (W7).
- [ ] `make test` green (≥ 247 + new tests, 2 skip); `python -m eval.harness --assert-thresholds` green; the four existing gate values unchanged from baseline.
- [ ] All new logging to stderr; no new dependency on the free path.

## Open questions

- **`modality` enum coverage.** The MCP schema enum is `["any","speech","visual"]` (`mcp.py:42`), but `Moment.modality` emits `"speech+slide"` (`models.py:71-76`). Should `modality="visual"` mean "turn the visual *leg* on" (retrieval routing) or "filter citations to visual-bearing Moments" (output filtering)? This doc assumes the former (routing); confirm before W3.
- **ColPali multi-vector storage shape.** `visual_vectors` is one-vector-per-moment (`store.py:90-93`). ColPali is multi-vector-per-frame with MaxSim scoring — does it get a new `visual_patches` table (and MaxSim in `vectormath`), or is it deferred entirely behind the interface with only the signature path materialized now? Recommend: define the `VisualEmbedder` interface to *allow* multi-vector, but materialize only the signature single-vector path in this track; ColPali storage lands when it's actually wired/benchmarked (M3.4).
- **Frame-type classifier thresholds.** OCR-density and signature-variance cutoffs for slide/document/diagram vs talking-head are heuristic; with no real-video golden fixture they can only be unit-tested on synthetic signatures. Confirm acceptable to ship heuristic thresholds now and tune against real keyframes opportunistically.
- **Should the free signature leg ever be ON by default for explicitly-visual queries even without ColPali?** This doc keeps it OFF-by-default and only ON when the planner routes to visual, so the free path's golden numbers never move. Confirm that the planner-routed-ON behavior is acceptable for the (ungated) `multimodal` block while staying OFF for every other query.
