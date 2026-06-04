# M0.3 — Word-precise spans, fail-loud device, ingest-signature owner

> **Wave:** 0 · **Effort:** M · **Status:** not started
> **Depends on:** none · **Owns (single-owner concerns):** the `pipeline.ingest()` signature change (keyword-only additions: `published_at=`, `visual_result=`, `modality=`, and the ASR device knobs — coordinated HERE so M1.1/M3.1 layer on, not churn it) · **Blocks:** M1.2 (span/citation gate), M2.3 (clip precision)
> **Spec:** §4.1, §4.2, §7, §9

## Goal
Thread optional per-word timings (`Word`) from the ASR layer through `SegmentRef` → `Moment.segments` into `assay/spans.py::locate_span`, so a claim's `(t_start_s, t_end_s)` is **tightened to the matched word window when words exist** and is **byte-for-byte identity when they don't** (the free VTT/captions path). Make a silently-CPU-placed faster-whisper `large-v3` **fail loud** via a new `DevicePlacementError` (with `--allow-cpu` / `MEMOVOX_ASR_ALLOW_CPU` escape). Establish the canonical `pipeline.ingest()` signature so the four Phase-4 tracks that touch its parameter list add keyword-only params instead of fighting over it. Fold in the §4.1 ffprobe ASR-readiness pre-check hardening and the §9 captions-as-fused-prior cost lever. Trail optional WhisperX forced-alignment + pyannote diarization backends as opt-in, `is_available`-gated, lazy-imported, graceful-fallthrough upgrades.

## Why it matters
Citation precision is the product. Today a claim's provenance window is **cue-granular** — it points at the whole VTT cue (often 5–12 s) the sentence overlaps, because that is the finest grain `SegmentRef` carries (`fusion.py:47`). M2.3 ("answer-with-video" clip stitching) and the M1.2 unified span/citation-accuracy gate both need **word-precise** windows ("the 92 seconds where she actually says it"), not a 10-second ceiling. WhisperX already emits word timestamps that we currently **discard** at the `SegmentRef` boundary. This track preserves them end-to-end. Separately, faster-whisper silently falling back to CPU turns a 5-minute transcribe into a 2-hour one with no signal — a §9 throughput failure mode the spec calls out; failing loud (escape-hatched) protects the user. And because four tracks all want to add a keyword to `ingest()`, owning that signature change once prevents a four-way merge churn.

## Scope (reconciled)
In scope:
- **Word timings through the data model.** Add an optional `words` field to `SegmentRef` (default empty) so the existing positional unpack `(t0, t1, text) = ref` stays valid; populate it in `escapement/fusion.py::_make_moment` from `Segment.words`.
- **`locate_span` word-window tightening.** When the best-matched segment carries `words`, narrow the returned `(t0, t1)` to the span of the words that actually match the claim sentence; when it carries no words, return the existing segment window **unchanged** (identity). Add an opt-out so the cue-granular behavior can be forced if needed.
- **Provenance caveat (design constraint).** Tighten the **citation/display** span only. Keep the NLI **premise** (`span_text`, used by the verification gate at `assay/__init__.py:36`) **segment-granular**, so the displayed span never drifts narrower than the text the gate actually verified. Assert this invariant in a test.
- **`DevicePlacementError`** in `errors.py`; raised in `backends/asr_whisper.py::_load` when `large-v3` (or any non-tiny model) would run on CPU, unless `--allow-cpu` (CLI) / `MEMOVOX_ASR_ALLOW_CPU=1` (env) / `Settings.asr_allow_cpu` is set.
- **ASR device knobs on `Settings`** (`asr_device`, `asr_compute_type`, `asr_allow_cpu`), wired through `stentor.run` → `run_asr` → `WhisperASR.options`, replacing the current hard-coded `options.get("device", "auto")` (`asr_whisper.py:28`).
- **`pipeline.ingest()` canonical signature** (single-owner): add keyword-only `published_at=None`, `visual_result=None`, `modality=None`, and the ASR device knobs are surfaced via `settings`. Stubs that are not yet consumed are accepted and threaded to the right place (`published_at` → `Video.published_at`; `visual_result`/`modality` reserved seams for M1.1) but do nothing surprising on the free path.

Folded in from the completeness review:
- **§4.1 ffprobe ASR-readiness pre-check** — harden `_prepare_audio` (`asr.py:114`) so a media file with no decodable audio stream fails with a clear `DemuxError` *before* model load (the check exists at `asr.py:117`; extend it to surface the probe verdict as a structured pre-check and ensure it runs on the `auto`→`whisper` path).
- **§9 captions-as-fused-prior cost lever** — when both a captions file AND media exist, allow using the (free, exact-timing) captions as the segment/word prior instead of paying for Whisper, surfaced as a backend/option choice (default unchanged: captions still win on `auto`, `asr.py:74`).

Non-goals / deferrals:
- **WhisperX forced-alignment + pyannote diarization (turns + voiceprints) backends are opt-in, NOT in the gate.** They trail as `is_available`-gated, lazy-import, graceful-fallthrough upgrades; the free path must not import them.
- The **eval golden corpus stays cue-granular** (talk_a/talk_b VTTs carry no inline word tags — verified: 0 `<00:..>` tags). Word-precision is exercised on a **new word-bearing fixture**, not by mutating the frozen corpus.
- Multi-worker / frame-parallelism throughput is M-X.1 / M3.3, not here.

## Current state (grounded)
- **Word type exists but is dropped at the Moment boundary.** `Word(word, start, end)` is a dataclass (`backends/base.py:21-25`); `Segment.words: List[Word]` exists (`base.py:34`); `WhisperASR.transcribe` already builds per-word lists with `word_timestamps=True` (`asr_whisper.py:55-75`). But `SegmentRef` is a 3-field `NamedTuple(t_start_s, t_end_s, text)` with **no words slot** (`loom/models.py:15-25`), and `_make_moment` constructs `SegmentRef(round(s.start,3), round(s.end,3), s.text.strip())` — **discarding `s.words`** (`escapement/fusion.py:47-48`). So word timing dies at fusion.
- **`locate_span` is cue-granular.** It picks the best segment by token-overlap and returns that segment's full `(t0, t1)` (`assay/spans.py:19-42`). No sub-segment narrowing. Called twice in `assay/claims.py` (rule-based `:184`, LLM `:223`), both with `default=(moment.t_start_s, moment.t_end_s)`.
- **NLI premise is already segment-granular** and must stay so: `premise = span_text(moment.segments, claim.t_start_s, claim.t_end_s) or moment.text_for_embedding()` (`assay/__init__.py:36`). `span_text` selects segments by strict overlap (`spans.py:58`). If `locate_span` narrows `(t0,t1)` inside a single cue, `span_text` still returns that whole cue's text (overlap is non-empty), so the premise stays segment-granular for free — but this must be asserted, not assumed.
- **Device is hard-coded to `"auto"`** with no fail-loud: `device = self.options.get("device", "auto")` and `compute_type = self.options.get("compute_type", "default")` (`asr_whisper.py:28-29`). `DEFAULT_MODEL = "large-v3"` (`asr_whisper.py:15`). faster-whisper silently runs on CPU when CUDA is absent. There is **no `DevicePlacementError`** — `errors.py` defines only `MemovoxError, ConfigError, AcquisitionError, DemuxError, BackendUnavailable, IngestionError, NotFoundError` (`errors.py:6-31`).
- **`Settings` has ASR backend selection but no device knobs:** `asr_backend: str = "auto"` (`config.py:31`), `voiceprint_backend` (`config.py:40`), env-coercion via `from_env` (`config.py:73-82`). No `asr_device` / `asr_compute_type` / `asr_allow_cpu`.
- **`pipeline.ingest()` signature** (`pipeline.py:90-103`) is keyword-rich already (`source_url, title, captions, cookies, language, glossary, force, settings, store`). `Video.published_at` is plumbed from `meta.published_at` (`pipeline.py:121`, `acquire.py:31/151`) but **not accepted as an explicit `ingest()` override** — local sources can't inject it. `modality`/`visual_result` are not params.
- **ffprobe pre-check exists but is narrow:** `_prepare_audio` calls `audio.has_audio_stream` and raises `DemuxError` (`asr.py:117-118`); `audio.probe` returns `{has_audio, has_video, duration, codecs}` with a WAV-header fallback when ffprobe is absent (`audio.py:34-70`). It does not surface a structured "ASR-not-ready" verdict and runs only inside the whisper branch.
- **Opt-in backend precedent to mirror:** `PyannoteVoiceprint` (`backends/diarize_voiceprint.py`) — `is_available()` guarded by `importlib.util.find_spec` with the parent-package `try/except` idiom, lazy import inside `embed`, `# pragma: no cover` on the model path. WhisperASR uses the same `find_spec` gate (`asr_whisper.py:22-24`). There is **no WhisperX backend** and **no diarization-turns backend** yet.
- **Eval harness pins `_FREE_BACKENDS` only** (`eval/harness.py:62-70`) — a 7-key dict, no frozen snapshot of the growing default-OFF surface. New ASR device/word flags must be added to a pinned snapshot per the program discipline (see Eval gate).
- **Tests:** `tests/test_spans.py` covers `locate_span` with plain 3-tuples and the overlap floor (9 cases). No word-window test, no device test, no fusion-word-preservation test. `make test` ⇒ 247 pass / 2 skip (`Makefile:6-7`).

## Free-path guarantee
The stdlib-only deterministic default is the **captions/VTT path** with the `hashing`/`lexical`/`none` backends. It stays intact by construction:
- `SegmentRef.words` defaults to `()` (empty tuple). The VTT parser (`transcript.py::parse_cues`) and JSON parser produce `Segment`s with **no** `words`, so every `SegmentRef` on the free path carries `words == ()`.
- `locate_span` with an empty `words` list returns the **exact same segment window** it returns today — the narrowing branch is entered only when `best segment.words` is non-empty. This is the **identity** the gate asserts byte-for-byte against the golden corpus.
- The NLI premise (`span_text`) is untouched: it already operates on segment overlap, and narrowing the citation window inside a cue does not change which segments overlap. Free-path verification output is unchanged.
- `DevicePlacementError` only fires inside `WhisperASR._load`, which is never reached on the captions path (no `audio_path`, `name != "whisper"`, `asr.py:100`). The default `asr_backend="auto"` still resolves to `captions` whenever a transcript is present (`asr.py:74`).
- `published_at`/`visual_result`/`modality` are keyword-only with `None` defaults; `None` reproduces today's behavior exactly (`published_at` still falls back to `meta.published_at`).
- WhisperX/pyannote-turns backends are `is_available`-gated with lazy imports; importing `memovox` pulls in nothing beyond stdlib (mirrors `diarize_voiceprint.py`).
- New flags land in the frozen eval-settings snapshot so a future default flip can't silently move a gate number.

## Workstreams

### W1 — `SegmentRef` carries words; fusion preserves them · S
- **Files:** `src/memovox/loom/models.py`, `src/memovox/escapement/fusion.py`, `tests/test_fusion_words.py` (new)
- **Red (failing test first):** `tests/test_fusion_words.py` builds `Segment`s with `words=[Word("the",0.0,0.2), Word("chain",0.2,0.5), ...]`, runs `escapement.build_moments(...)`, and asserts `moment.segments[i].words` is non-empty and word boundaries round-trip. Fails today because `_make_moment` (`fusion.py:47`) drops `s.words`.
- **Green (implement):** Add `words: Tuple = ()` (or `List` default-empty via a factory-safe NamedTuple field) as a 4th `SegmentRef` field with a default so positional `(t0,t1,text) = ref` unpacking in `spans.py:38/58` still works. Populate it in `_make_moment`: `words=tuple((round(w.start,3), round(w.end,3), w.word) for w in s.words)`. Confirm `span_text`/`locate_span` 3-tuple unpacking is unaffected (they index `[0:3]` positionally).
- **Verify:** `make test` (new test green; `test_spans.py` still green; 247→248+ pass).
- **Commit:** `feat(escapement): preserve per-word timings on SegmentRef (spec §4.2)`

### W2 — `locate_span` tightens to the matched word window (identity when absent) · M
- **Files:** `src/memovox/assay/spans.py`, `tests/test_spans.py`
- **Red (failing test first):** add to `test_spans.py`: a segment with words `[("the",0,0.2),("chain",0.2,0.5),("rule",0.5,0.9),("here",0.9,1.2)]` spanning a 10 s cue `(0.0,10.0)`; assert `locate_span("chain rule", [seg_with_words])` returns approximately `(0.2, 0.9)` — the word window, NOT `(0.0,10.0)`. Also assert the **identity** case: `locate_span("chain rule", [(0.0,10.0,"the chain rule here")])` (no words) still returns `(0.0,10.0)`. The word-window assertion fails today (returns the whole cue).
- **Green (implement):** After selecting `best` segment, if it carries words, compute the min-start/max-end over the words whose normalized token is in the claim's token set (reuse `util.tokenize`); return that `(w0, w1)` clamped within `(t0,t1)`. Fall back to `(t0,t1)` when no word matches (defensive) or when `words` is empty (identity — the free path). Add a keyword `tighten: bool = True` so a caller can force cue-granular if ever needed. Keep the 0.5 overlap floor at the segment level (unchanged selection logic).
- **Verify:** `make test`; assert the golden corpus span output is unchanged (W5 gate harness).
- **Commit:** `feat(assay): tighten located spans to the matched word window (spec §4.5)`

### W3 — Provenance caveat: premise stays segment-granular (assert it) · S
- **Files:** `tests/test_span_premise_invariant.py` (new); possibly a 1-line guard in `assay/__init__.py`
- **Red (failing test first):** construct a Moment with one 10 s cue + words, extract a claim, narrow its `(t0,t1)` via the W2 path, then assert `span_text(moment.segments, claim.t_start_s, claim.t_end_s)` returns the **whole cue's text** (the NLI premise), NOT just the matched words — i.e. the displayed citation window is ≤ the verified premise window. Document why: provenance-is-sacred (review risk #4). This should already hold after W2 (overlap is non-empty); the test pins it so a future `span_text` change can't silently let the display drift past the verified premise.
- **Green (implement):** No behavior change expected — `span_text` overlap is `s0 < t_end and s1 > t_start` (`spans.py:58`), and a narrowed window still overlaps its parent cue. If the test surprises us, add an explicit guard so the premise window ⊇ the citation window. Add a one-line comment at `assay/__init__.py:36` cross-referencing this invariant.
- **Verify:** `make test`.
- **Commit:** `test(assay): pin citation-span ⊆ verified-premise invariant (spec §4.5)`

### W4 — `DevicePlacementError` + ASR device knobs (fail loud, escape-hatched) · M
- **Files:** `src/memovox/errors.py`, `src/memovox/backends/asr_whisper.py`, `src/memovox/config.py`, `src/memovox/stentor/asr.py`, `src/memovox/stentor/__init__.py`, `tests/test_asr_device.py` (new)
- **Red (failing test first):** `tests/test_asr_device.py` monkeypatches `WhisperASR.is_available`→True and a fake `WhisperModel` reporting a CPU placement for `large-v3`; assert `WhisperASR(...)._load()` raises `DevicePlacementError`. Then assert that with `asr_allow_cpu=True` (or `MEMOVOX_ASR_ALLOW_CPU=1`) it does NOT raise. Fails today (`_load` never checks device; no such error class).
- **Green (implement):** Add `class DevicePlacementError(MemovoxError)` to `errors.py`. In `WhisperASR._load`, after resolving `device`/`compute_type` from options, if the effective model is large (`large-*`) and the resolved device is CPU and `allow_cpu` is False → raise `DevicePlacementError` with a message naming `--allow-cpu` / `MEMOVOX_ASR_ALLOW_CPU` / a smaller model. Add `asr_device: str = "auto"`, `asr_compute_type: str = "default"`, `asr_allow_cpu: bool = False` to `Settings` (`config.py`), pick up `MEMOVOX_ASR_ALLOW_CPU` etc. via the existing `from_env` machinery. Thread them: `stentor.run(..., settings=...)` → `run_asr` passes `device`/`compute_type`/`allow_cpu` into `get_asr("whisper", ...)` options (`asr.py:95-97`). Surface `--allow-cpu` on the CLI ingest command.
- **Verify:** `make test`; manual: free/captions path raises nothing (whisper branch never runs).
- **Commit:** `feat(asr): fail loud on silent CPU placement of large-v3 (--allow-cpu escape, spec §9)`

### W5 — Own the `pipeline.ingest()` signature (keyword-only additions) + free-VTT unchanged gate · M
- **Files:** `src/memovox/pipeline.py`, `eval/harness.py` (span metric + snapshot), `tests/test_ingest_signature.py` (new), `eval/golden/` (NO mutation), `eval/fixtures/` word-bearing fixture (new, outside the frozen golden dir)
- **Red (failing test first):** (a) `tests/test_ingest_signature.py` calls `pipeline.ingest(config, source, published_at="2024-01-02", visual_result=None, modality=None, ...)` on a captions fixture and asserts it succeeds and `store.get_video(vid).published_at == "2024-01-02"` — fails today (`ingest` rejects unknown kwargs). (b) A span-output snapshot test ingests the **golden** corpus and asserts every claim's `(t_start_s, t_end_s)` is byte-identical to a checked-in baseline (proves no silent widening). (c) A word-bearing fixture (`.json` transcript with `words`) asserts at least one claim's span is **strictly narrower** than its cue.
- **Green (implement):** Add keyword-only `published_at=None`, `visual_result=None`, `modality=None` to `ingest` (`pipeline.py:90`). Use `published_at or meta.published_at` for `Video.published_at`. Accept `visual_result`/`modality` as reserved seams: thread `visual_result` to where `tessera.run` output is consumed (so M1.1 can inject a precomputed result instead of recomputing) and `modality` is stored/ignored until M1.1 — document the contract in the docstring so M1.1/M3.1 layer on without re-editing the param list. Extend the JSON transcript parser (`transcript.py::parse_json`) to read an optional per-cue `words` array so a fixture can carry word timings on the free path.
- **Verify:** `make test`; `python -m eval.harness` span-output snapshot unchanged.
- **Commit:** `feat(pipeline): own ingest() signature — published_at/visual_result/modality keyword-only seams`

### W6 — Folded-in §4.1 ffprobe ASR-readiness + §9 captions-as-fused-prior · S
- **Files:** `src/memovox/stentor/asr.py`, `src/memovox/audio.py` (optional helper), `tests/test_asr_readiness.py` (new)
- **Red (failing test first):** assert that resolving `whisper` against a media file whose `probe()` reports `has_audio=False` raises a clear `DemuxError` (the ASR-readiness verdict) **before** any model load; and that a captions-prior option, when set, makes `auto` prefer the captions segments/words over Whisper even when media is present (default stays: captions already win, so the test guards the explicit-flag path).
- **Green (implement):** Promote the `_prepare_audio` audio-stream check (`asr.py:117`) into a named readiness pre-check that runs on the `auto`→`whisper` resolution and emits a structured reason; wire a `captions_as_prior` option/setting that routes `auto` to `CaptionsASR` when a transcript is present (already the default ordering at `asr.py:74` — make it explicit and documented as the §9 cost lever).
- **Verify:** `make test`.
- **Commit:** `feat(stentor): ffprobe ASR-readiness pre-check + captions-as-fused-prior cost lever (spec §4.1/§9)`

### W7 — Opt-in WhisperX alignment + pyannote diarization-turns backends (trailing) · M
- **Files:** `src/memovox/backends/asr_align.py` (new), `src/memovox/backends/diarize_turns.py` (new), `src/memovox/stentor/asr.py` (registry), `tests/test_optional_align_backends.py` (new)
- **Red (failing test first):** assert that when the optional packages are absent, `is_available()` is False, the registry resolution falls through to the free path, and importing the modules pulls in nothing beyond stdlib (mirror `tests` for `diarize_voiceprint`). Assert that a forced request for an unavailable backend raises `BackendUnavailable` (matching `get_asr`, `asr.py:65-67`).
- **Green (implement):** Add a `WhisperXAlign` backend that, when available, refines word boundaries via forced alignment (lazy import, `is_available` via `find_spec` with the parent-package `try/except` idiom from `diarize_voiceprint.py:43`), and a pyannote diarization-turns backend producing speaker turns + (reusing the existing `PyannoteVoiceprint`) voiceprints. Both `# pragma: no cover` on the model path. Graceful fallthrough: never imported on the free path; never in the gate.
- **Verify:** `make test`; confirm `python -c "import memovox"` imports no optional deps.
- **Commit:** `feat(backends): opt-in WhisperX alignment + pyannote turns (is_available-gated, spec §7)`

## Eval gate
- **New metric: unified `span_accuracy` (a.k.a. span/citation-accuracy) — the SAME metric M1.2 owns end-to-end.** This track lands its **measurement seam and the free-path guard**; M1.2 owns gating it. Per the program discipline, the metric **lands UNGATED** here (computed and printed in the harness report, no threshold) and is **gated only once ≥3 stable golden items** with word timings exist (M1.2's job). Proposed gated threshold when promoted: **`span_accuracy ≥ 0.5`** mean IoU vs gold spans (matches the M1.2 § "IoU≥0.3 vs gold clip" family, conservative for words). State explicitly in the harness comment that it is ungated-then-gated.
- **Hard guard that ships in THIS track (gating, not ungated): the free VTT/captions span output is asserted UNCHANGED.** A snapshot test (W5) ingests the frozen golden corpus and asserts every committed claim's `(t_start_s, t_end_s)` equals a checked-in baseline. Any silent widening/narrowing fails CI. This is the byte-identity contract the reconciled scope demands.
- **Frozen eval-settings snapshot:** add `asr_device`, `asr_compute_type`, `asr_allow_cpu`, `captions_as_prior`, and the `tighten` default to a pinned snapshot alongside `_FREE_BACKENDS` (`eval/harness.py:62`) so a future default flip cannot move a gate number silently (program discipline (b)).
- **Existing gates stay green:** `python -m eval.harness --assert-thresholds` must still pass `hit_rate ≥ 0.6`, `groundedness ≥ 0.8`, `contradiction.f1 ≥ 0.5`, `synthesis.groundedness ≥ 0.8` (`harness.py:642-649`). Because the golden VTTs carry no word tags (verified: 0 inline `<00:..>` tags), `locate_span` runs the identity branch on the corpus, so these numbers are unchanged by construction — the W5 snapshot proves it.

## Risks & mitigations
- **Span tightening drifts from the verified premise (review risk #4, provenance-is-sacred).** Mitigation: W3 pins `citation_window ⊆ premise_window`; the premise (`span_text`) stays segment-granular by overlap, and the test fails if a future edit lets the displayed span drift past what NLI checked.
- **Silent free-path widening.** Mitigation: the W5 byte-identity snapshot of golden-corpus claim spans is a CI gate, not an ungated metric; the empty-`words` identity branch is unit-tested in W2.
- **`SegmentRef` is a `NamedTuple` consumed by positional unpack in two hot paths** (`spans.py:38`, `spans.py:58`). Adding a field can break `(t0,t1,text) = ref`. Mitigation: append `words` as the LAST field with a default; the existing 3-tuple unpack uses `for (t0,t1,text) in ...` which is brittle to a 4-tuple — change those loops to index `seg[0], seg[1], seg[2]` or unpack `(t0,t1,text,*_)`. W1/W2 tests cover both production `SegmentRef` and plain 3-tuples (the test fixtures still pass 3-tuples).
- **`DevicePlacementError` false-positives** (e.g. Apple Metal / a deliberate CPU dev run). Mitigation: the `--allow-cpu` / `MEMOVOX_ASR_ALLOW_CPU` / `Settings.asr_allow_cpu` escape; the check fires only for large models on CPU, and small models never trip it.
- **Signature churn across tracks (the reason this track owns it).** Mitigation: ALL of `published_at`/`visual_result`/`modality`/device-knobs land here as keyword-only with `None`/inert defaults and a documented contract; M1.1/M3.1 add no positional params and re-use the reserved seams.
- **Optional-backend import leakage onto the free path.** Mitigation: mirror `diarize_voiceprint.py` exactly — `find_spec` with parent-package `try/except`, lazy import inside the method, `# pragma: no cover` on model code; a test asserts `import memovox` pulls no optional deps.
- **Determinism erosion from new default-OFF flags (review risk #2).** Mitigation: the frozen eval-settings snapshot pins them.

## Definition of done
- [ ] `SegmentRef` carries optional `words`; `_make_moment` populates them from `Segment.words`; positional 3-tuple unpack in `spans.py` still works.
- [ ] `locate_span` tightens to the matched word window when words exist; identity when absent; unit-tested both ways.
- [ ] NLI premise stays segment-granular; `citation_window ⊆ premise_window` invariant pinned by a test.
- [ ] `DevicePlacementError` exists and fires on silent CPU placement of `large-v3`, with `--allow-cpu` / `MEMOVOX_ASR_ALLOW_CPU` / `Settings.asr_allow_cpu` escape; ASR device knobs threaded `Settings`→`run_asr`→`WhisperASR`.
- [ ] `pipeline.ingest()` accepts keyword-only `published_at=`, `visual_result=`, `modality=` with documented contracts; free path byte-identical with `None`.
- [ ] §4.1 ffprobe ASR-readiness pre-check surfaces a clear verdict before model load; §9 captions-as-fused-prior lever wired (default unchanged).
- [ ] Optional WhisperX-align + pyannote-turns backends are `is_available`-gated, lazy-imported, fall through gracefully, never imported on the free path, never in the gate.
- [ ] New `span_accuracy` metric computed and printed UNGATED; free-VTT-corpus span output asserted UNCHANGED (gating snapshot).
- [ ] New flags added to the frozen eval-settings snapshot.
- [ ] `make test` green (247→target with new tests); `python -m eval.harness --assert-thresholds` passes all four existing gates.

## Open questions
- **Word-window granularity for multi-cue claims.** When a claim's sentence spans two adjacent cues each carrying words, do we tighten to the union of matched words across both cues, or stay segment-granular (current `locate_span` already falls back below the 0.5 floor for boundary-crossing sentences, `test_spans.py:58`)? Proposed: tighten only within the single best segment; defer cross-cue word stitching to M2.3 clip merging. Confirm.
- **`visual_result` injection shape.** M1.1 owns the `visual_result` seam; this track only reserves the keyword. Confirm with M1.1 the exact type (a `tessera` result object vs a dict) so the reserved param's type hint is right the first time.
- **`captions_as_prior` surface.** Is this a `Settings` flag, an `asr_backend` value, or a `run_asr` option? Proposed: a `Settings.captions_as_prior` bool defaulting to the current behavior (captions win on `auto`); confirm naming so the frozen snapshot key is stable.
- **Apple-Silicon device classification.** Should faster-whisper on Metal/MPS count as "not CPU" for the fail-loud check? faster-whisper (CTranslate2) does not support MPS, so large-v3 on a Mac will be CPU — the escape hatch covers it, but confirm we want Mac dev users to hit the loud error by default.
