# memovox — implementation notes & spec mapping

This document maps the codebase to [`../spec.md`](../spec.md) and records what is
fully implemented in **Phases 0–1** versus scaffolded for later phases.

## Guiding constraint

Built to run **free, local-first, no API keys**. Every model slot has a
deterministic standard-library fallback, so the entire pipeline executes with no
downloads, no GPU, and no network. Real backends are optional upgrades behind a
common interface (spec §7). The core depends on **nothing but the Python
standard library**.

## Subsystem ↔ module map (spec §3)

| Codename | Module | Status |
|----------|--------|----------------|
| **Stentor** (acquire/ASR/diarize) | `memovox.stentor` | ✅ local files + URL (yt-dlp opt); captions/whisper/fake ASR; ffprobe pre-check + demux; speaker-label fallback |
| **Tessera** (visual track) | `memovox.tessera` | ✅ ffmpeg frame sampling, content-aware scene segmentation, information-gain keyframe selection, free visual-signature embedding; VLM caption + OCR slots (Null fallback / Ollama-vision / tesseract); graceful degradation with no video |
| **Escapement** (fusion → Moments) | `memovox.escapement` | ✅ speaker/gap/duration/event/topic-shift boundaries; ✅ tri-modal fusion of visual events (caption/OCR/visual embedding) into Moments |
| **Assay** (claims + verify) | `memovox.assay` | ✅ rule-based + LLM-opt extraction, epistemic typing, NLI gate, salience |
| **Loom** (indices + graph + synthesis) | `memovox.loom` | ✅ relational + FTS5 lexical + BLOB vectors + visual vectors + edge-table graph; ✅ basic cross-corpus contradiction/agreement |
| **Augur** (retrieve + answer) | `memovox.augur` | ✅ planner, dense+lexical RRF, cited extractive/LLM answers (on-screen text retrievable via fused Moment text) |

## Pipeline stages (spec §4)

| Stage | Where | Notes |
|-------|-------|-------|
| 0 Acquire | `stentor/acquire.py` | local media/transcript (free) + yt-dlp; content-hash idempotency key |
| 1 Demux | `audio.demux_to_wav`, `audio.probe` | ffmpeg → 16 kHz mono WAV; ffprobe validation pre-check |
| 2 Audio/ASR | `stentor/asr.py`, `backends/asr_whisper.py` | captions / faster-whisper / fake; filler stripping + audio-event markers; glossary biasing |
| 3 Visual | `tessera/` (`frames`/`scenes`/`keyframes`) | ffmpeg signatures → content-aware scenes → information-gain keyframes → caption/OCR/visual-embedding events; degrades to no-op without a video stream |
| 4 Fusion | `escapement/fusion.py` | Moments are the atomic retrieval/citation unit; visual events bound by time overlap → `visual_caption`/`ocr_text`/visual embedding |
| 5 Claims+verify | `assay/` | entailment gate; unsupported claims flagged, never silently dropped |
| 6 Resolution+index | `loom/store.py`, `pipeline.py` | triple write; within-video speakers; PRECEDES/STATES/ATTRIBUTED_TO edges |
| 7 Consolidation | `loom/consolidate.py` | contradiction/agreement via inverted-index + NLI (sync, not yet async/background) |

## Data model (spec §6)

`loom/models.py` implements Video, Moment, Claim, Entity, Speaker, Topic, and the
Provenance object. The single SQLite DB (`loom/store.py`) carries all four
indices: relational, lexical (FTS5), vector (float32 BLOB + brute-force cosine),
and the temporal knowledge graph (provenanced edge table). This honors the
"human-readable substrate" principle — `sqlite3 <store>/memovox.db` works, and a
Markdown digest is written per video.

## Design principles (spec §2) — how each is enforced

- **Provenance is sacred** — every `Citation`/edge carries `(video, span,
  modality, confidence)` + a deep link; answers are built only from retrieved Moments.
- **Verify before commit** — `assay.verify.verify_claim` gates every claim by NLI
  entailment against its source span.
- **Idempotent ingestion** — deterministic content-hash IDs; `LoomStore.is_unchanged`
  skips no-op re-ingests; changed content is replaced cleanly.
- **Model-agnostic / local-first** — `backends/` registries with `auto` selection
  and free fallbacks.
- **Human-readable substrate** — per-video Markdown digests + an inspectable DB.

## What is intentionally deferred (spec §11 roadmap)

- **Phase 1 — remaining upgrades.** The visual track ships (scene detection,
  information-gain keyframe selection, caption/OCR slots, Moment fusion). Still
  optional/future: PySceneDetect/SigLIP/ColPali multi-vector embeddings and
  query-side visual late-interaction retrieval (the stored visual vectors are not
  yet a retrieval leg in Augur); the free defaults are a content-diff scene
  detector and a downscaled-intensity visual signature.
- **Phase 2** Full entity/speaker resolution across videos; richer graph retrieval.
- **Phase 3** Async background consolidation, claim-evolution tracking, consensus
  scoring, topic induction.
- **Phase 4** Subscriptions/incremental sync (basic `sync()` present), answer-with-video
  clip stitching (`/clip` returns spans), decay/versioning, dashboards.

## Backends (free fallback ↔ optional upgrade)

| Slot | Free fallback | Upgrade (extra) |
|------|---------------|-----------------|
| ASR | captions / fake | `faster-whisper` `[asr]` |
| Acquire | local file | `yt-dlp` `[acquire]` |
| Embedder | hashing | `sentence-transformers` `[embed]` |
| NLI | lexical | DeBERTa-NLI `[nli]` |
| LLM | none (extractive) | Ollama (stdlib HTTP) / `[llm]` |
| VLM caption | none (NullVLM) | Ollama vision (stdlib HTTP) |
| OCR | none (NullOCR) | `tesseract` binary on PATH |
| Frames/scenes | ffmpeg + stdlib signature | PySceneDetect/SigLIP (future) |
| Vector/Lexical/Graph | SQLite | Qdrant/LanceDB/Kùzu (future) |

## Tests

Stdlib `unittest` (no pytest needed): `make test`. Coverage spans backends,
Stentor parsing/ASR, Escapement fusion, Loom storage/search/graph, Assay
extraction+gate, Augur retrieval+answers, the end-to-end pipeline (ingest→ask→
export→idempotency→contradiction), and the CLI + MCP dispatch.
