# MEMOVOX — Technical Specification

*A multimodal video-to-knowledge engine. Voice in, queryable memory out.*

**Version:** 0.1 (design)
**Status:** Specification / pre-build
**One-line:** Memovox ingests video at the level of *meaning* — fusing speech, on-screen content, and speakers onto a single timeline, distilling it into a verified temporal knowledge graph you can query, synthesize across, and cite back to the exact second.

---

## 1. Thesis — why this is different

The overwhelming majority of "chat with your video" tools are **transcript-only RAG**: pull captions or run Whisper, chunk the text, embed it, do top-*k* vector retrieval, hand it to an LLM. That pattern is now commodity, and it has four structural failures:

1. **It is blind.** Everything that lives only on the screen — slides, code, equations, diagrams, charts, UI demonstrations — is discarded. For technical talks, lectures, and tutorials this is often where the actual knowledge is.
2. **It is flat.** A video is collapsed into an unordered bag of fixed-size chunks. Temporal, causal, and rhetorical structure ("they said X, *then corrected it to* Y") is lost.
3. **It is anonymous.** It cannot tell *who* said something, and cannot recognize the same speaker across different videos.
4. **It is solipsistic.** Each video is an island. It cannot tell you that two sources disagree, or that a claim made in 2024 was revised in 2026.

Memovox is built on six bets that target exactly these gaps. Each is independently shippable and independently testable.

| # | Differentiator | What it fixes |
|---|----------------|---------------|
| 1 | **Tri-modal fusion on a unified timeline** — transcript + visual semantics + on-screen text (OCR), temporally aligned | Blindness |
| 2 | **Adaptive keyframe selection** — sample frames by *information gain*, not a fixed interval | Cost + visual recall |
| 3 | **Moments, not chunks** — atomic units bounded by natural topic/scene breaks, binding co-occurring speech + slide + speaker | Flatness |
| 4 | **Temporal knowledge graph** — claims, entities, speakers, relations, every edge timestamped and provenanced | Flatness + anonymity |
| 5 | **Cross-corpus synthesis** — agreement/contradiction detection and claim-evolution tracking across the whole library | Solipsism |
| 6 | **Self-verifying ingestion** — every committed claim is entailment-checked against its source span before it enters the graph | Trust / hallucination |

The combination is the product. Transcript-only tools can bolt on one of these; the value is in fusing all six behind one provenance-first interface.

---

## 2. Design principles (non-negotiable)

- **Provenance is sacred.** Every assertion in the system resolves to `(video_id, [t_start, t_end], modality, confidence)`. There are no orphan facts. If Memovox can't say where it learned something, it doesn't claim it.
- **Verify before commit.** Extracted knowledge is checked against its source for entailment *before* it lands in the graph. The knowledge base never asserts something the video didn't.
- **Modality-agnostic retrieval.** One query interface spans text, on-screen visual content, and audio events. The user asks a question; the planner decides which modality answers it.
- **Idempotent, incremental ingestion.** Re-ingesting is safe. New content merges, dedupes, and *supersedes* (with version history) rather than duplicating. Nothing is ever silently deleted.
- **Model-agnostic and local-first.** ASR, VLM, embedder, and LLM are all swappable backends behind a common interface. The whole system can run fully local for sensitive corpora, or call hosted models for quality. Backends are *benchmarkable* against each other (see §10).
- **Human-readable substrate.** Knowledge persists as inspectable artifacts — per-video Markdown digests plus a queryable graph plus vectors — not an opaque blob. You can `cat` your knowledge, version it in git, and back it up.
- **Eval-driven.** Nothing ships unless it moves a benchmark number and doesn't regress faithfulness or retrieval quality.

---

## 3. System architecture

```
                                    ┌─────────────────────────────────────────────┐
  source(s) ──▶  STENTOR  ──▶ audio │  faster-whisper + WhisperX  + pyannote        │
  (URL /          (acquire,         │  → word-aligned, speaker-labeled transcript   │
   playlist /      demux,           └─────────────────────────────────────────────┘
   channel)        ASR,                                    │
                   diarize)                                ▼
                       │            ┌─────────────────────────────────────────────┐
                       └──▶ video ─▶│  TESSERA                                      │
                                    │  scene detect → adaptive keyframe select →    │
                                    │  VLM caption + OCR + visual embedding         │
                                    └─────────────────────────────────────────────┘
                                                           │
                                            ┌──────────────▼──────────────┐
                                            │  ESCAPEMENT                  │
                                            │  temporal fusion → MOMENTS   │
                                            │  (speech + slide + speaker   │
                                            │   on one timeline)           │
                                            └──────────────┬──────────────┘
                                                           ▼
                                            ┌──────────────────────────────┐
                                            │  ASSAY                        │
                                            │  claim extraction →           │
                                            │  NLI entailment gate →        │
                                            │  salience + type tagging      │
                                            └──────────────┬──────────────┘
                                                           ▼
                          ┌────────────────────────────────────────────────────────┐
                          │  LOOM   — write to three indices + resolve + consolidate │
                          │  ┌───────────┐  ┌───────────┐  ┌────────────────────┐    │
                          │  │  vector   │  │  lexical  │  │  temporal knowledge │    │
                          │  │ (dense +  │  │ (BM25 +   │  │  graph (nodes/edges │    │
                          │  │  ColPali) │  │  SPLADE)  │  │  + provenance)      │    │
                          │  └───────────┘  └───────────┘  └────────────────────┘    │
                          │  entity resolution · speaker resolution · contradiction  │
                          │  detection · claim-evolution tracking · topic induction  │
                          └────────────────────────────┬───────────────────────────┘
                                                        ▼
                                            ┌──────────────────────────────┐
                                            │  AUGUR — query layer          │
                                            │  plan → hybrid retrieve →     │
                                            │  rerank → grounded answer +   │
                                            │  timestamped clip citations   │
                                            └──────────────────────────────┘
                                                        ▼
                              CLI   ·   Python SDK   ·   REST   ·   MCP server
```

**Subsystem codenames** (house style — single-word, watch / aviation / myth / nature):

| Codename | Subsystem | Why the name |
|----------|-----------|--------------|
| **Stentor** | Acquisition + ASR + diarization | The herald in the *Iliad* whose voice was as loud as fifty men — the system's ears |
| **Tessera** | Visual track | A mosaic tile — keyframes are the tiles that reconstruct the visual picture |
| **Escapement** | Temporal fusion engine | The watch component that regulates the beat of time — here it regulates the timeline |
| **Assay** | Claim extraction + verification | The metallurgical test of purity — proving a claim before it's accepted |
| **Loom** | Knowledge graph + cross-corpus synthesis | Weaving individual claims into a connected fabric |
| **Augur** | Agentic retrieval + answer synthesis | Divining the answer from the available signs |

---

## 4. Ingestion pipeline (detailed)

The pipeline is a resumable DAG. Each stage is idempotent and content-addressed; a failure or a re-run never corrupts or duplicates state.

### Stage 0 — Acquire (Stentor)
- **Fetch** via `yt-dlp` (source-agnostic: YouTube plus ~1,800 other sites). Retrieve the best audio stream, the video stream, **native captions** (used as a cheap prior and fallback, not the source of truth), and metadata (title, channel, description, chapter markers, publish date, thumbnail).
- **Gated content:** cookie-file support (Netscape format) for member-only / authenticated videos.
- **Resilience:** native-captions-as-prior reduces ASR cost when good captions exist; exponential-backoff retries handle rate limiting; the acquisition layer is fully modular so `yt-dlp` breakage is isolated.

### Stage 1 — Demux
- `ffmpeg` → normalized **16 kHz mono WAV** for ASR; retain the video stream for frame extraction.
- **`ffprobe` pre-check** validates the audio stream before transcription (a known failure mode: `yt-dlp` can emit files `ffmpeg` decodes but the ASR model chokes on).

### Stage 2 — Audio track (Stentor)
- **ASR:** `faster-whisper` (large-v3) as default for ~4× throughput at equal accuracy; **WhisperX** for word-level timestamps via forced alignment.
- **Diarization:** `pyannote.audio` produces speaker turns; align with word timestamps → *who said what, when*.
- **Domain biasing:** inject a per-channel / per-domain glossary into the decoder prompt to fix jargon, proper nouns, and acronyms (materially improves technical-content accuracy).
- **Audio-event tagging:** detect `[music]` / `[applause]` / laughter / long silence; **strip filler tokens** (`um`, `uh`, `[Music]`) from the knowledge text but retain them as timeline events (they're useful boundaries, and they otherwise waste context budget).
- **Output:** cleaned, word-aligned, speaker-labeled transcript segments on the master timeline.

### Stage 3 — Visual track (Tessera) — *the differentiating core*
This is where Memovox stops being a transcript tool. The hard problem is **which frames to look at** — uniform sampling (a frame every *N* seconds) is the standard approach and the standard mistake: it wastes compute on static talking-head footage and *misses* fast slide changes. The 2025–26 keyframe-selection literature is consistent that information-aware selection beats uniform sampling at equal or lower cost.

- **Scene segmentation:** `PySceneDetect` (content-aware) for hard cuts, **plus perceptual hashing (pHash) on the screen region** to catch slide/screen changes that aren't scene cuts (a 50-minute single-shot lecture may contain 40 distinct slides).
- **Adaptive keyframe selection (information gain):** within each scene, embed candidate frames (SigLIP/CLIP) and keep a frame only when its embedding distance from the last *kept* frame exceeds a threshold. Static segments collapse to one or two frames; slide- and demo-dense segments are sampled densely. Per-scene caps and near-duplicate suppression bound cost.
- **Per kept keyframe:**
  - **VLM captioning** — a vision-language model produces a dense description ("Slide titled *Backpropagation*; bar chart comparing train vs. val loss; bullet text reads…").
  - **OCR** (`Surya` / `PaddleOCR`) — extract slide text, code, equations, lower-thirds. *This is knowledge that exists nowhere in the audio.*
  - **ColPali-style visual embedding** (gated to slide/document/diagram frames) — a multi-vector page embedding so the frame is retrievable *directly*, capturing charts and diagrams that OCR can't faithfully linearize.
- **Output:** timestamped visual events `{caption, ocr_text, visual_embedding, frame_ref}`.

### Stage 4 — Temporal fusion (Escapement)
- Merge audio segments and visual events onto one monotonic timeline.
- Build **Moments**: coherent, time-bounded units that bind co-occurring speech + on-screen content + active speaker. *Example:* `00:12:30–00:14:05 — Speaker A explains the chain rule while Slide 14 (the derivative diagram) is shown.`
- **Boundary detection:** topic boundaries from transcript embedding-shift (TextTiling-style) are *reconciled with* scene/slide boundaries, so Moments break at natural seams — not at arbitrary token counts. Moments, not fixed chunks, are the atomic unit of retrieval and citation.

### Stage 5 — Claim extraction + verification (Assay) — *the trust layer*
- An LLM extracts from each Moment: **atomic claims** (subject–predicate–object + qualifiers), definitions, questions raised, procedures, and entity mentions — each tied to its exact source span.
- **Verification gate (the anti-hallucination guarantee):** each extracted claim is run through an **NLI entailment check** against its cited source span(s). Claims not entailed by the source are dropped or flagged `unsupported` — they never enter the graph as facts.
- **Salience scoring:** rank each claim by centrality, speaker authority, and in-video redundancy → drives retrieval priority and summary inclusion.
- **Epistemic typing:** classify each as `FACT` / `DEFINITION` / `OPINION` / `PROCEDURE` / `EXAMPLE` / `PREDICTION` / `CORRECTION` so retrieval and synthesis can reason about the *kind* of statement, not just its content.

### Stage 6 — Resolution & indexing (Loom)
- **Entity resolution:** link mentions to canonical entities (people, orgs, papers, products, concepts) across the corpus; optionally ground to an external KB (Wikidata) for disambiguation.
- **Speaker resolution:** cluster speaker embeddings (voiceprints) *across videos* so the same person is recognized library-wide; merge with names from metadata and on-screen lower-thirds where available.
- **Triple write:**
  1. **Vector** — Moment text embeddings + ColPali visual vectors (Qdrant in prod, LanceDB embedded/local).
  2. **Lexical** — BM25 + SPLADE over transcript and OCR text for exact-term, jargon, and code recall.
  3. **Graph** — the temporal knowledge graph (§6).

### Stage 7 — Cross-corpus consolidation (Loom, async)
Runs as a background job as the library grows.
- **Contradiction & agreement detection:** cluster semantically equivalent claims across videos; pairwise NLI marks `SUPPORTS` / `CONTRADICTS` edges. Surfaces *"Source A (2024) asserts X; Source B (2026) asserts ¬X."*
- **Claim-evolution tracking:** for an entity/topic, order claims by source publish date to trace how a position or a number changes over time.
- **Consensus scoring:** weight a claim cluster by source count, recency, and speaker authority → a confidence estimate.
- **Topic induction:** cluster Moments into emergent topics; maintain a topic map of the whole library.
- **Dedup & decay:** merge duplicate claims; optionally down-weight stale facts. Superseded claims are versioned, never deleted.

---

## 5. Retrieval & query layer (Augur)

- **Query planner (agentic):** decompose the question, then choose the retrieval strategy and modality.
  - *"How did X's view on Y change?"* → temporal graph query over X's claims about Y, ordered by date.
  - *"Show me where they demo the install."* → visual + `PROCEDURE` retrieval.
  - *"What did the speaker say right after the chain-rule slide?"* → timeline-anchored retrieval.
- **Hybrid retrieval:** dense + sparse + graph run in parallel; fuse with Reciprocal Rank Fusion; rerank with a cross-encoder. For visual-heavy queries, ColPali late-interaction over frames. (Hybrid dense+lexical with reranking is the empirically strongest baseline configuration in the video-RAG literature; the graph leg is the extension.)
- **Multi-hop traversal:** follow `SUPPORTS` / `CONTRADICTS` / `ELABORATES` edges for synthesis questions.
- **Answer synthesis:** the LLM composes strictly from retrieved Moments; **every sentence carries a citation** to `(video, timestamp, modality)`; the system flags low-evidence answers rather than confabulating.
- **Answer-with-video:** return stitched, deep-linked clip references — *"the 92 seconds where this is explained"* — not just text.
- **Output modes:** grounded chat answer + citations · structured JSON extraction · per-video study-note digest · corpus-level "literature review" synthesis across many sources.

---

## 6. Data model (concrete)

Relational metadata + provenance in SQLite (local) / Postgres (prod); vectors in Qdrant/LanceDB; graph in **Kùzu** (embeddable) or Neo4j.

```sql
-- Core relational schema (abridged)

Video(
  video_id TEXT PK, source_url TEXT, title TEXT, channel TEXT,
  published_at DATE, duration_s INT, lang TEXT,
  content_hash TEXT,            -- dedupe / idempotency key
  ingested_at TIMESTAMP, pipeline_version TEXT
)

Moment(
  moment_id TEXT PK, video_id FK,
  t_start_s REAL, t_end_s REAL,
  transcript TEXT,              -- cleaned speech for this span
  speaker_id FK NULL,
  visual_caption TEXT NULL,     -- VLM description of co-occurring frame
  ocr_text TEXT NULL,           -- on-screen text
  topic_id FK NULL,
  text_embedding_ref TEXT,      -- pointer into vector store
  visual_embedding_ref TEXT NULL
)

Claim(
  claim_id TEXT PK, moment_id FK,
  subject TEXT, predicate TEXT, object TEXT, qualifiers JSON,
  claim_type TEXT,              -- FACT|DEFINITION|OPINION|PROCEDURE|EXAMPLE|PREDICTION|CORRECTION
  salience REAL,
  entailment_score REAL,        -- NLI vs. source span; gate threshold applied
  status TEXT,                  -- committed|unsupported|superseded
  superseded_by FK NULL
)

Entity(entity_id PK, canonical_name TEXT, type TEXT, wikidata_qid TEXT NULL, aliases JSON)
Speaker(speaker_id PK, label TEXT, voiceprint_ref TEXT, resolved_name TEXT NULL)
```

```text
-- Temporal knowledge graph (Loom)

Nodes:  Video · Moment · Claim · Entity · Speaker · Topic
Edges (all timestamped + provenance-stamped):
  (Speaker)-[:STATES {t}]->(Claim)
  (Claim)-[:MENTIONS]->(Entity)
  (Claim)-[:SUPPORTS | :CONTRADICTS | :ELABORATES | :CORRECTS]->(Claim)
  (Moment)-[:PRECEDES]->(Moment)
  (Claim)-[:ATTRIBUTED_TO]->(Speaker)
  (Moment)-[:ABOUT]->(Topic)

-- Every edge carries: {source_video_id, t_start, t_end, modality, confidence}
```

**Provenance object** (attached to every retrievable fact):
```json
{
  "video_id": "yt:dQw4w9WgXcQ",
  "t_start_s": 750.0,
  "t_end_s": 845.0,
  "modality": "speech+slide",
  "speaker": "spk_03 (Dr. A. Researcher)",
  "confidence": 0.91,
  "deep_link": "https://youtu.be/dQw4w9WgXcQ?t=750"
}
```

---

## 7. Tech stack

Default is **local-first**; every model slot is a swappable backend behind a common interface.

| Layer | Default | Swappable alternatives | Rationale |
|-------|---------|------------------------|-----------|
| Acquisition | `yt-dlp` | — | Source-agnostic, battle-tested, broad site coverage |
| Media | `ffmpeg` / `ffprobe` | — | Normalization + the validation pre-check |
| ASR | `faster-whisper` (large-v3) | WhisperX, hosted ASR APIs | ~4× faster at equal accuracy |
| Word timing | WhisperX forced alignment | — | Word-level timestamps for precise citation |
| Diarization | `pyannote.audio` | — | Speaker turns + voiceprints |
| Scene/keyframe | `PySceneDetect` + `imagehash` + SigLIP | — | Content-aware + info-gain selection |
| Visual captioning | Qwen2.5-VL (local) | Claude vision, GPT-4o-class VLMs | Dense on-screen description |
| OCR | `Surya` | PaddleOCR | Slides, code, equations |
| Visual retrieval | ColPali | — | Multi-vector page retrieval for diagrams/charts |
| Text embedding | BGE-M3 | OpenAI / Voyage / Cohere | Single model emits dense **+** sparse **+** multi-vector |
| Vector DB | Qdrant (prod) | LanceDB (embedded) | Hybrid + payload filtering |
| Lexical | Tantivy BM25 + SPLADE | — | Exact-term, jargon, and code recall |
| Graph DB | Kùzu (embeddable) | Neo4j | Columnar, fast multi-hop, no server to run |
| Verification | DeBERTa-NLI | LLM-as-judge | Entailment gate + contradiction detection |
| LLM (extract / synthesize) | model-agnostic | Claude / local | Behind a `Backend` interface |
| Orchestration | Prefect / Temporal | Dagster | Resumable, idempotent DAG |
| Job queue | Redis | RabbitMQ | Ingestion fan-out |
| Serving | FastAPI | — | REST + MCP |
| Observability | structured logs + per-stage metrics + traces | — | Stage-level cost/latency visibility |

**Backend abstraction:** ASR, VLM, embedder, and LLM each implement a thin interface (`transcribe()`, `caption()`, `embed()`, `complete()`), so the engine never hard-depends on a vendor and any backend can be **A/B-benchmarked** against another on a fixed eval set (§10).

---

## 8. Interfaces

### CLI
```bash
memovox ingest <url | playlist | channel>     # one video, a playlist, or a whole channel
memovox ask "how did the speaker's view on X change over time?"
memovox sync                                  # pull + ingest new videos from subscriptions
memovox contradictions --topic "scaling laws" # surface disagreements in the corpus
memovox export --video <id> --format md       # human-readable digest
```

### Python SDK
```python
from memovox import Memovox
mv = Memovox(store="~/knowledge")
mv.ingest("https://youtu.be/...")
ans = mv.ask("what's the recommended chunk size, and who recommended it?")
for c in ans.citations:
    print(c.video_id, c.t_start_s, c.deep_link)
```

### REST (FastAPI)
`POST /ingest` · `POST /query` · `GET /clip?video&t_start&t_end` · `GET /export/{video_id}` · `GET /graph/contradictions`

### MCP server — *the agent-native interface*
The whole niche runs on MCP, and the natural home for Memovox is inside Claude Code / Claude Desktop. Exposed tools:
- `ingest_video(url)`
- `search_knowledge(query, modality?)`
- `get_claim_provenance(claim_id)`
- `synthesize_topic(topic)`
- `find_contradictions(topic?)`

This makes the knowledge base directly callable from an agent session — ingest a talk and immediately interrogate it without leaving the editor.

---

## 9. Non-functional requirements

- **Throughput:** target ~1 hour of video ingested in a few minutes on a single modern GPU. The **visual track is the bottleneck**, so frame work is parallelized and gated by adaptive selection.
- **Hardware reality:** large-v3 needs ~10 GB VRAM and will *silently* fall back to CPU (≈10× slower) on smaller cards — the pipeline detects device placement and fails loud rather than silently degrading.
- **Query latency:** < ~2 s p95 for a grounded answer over a mid-size corpus.
- **Scale:** designed for 10k+ video libraries with incremental ingestion and background consolidation.
- **Cost:** native-captions-as-prior + adaptive keyframe selection are the two biggest cost levers — together they cut spend dramatically versus the naive "transcribe-everything + uniform-frame-sampling + caption-every-frame" approach. Per-video token/compute budgets are configurable and logged.
- **Privacy:** a fully local mode (local ASR/VLM/embedder/LLM, embedded DBs) so no data leaves the machine for sensitive corpora.
- **Reliability:** per-stage retries; resumable DAG; **graceful degradation** — if the VLM fails, transcript-derived knowledge still commits and the visual layer is flagged missing for later backfill.

---

## 10. Evaluation — how we know it's working

A golden eval set lives in the repo; CI gates merges on faithfulness and retrieval regressions.

| Dimension | Metric | Notes |
|-----------|--------|-------|
| Retrieval quality | Hit-rate / MRR / nDCG | On a labeled QA set built from the corpus |
| **Multimodal lift** | Accuracy delta: transcript-only vs. tri-modal | On a **visual-heavy subset** (slides/code/diagrams) — this validates the core thesis |
| Groundedness | % of answer sentences entailed by cited spans | NLI-checked; the headline trust metric |
| Citation accuracy | Do timestamps point to the supporting Moment? | Auto + spot-check |
| Keyframe efficiency | Frames embedded vs. uniform sampling at equal accuracy | The cost/quality curve for adaptive selection |
| Contradiction detection | Precision / recall on seeded contradictions | Cross-corpus correctness |
| Diarization / entity res | DER / clustering F1 | Speaker + entity resolution quality |

The model-agnostic backend abstraction means each ASR / VLM / embedder / LLM choice is itself a variable to be measured on these metrics, not assumed — the same evaluation harness ranks backends.

---

## 11. Build roadmap

| Phase | Codename scope | Deliverable | Differentiator unlocked |
|-------|----------------|-------------|--------------------------|
| **0 — Skeleton** | Stentor (audio) + minimal Loom + Augur | `yt-dlp` → faster-whisper → semantic Moments → embed → hybrid retrieve → cited answer → **MCP server** | Already edges transcript-only tools via provenance + agent-native interface |
| **1 — Multimodal** | Tessera + Escapement | Scene detect + adaptive keyframe selection + VLM caption + OCR + Moment fusion | **Tri-modal fusion** + visual-lift eval |
| **2 — Knowledge + trust** | Assay + full Loom | Claim extraction + NLI gate + salience; temporal KG; entity/speaker resolution; graph retrieval | **Verified knowledge graph** |
| **3 — Synthesis** | Loom (async) | Contradiction/agreement detection, claim-evolution tracking, consensus scoring, topic induction, corpus-level synthesis | **Cross-corpus reasoning** |
| **4 — Scale & polish** | all | Channel/playlist subscriptions + incremental sync, answer-with-video clip stitching, ColPali visual retrieval, decay/versioning, dashboards | Production-grade library |

Phase 0 is a genuinely useful tool on its own. Each subsequent phase is independently shippable and independently measurable.

---

## 12. Open questions & risks

- **VLM cost is the dominant variable.** Adaptive keyframe selection mitigates it, but the information-gain threshold needs per-content-type tuning (a lecture vs. a vlog vs. a screencast behave very differently).
- **ColPali storage.** Multi-vector-per-frame is expensive at scale; gate strictly to slide/document/diagram frames.
- **Claim granularity.** Too fine → a noisy graph; too coarse → weak provenance. Eval-driven tuning of extraction prompts.
- **Cross-video diarization is hard.** Voiceprint drift and name collisions are real; start within-video, expand to library-wide only behind the diarization eval.
- **Acquisition fragility.** `yt-dlp` breaks when sites change; gated content and rate limits add friction. Mitigated by captions-as-prior, cookie handling, retries, and a modular acquisition layer.
- **Legal / ToS.** Personal-knowledge-base use of transcripts and stored frames; respect source terms and keep the corpus private by default.

---

*Memovox — Jaeger-LeCoultre's Memovox (1950) was the wristwatch that spoke back: an alarm that gave time a voice. This is the memory that speaks back — voice in, knowledge out.*
