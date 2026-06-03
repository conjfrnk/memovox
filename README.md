# memovox

**A multimodal video-to-knowledge engine. Voice in, queryable — and *cited* — memory out.**

Most "chat with your video" tools are transcript-only RAG: pull captions, chunk,
embed, top-*k* retrieve. memovox ingests video at the level of *meaning* — and is
built so that **every assertion resolves to `(video, [t_start, t_end], modality,
confidence)`**. No orphan facts. If memovox can't say where it learned something,
it doesn't claim it.

> *Jaeger-LeCoultre's Memovox (1950) was the wristwatch that spoke back — an alarm
> that gave time a voice. This is the memory that speaks back: voice in, knowledge out.*

This repository implements **Phases 0–3** of [`spec.md`](spec.md): the audio
skeleton ("a genuinely useful tool on its own"), the **tri-modal visual track**
(content-aware scene detection, information-gain keyframe selection, on-screen
text/caption fusion), the **verified temporal knowledge graph** (claim extraction
→ NLI gate → entities/speakers resolved across the corpus → graph retrieval), and
**cross-corpus synthesis** — topic induction, consensus scoring, claim-evolution
tracking, and a literature-review synthesis that surfaces what sources agree and
disagree on.

## Local-first and free by default

The core runs on the **Python standard library alone** — no GPU, no model
downloads, no API keys. This is possible because every model slot has a
deterministic fallback behind a common backend interface (spec §7):

| Slot | Free fallback (always works) | Optional upgrade |
|------|------------------------------|------------------|
| ASR (speech→text) | `captions` (parse VTT/SRT) · `fake` (tests) | `faster-whisper` (`[asr]`) |
| Acquire | local file / transcript | `yt-dlp` (`[acquire]`) |
| Embedder | hashing embedder (deterministic) | `sentence-transformers` BGE-M3 (`[embed]`) |
| NLI gate | lexical-entailment | DeBERTa-NLI (`[nli]`) |
| LLM (extract/synthesize) | rule-based extractor + extractive answers | Ollama / local (`[llm]`) |
| Vector / lexical / graph | one embedded SQLite (FTS5 + BLOB vectors + edge tables) | LanceDB / Kùzu |

So you can ingest a local lecture recording (or even just its `.vtt`) and ask
grounded, cited questions **right now, for free** — then `pip install` a real ASR
or embedder to raise quality, with no code changes.

## Architecture (spec §3)

```
source ─▶ Stentor (acquire · demux · ASR · diarize)
              └─▶ Escapement (temporal fusion → Moments)
                      └─▶ Assay (claim extraction → NLI gate → typing)
                              └─▶ Loom (vector + lexical + graph indices)
                                      └─▶ Augur (plan → hybrid retrieve → cited answer)
                                              └─▶ CLI · Python SDK · REST · MCP server
```

| Codename | Subsystem | Module |
|----------|-----------|--------|
| **Stentor** | acquisition + ASR + diarization | `memovox.stentor` |
| **Tessera** | visual track (keyframes, OCR, VLM) | `memovox.tessera` |
| **Escapement** | temporal fusion into Moments | `memovox.escapement` |
| **Assay** | claim extraction + verification | `memovox.assay` |
| **Loom** | indices + knowledge graph + synthesis | `memovox.loom` |
| **Augur** | agentic retrieval + cited answers | `memovox.augur` |

## Install

```bash
pip install -e .                 # core only — stdlib, runs immediately
pip install -e ".[asr,embed]"    # add offline Whisper + dense embeddings
pip install -e ".[acquire]"      # add yt-dlp URL ingestion
```

Or run without installing: `python -m memovox --help`.
`ffmpeg`/`ffprobe` are recommended (used for demux + validation) but the pipeline
can ingest existing transcripts without them.

## Quick start

```bash
# Ingest a local media file (uses ffmpeg + ASR if available)...
memovox ingest ~/talks/scaling-laws.mp4 --title "Scaling laws talk"

# ...or ingest an existing transcript with no models at all (fully free):
memovox ingest ~/talks/scaling-laws.en.vtt --source-url https://youtu.be/abc123

# Ask a grounded question — every answer sentence carries a citation:
memovox ask "what chunk size did they recommend, and who said it?"

# Cross-corpus synthesis: run the consolidation pass after ingesting, then
# synthesize what the whole library says about a topic (consensus + disagreements):
memovox consolidate
memovox synthesize "scaling laws"

# Surface disagreements, or trace how a position changed over time:
memovox contradictions --topic "scaling laws"
memovox evolution --entity "Chinchilla"

# Human-readable per-video digest:
memovox export --video yt:abc123 --format md

memovox list            # what's ingested
memovox backends        # which real backends are installed
memovox stats           # store summary
```

`consolidate` is the cross-corpus background job (spec §4 stage 7): topic
induction, contradiction/agreement detection, consensus scoring, and dedup. It
is kept off the per-video ingest path — run it after ingesting new sources.

### Python SDK (spec §8)

```python
from memovox import Memovox

mv = Memovox(store="~/knowledge")
mv.ingest("~/talks/scaling-laws.en.vtt", source_url="https://youtu.be/abc123")
ans = mv.ask("what's the recommended chunk size, and who recommended it?")
print(ans.text)
for c in ans.citations:
    print(c.video_id, c.t_start_s, c.deep_link)
```

### MCP server (agent-native)

```bash
memovox mcp        # speaks MCP over stdio — wire into Claude Code / Desktop
```

Implemented with the standard library (no `mcp` package required): tools
`ingest_video`, `search_knowledge`, `get_claim_provenance`, `synthesize_topic`,
`find_contradictions`, `consolidate`.

## Design principles (non-negotiable, spec §2)

- **Provenance is sacred** — every fact resolves to a timestamped, modality-tagged span.
- **Verify before commit** — claims are entailment-checked against their source before entering the graph.
- **Idempotent ingestion** — content-hash IDs; re-ingesting merges instead of duplicating.
- **Model-agnostic, local-first** — swappable backends; fully offline mode.
- **Human-readable substrate** — `cat` your knowledge (Markdown digests + a SQLite DB you can inspect).

## Development

```bash
make test     # stdlib unittest (no pytest required)
make lint     # ruff, if installed
```

## Status

Phases 0–3 are implemented and tested with stdlib fallbacks:

- **Phase 0** — audio skeleton: ingest → Moments → verified claims → hybrid cited
  retrieval → CLI/SDK/REST/MCP.
- **Phase 1** — visual track: scene detection → information-gain keyframes →
  VLM-caption/OCR slots → tri-modal Moment fusion (degrades gracefully to a no-op
  with no video stream).
- **Phase 2** — verified temporal knowledge graph: claim extraction + a real NLI
  gate, cross-corpus entity/speaker resolution, `MENTIONS`/`ELABORATES`/`CORRECTS`
  edges, and a graph-expansion retrieval leg fused into RRF.
- **Phase 3** — cross-corpus synthesis: topic induction + `ABOUT` edges, claim
  clustering + consensus scoring, claim-evolution tracking, contradiction/agreement
  detection, a `consolidate` background job (with the supersede/dedup lifecycle),
  and a corpus-level `synthesize` literature review.

The whole pipeline runs free/stdlib-only and is gated by a golden-corpus eval
(`python -m eval.harness --assert-thresholds`). Phase 4 (scale & polish:
subscriptions/incremental sync, answer-with-video clip stitching, ColPali visual
retrieval, dashboards, named production backends) is next. See [`spec.md`](spec.md).

## License

MIT — see [LICENSE](LICENSE).
