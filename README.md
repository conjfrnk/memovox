# memovox

**A multimodal video-to-knowledge engine. Voice in, queryable — and *cited* — memory out.**

Most "chat with your video" tools are transcript-only RAG: pull captions, chunk,
embed, top-*k*. memovox ingests video at the level of *meaning* and guarantees that
**every assertion resolves to `(video, [t_start, t_end], modality, confidence)`**.
If memovox can't say where it learned something, it doesn't claim it.

## Local-first and free by default

The core runs on the **Python standard library alone** — no GPU, no model downloads,
no API keys. Every model slot has a deterministic fallback behind one backend
interface, so you can ingest a local recording (or just its `.vtt`) and ask grounded,
cited questions right now, for free — then `pip install` a real ASR or embedder to
raise quality, with no code changes.

| Slot | Free fallback (always works) | Optional upgrade |
|------|------------------------------|------------------|
| ASR | captions (VTT/SRT) · fake (tests) | faster-whisper (`[asr]`) |
| Acquire | local file / transcript | yt-dlp (`[acquire]`) |
| Embedder | deterministic hashing | sentence-transformers BGE-M3 (`[embed]`) |
| NLI gate | lexical entailment | DeBERTa-NLI (`[nli]`) |
| LLM | rule-based extractor + extractive answers | Ollama / local (`[llm]`) |
| Vector / lexical / graph | one SQLite (FTS5 + BLOB vectors + edges) | LanceDB / Kùzu |

## Architecture

```
source ─▶ Stentor (acquire · demux · ASR · diarize)
   └─▶ Tessera (keyframes · OCR · VLM)
        └─▶ Escapement (temporal fusion → Moments)
             └─▶ Assay (claim extraction → NLI gate → typing)
                  └─▶ Loom (vector + lexical + graph indices · synthesis)
                       └─▶ Augur (plan → hybrid retrieve → cited answer)
                            └─▶ CLI · Python SDK · REST · MCP
```

## Install

```bash
pip install -e .                 # core only — stdlib, runs immediately
pip install -e ".[asr,embed]"    # add offline Whisper + dense embeddings
pip install -e ".[acquire]"      # add yt-dlp URL ingestion
```

`ffmpeg`/`ffprobe` are recommended (demux + validation), but transcripts ingest
without them. Or run without installing: `python -m memovox --help`.

## Quick start

```bash
# Ingest a transcript with no models at all (fully free):
memovox ingest ~/talks/scaling-laws.en.vtt --source-url https://youtu.be/abc123

# Ingest from a URL ([acquire] extra). Audio-only by default; add --with-video
# to also analyze the visual track (keyframes · OCR · captions):
memovox ingest https://youtu.be/abc123 --with-video

# Ask a grounded question — every answer sentence carries a citation:
memovox ask "what chunk size did they recommend, and who said it?"

# Cross-corpus synthesis (run consolidate after ingesting new sources):
memovox consolidate
memovox synthesize "scaling laws"
memovox contradictions --topic "scaling laws"
memovox evolution --entity "Chinchilla"

memovox list        # what's ingested
memovox backends    # which real backends are installed
memovox stats       # store summary
```

### Python SDK

```python
from memovox import Memovox

mv = Memovox(store="~/knowledge")
mv.ingest("~/talks/scaling-laws.en.vtt", source_url="https://youtu.be/abc123")
ans = mv.ask("what's the recommended chunk size, and who recommended it?")
print(ans.text)
for c in ans.citations:
    print(c.video_id, c.t_start_s, c.deep_link)
```

### MCP server

```bash
memovox mcp     # speaks MCP over stdio — wire into Claude Code / Desktop
```

Stdlib-only (no `mcp` package required); tools: `ingest_video`, `search_knowledge`,
`get_claim_provenance`, `synthesize_topic`, `find_contradictions`, `claim_timeline`,
`consolidate`, `job_status`.

## Design principles

- **Provenance is sacred** — every fact resolves to a timestamped, modality-tagged span.
- **Verify before commit** — claims are entailment-checked against their source first.
- **Idempotent ingestion** — content-hash IDs; re-ingesting merges, never duplicates.
- **Model-agnostic, local-first** — swappable backends; a fully offline default.
- **Human-readable substrate** — Markdown digests over a SQLite DB you can inspect.

## Development

```bash
make test                                    # stdlib unittest (no pytest required)
make lint                                    # ruff, if installed
python -m eval.harness --assert-thresholds   # golden-corpus quality gates
```

## Status

Phases 0–4 of [`spec.md`](spec.md) are implemented and gated by a golden-corpus eval:
the audio skeleton; the tri-modal visual track; a verified temporal knowledge graph
(claim extraction → NLI gate → cross-corpus entity/speaker resolution → graph
retrieval); cross-corpus synthesis (topic induction, consensus scoring,
claim-evolution tracking, contradiction detection); and scale/polish (incremental
subscriptions, answer-with-video clip stitching, async jobs, named production-backend
slots). Optional ML backends are opt-in; the free stdlib path is the tested default.
See [`spec.md`](spec.md) and [`docs/DESIGN.md`](docs/DESIGN.md).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE). Copyright (C) 2026 Connor.
