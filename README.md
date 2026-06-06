# memovox

**memovox turns videos into a searchable, trustworthy memory you can ask questions of.**
Point it at a talk, lecture, meeting, interview, webinar, or podcast and ask plain-English
questions later. Every sentence of the answer comes with a **receipt** — which video it
came from, the exact moment (with a link that jumps you straight there), and who said it.
The one rule it never breaks: **if memovox can't point to where it learned something, it
won't claim it.**

> **New here?** Start with the plain-language overview: **[What is memovox? →](docs/EXPLAINER.md)**
>
> **Heads up:** memovox is a **command-line tool** — you run it by typing commands in a
> terminal; there's no clickable app yet. (It also offers a Python library, a REST server,
> and an MCP server for wiring into AI assistants.)

## What it's good for

Anything you'd rather not re-watch — a lecture or podcast, or a meeting, interview, or
Zoom/Teams call you recorded yourself: find what was said (and shown) across long
recordings, and get answers you can verify. Across many videos it can also summarize a
topic, surface where sources **disagree**, and trace how an idea **changed over time**. It
works on any recording with speech — the subject matter doesn't matter.

## Free, private, and local-first

Hand memovox a transcript and the whole thing runs **offline on your machine** — no
account, no API key, nothing uploaded. Your videos, transcripts, and answers never leave
your computer. The core needs only the Python standard library. Want higher quality? Drop
in optional, still-local upgrades (better transcription, smarter search, a local language
model) with no change to how you use it. Each model upgrade downloads its weights the first
time you use it, then runs locally; the optional local language model runs on
[Ollama](https://ollama.com) (a free app for running AI models on your own computer) that
you set up yourself.

| Step | Free — always works | Optional local upgrade |
|------|---------------------|------------------------|
| Speech → text (ASR) | use a transcript you already have (VTT/SRT) | transcribe audio yourself: faster-whisper (`[asr]`) |
| Get the video in | local file or transcript | download from a URL: yt-dlp (`[acquire]`) |
| Search by meaning (embeddings) | built-in deterministic hashing | sentence-transformers BGE-M3 (`[embed]`) |
| Fact-check each claim (NLI) | built-in lexical entailment | DeBERTa-NLI (`[nli]`) |
| Write the answer (LLM) | built-in rule-based + extractive answers | a local model via Ollama (run an Ollama server — no pip extra) |
| Read on-screen text & describe visuals | not in the free core | the `tesseract` program (on-screen text) + a local Ollama vision model (describe visuals) |
| Store & search | one SQLite file (full-text + vectors + links) | LanceDB / Kùzu |

> The fully-free path needs a **transcript** (most sites, including YouTube, let you
> download one). To transcribe raw audio or read what's on screen, add the matching
> upgrade above.

Free answers are concise and quoted straight from the source; add the embedding upgrade for
sharper search and a language model to rewrite the quotes into a single, natural-language answer. Run `memovox backends` to see what's active,
and swap any piece with `--asr` / `--embed` / `--nli` / `--llm`.

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

# Ingest from a URL ([acquire] extra). Audio-only by default; add --with-video to also
# analyze the picture (keyframes are free; reading on-screen text needs the tesseract program):
memovox ingest https://youtu.be/abc123 --with-video

# Ask a grounded question — every answer sentence carries a citation:
memovox ask "what chunk size did they recommend, and who said it?"

# Cross-corpus synthesis (run consolidate after ingesting new sources):
memovox consolidate
memovox synthesize "scaling laws"
memovox contradictions --topic "scaling laws"
memovox evolution --entity "Chinchilla"

memovox list        # what's ingested
memovox backends    # which optional upgrades are active
memovox stats       # store summary
```

*The Python SDK, REST, and MCP sections below are for developers building on memovox — if
you just want cited answers, the commands above are all you need.*

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

### REST server

```bash
memovox serve              # HTTP API on 127.0.0.1:8808 — stdlib, no extra needed
memovox serve --fastapi    # use FastAPI/uvicorn instead (pip install -e ".[serve]")
```

## How it's different (under the hood)

*A local-first, multimodal video-to-knowledge engine — meaning in, cited memory out.*

Most "chat with your video" tools are transcript-only **RAG** (retrieval-augmented
generation): they pull captions, chop them into chunks, embed them, and hand the closest
matches to a chatbot. Two costs: the chatbot can confidently state things that aren't in
the video, and everything you *see* — slides, charts, code — is thrown away.

memovox ingests at the level of *meaning* and guarantees that **every assertion resolves
to a specific source, time span, modality (spoken vs. shown), and confidence** — in code,
`(video, [t_start, t_end], modality, confidence)`. If it can't say where it learned
something, it doesn't claim it.

### Pipeline

You never need these names to use memovox — this is the internal flow, for developers:

```
source ─▶ Stentor (acquire · demux · ASR · diarize)
   └─▶ Tessera (keyframes · OCR · VLM)
        └─▶ Escapement (temporal fusion → Moments)
             └─▶ Assay (claim extraction → NLI gate → typing)
                  └─▶ Loom (vector + lexical + graph indices · synthesis)
                       └─▶ Augur (plan → hybrid retrieve → cited answer)
                            └─▶ CLI · Python SDK · REST · MCP
```

*ASR = speech-to-text · NLI = the fact-checking (entailment) step · OCR = reading text in
images · VLM = a vision model that describes frames.*

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

For design background, see [`docs/DESIGN.md`](docs/DESIGN.md) (how the pieces map to the
code) and [`spec.md`](spec.md) (the full specification).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE). Free and open source; note the GPL's copyleft
terms if you plan to redistribute. Copyright (C) 2026 Connor.
