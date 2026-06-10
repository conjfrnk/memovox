# Context-aware ASR glossary ("know who you're listening to")

**Date:** 2026-06-09
**Status:** Approved design, pending implementation plan

## Motivation

A Whisper re-ingest of `yt:XPlkaqXgeOA` (Jun Yuh, creator-economy YouTuber) mis-heard
brand names a knowledgeable human viewer would have gotten right: "creative college"
for **Creator College**, "creator live" for **Creator Life**, "nutonic" for
**Neutonic**, plus a hallucinated "canes festival" replacing "Tokyo, Japan". The
captions backend got the brands right but misspelled Neutonic ("Nutanic").

A human guesses these terms correctly because they bring *prior knowledge of who the
speaker is* to the listening. This feature replicates that: before Whisper runs,
memovox gathers what is knowable about the channel/creator, synthesizes the
vocabulary they are likely to speak, and biases the decoder with it via
faster-whisper's `initial_prompt` (the existing `--glossary` mechanism, today
manual-only).

## Goal and success criteria

- Whisper ingests are biased with creator-specific vocabulary with **zero
  configuration** and **zero new network egress** by default.
- With the web tier opted in, discovery extends to terms not present verbatim in
  local metadata.
- **Acceptance test:** on a Whisper re-ingest of `yt:XPlkaqXgeOA`, the offline tiers
  alone (channel + description context, LLM synthesis) yield "Creator College" in
  the transcript; with `--glossary-web` the glossary additionally contains the
  exact spelling "Neutonic" (via description-link fetch or Wikipedia), and the
  transcript renders it.
- Ingestion must never fail or block because of this feature (fail-soft everywhere).

## Non-goals

- No post-ASR transcript rewriting or near-miss flagging (possible follow-up).
- No paid or key-requiring APIs. No new third-party dependencies.
- Nothing changes on the captions ASR path; the glossary is built only when the
  resolved backend is `whisper`.
- No change to manual `--glossary` semantics; manual terms always survive ranking.

## Where it hooks in

In `ingest`, after acquisition produces `SourceMeta`, and only when
`resolve_asr_backend(...)` returns `"whisper"`. The result is merged with manual
`--glossary` terms and passed through the existing
`run_asr(glossary=...)` → `options["glossary_prompt"]` → `initial_prompt` path.

## Architecture

New module `src/memovox/stentor/glossary.py` with three pure stages mirroring the
human reasoning. A `build_glossary(config, settings, meta, store, manual_terms)`
orchestrator returns `list[str]` plus per-source provenance.

### Stage 1 — gather context ("what do I know about this channel?")

Produces a `GlossaryContext` (dataclass): blocks of text + candidate entity labels,
each tagged with its source.

Local tier (always on when `glossary_auto=True`):

- `SourceMeta` title, channel, description, tags (already captured by acquire;
  `extra["description"]`).
- The store: entities and stored glossaries from previously ingested videos of the
  same channel.

Web tier (only when `glossary_web=True` AND `local_only=False`), modeled on
`WikidataLinker` (reachability pre-check, ~3s timeouts, fail-soft to nothing):

- **Description-link fetch:** extract http(s) URLs from the video description, fetch
  at most 5 with stdlib `urllib`, accept only `text/html`, cap response size
  (256 KB), no auth, strip to `<title>`, og-tags, and `<h1>`s. Creators' own links
  are the most reliable spelling source (a Neutonic sponsor link resolves to a page
  titled "Neutonic").
- **Channel About metadata** via yt-dlp (existing dependency, keyless).
- **Wikipedia opensearch + Wikidata search** on the channel/person name; keep
  summary text and associated entity labels.

URL-fetch safety: http/https schemes only, no cookies, no redirects across scheme,
size + count + time caps. Fetched text is untrusted data — used only as candidate
text, never executed or echoed into shell/SQL.

### Stage 2 — synthesize ("so what words will they likely say?")

- **Primary (LLM):** the existing ollama LLM backend receives the gathered context
  and instructions to act as a viewer who knows this creator — combining context
  with the model's own world knowledge — and returns a JSON array of proper
  nouns/jargon likely to be *spoken aloud* in the video. Output is schema-validated
  (array of short strings); a parse failure falls through to the miner.
  Untrusted-context note: description/web text goes into the prompt as quoted data;
  the output is only ever a validated word list, capping injection blast radius.
- **Fallback (deterministic miner):** capitalized n-gram extraction (1–3 grams) +
  frequency/position heuristics over the gathered text, stdlib-only. Runs whenever
  the LLM backend is unavailable (`--llm none`, ollama down). This path is what
  eval gates exercise, since it is reproducible.

### Stage 3 — rank and cap

- Order: manual `--glossary` terms first (never dropped), then candidates by source
  weight — own-corpus 1.0 > description/fetched-page 0.8 > Wikipedia/Wikidata 0.6 >
  LLM-only inference 0.4 — then by frequency.
- Case-insensitive dedupe (keep the best-cased variant: prefer the source's casing
  over lowercase).
- Cap to ≤ 48 terms and ≤ 150 tokens (whisper tokenizer estimate via len/4
  heuristic), leaving headroom inside faster-whisper's ~224-token
  `initial_prompt` window.

## Settings / CLI

- `glossary_auto: bool = True` — offline tiers run whenever the whisper backend is
  selected. Env: `MEMOVOX_GLOSSARY_AUTO`.
- `glossary_web: bool = False` — enables the web tier. Env: `MEMOVOX_GLOSSARY_WEB`.
  CLI: `--glossary-web` (top-level, like `--allow-cpu`).
- `local_only=True` always wins over `glossary_web=True` (skip web silently).

## Observability & provenance

- New trace span `stage="glossary"` with counters: candidates per source,
  `kept`, `tokens`, and caps (mirroring the visual span's `caps` shape).
- The final term list + provenance is stored on the video record
  (`asr_glossary` JSON) and surfaced in the digest header as
  "ASR biased with: term (source), …" so citation audits can see what influenced
  the decoder.

## Error handling

Every tier fails soft to an empty contribution: dead links, offline Wikipedia,
ollama down, JSON parse failures — none may raise out of `build_glossary`. The
only deliberate fail-louds in the ASR path remain the existing
DevicePlacementError/DemuxError/BudgetExceeded set. Web-tier exceptions are
recorded in the trace span as a `degraded` counter, not raised.

## Testing

Hermetic throughout (matching the Phase-4 test contract — no network, no model
downloads in `make test`):

- Unit: miner, rank/cap, URL extraction, dedupe-casing — fixture metadata.
- Synthesis: `fake`-style LLM stub returning canned JSON; malformed-JSON fallback.
- Web tier: mocked HTTP (urllib opener injection), size/count/timeout cap tests,
  `local_only` precedence test.
- Integration: ingest with `--asr whisper` + fake ASR asserting `glossary_prompt`
  contains auto-mined terms merged after manual ones.
- Benchmark (non-gating, real corpus): brand-term WER on `yt:XPlkaqXgeOA` with the
  feature off/on (offline) /on (web) — reported as "glossary lift".

## Determinism note

The LLM tier is inherently nondeterministic; CI eval gates therefore pin
`--llm none` to exercise the miner path. The LLM path is covered by stubbed unit
tests and the (reported, non-gating) benchmark.
