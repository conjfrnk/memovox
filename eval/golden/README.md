# Golden eval corpus

A tiny, hand-labeled corpus that exercises memovox's knowledge + trust layers:
retrieval (QA), entity resolution (shared + unique entities), speaker resolution
(shared + distinct speakers), and contradiction detection. Everything here is
designed to be detectable by the **free / stdlib path** (hashing embedder,
lexical NLI, rule-based claim extraction — no LLM, no network).

## Logical video ids

**Logical video ids in the label files are the transcript filename stems
(`talk_a`, `talk_b`); the eval harness maps them to store video_ids on ingest.**

memovox derives a store `video_id` of the form `vid:<content_hash[:16]>` at
ingest time (`make_video_id` in `util.py`), which is not human-readable and not
known until the file is ingested. To keep the labels decoupled from ingest, the
JSON label files refer to videos by their filename stem only.

## Transcripts

WEBVTT files, mirroring `examples/scaling_laws.en.vtt`. Speakers are marked with
`<v Speaker Name>` voice tags (parsed by `src/memovox/stentor/transcript.py`).
Bracketed events (`[applause]`) are stripped to timeline markers; filler words
are stripped from the knowledge text.

- **`talk_a.en.vtt`** — Dr. Lee, optimistic talk on scaling laws. Mentions the
  **Transformer** architecture and the **Chinchilla** study; claims that
  *scaling laws will continue to hold beyond current compute budgets*.
- **`talk_b.en.vtt`** — A follow-up with two speakers: **Dr. Lee** (same person
  as in talk_a) and a new speaker **Prof. Kim** (appears only here). Mentions
  **Chinchilla** (shared with talk_a) and the **Llama** model family (unique to
  talk_b). Prof. Kim claims that *scaling laws will NOT hold beyond current
  compute budgets; they break down* — a direct contradiction of talk_a.

### Shared / distinct design

- **Shared entity:** `Chinchilla` (both talks). **Unique:** `Transformer`
  (talk_a only), `Llama` (talk_b only).
- **Shared speaker:** `Dr. Lee` (both talks) → must resolve to one person.
  **Distinct speaker:** `Prof. Kim` (talk_b only) → must NOT be merged with
  Dr. Lee.
- **Contradiction:** one shared topic ("scaling laws will hold beyond current
  compute budgets") asserted with opposite polarity across the two talks.

## Label files

- **`qa.json`** — list of
  `{ "q": str, "relevant_moment_substrings": [str], "answer_substrings": [str] }`.
  Every `answer_substrings` / `relevant_moment_substrings` entry appears verbatim
  in the **cleaned** speech text of some cue.
- **`entities.json`** — `{ "canonical": [str], "mentions": { "<logical_id>": [str] } }`.
- **`speakers.json`** — `{ "identities": { "<logical_id>:<raw_label>": "<canonical_identity_id>", ... } }`.
  Every speaker mention in the corpus is keyed by `"<logical_video_id>:<raw_label>"`
  (the `raw_label` matches the `<v ...>` voice tag exactly), and maps to a
  canonical identity id. The gold identity clusters are therefore:
  - `person:lee` = `{talk_a:Dr. Lee, talk_b:Dr. Lee}` — the **positive** cross-talk
    merge: Dr. Lee must resolve to one person across both talks.
  - `person:kim` = `{talk_b:Prof. Kim}` — the **negative** distinct identity:
    Prof. Kim must NOT be merged into Dr. Lee, and Dr. Lee must NOT be split
    across talks. A system that merges Kim into Lee, or splits Lee, is
    measurably wrong against this gold.
- **`contradictions.json`** — list of
  `{ "topic": str, "video_a": "<logical_id>", "video_b": "<logical_id>" }`.
  Note: `topic` is a **human-readable label** for the contradiction, NOT a
  verbatim transcript substring (unlike the substrings in `qa.json`, which are
  required to appear verbatim in the cleaned speech text).

## Phase 4 additions (M1.2)

- **`talk_c.en.vtt`** — a 3rd talk (speaker *Dr. Park*) that disputes talk_a's
  Transformer-is-the-foundation claim → a real cross-video `CONTRADICTS` edge
  (re-baselined into `contradictions.json`). The harness runs `mv.consolidate()`
  after ingest so these edges + `topic_id` exist before retrieval scoring.
- **`topics.json`** — gold topic clusters (logical moment ids that should share an
  induced topic); the `topic_f1` regression guard reads the persisted `topic_id`.
- **`gold_span`** (optional, per `qa.json` item) — a `[t0, t1]` the cited span is
  scored against (interval IoU) for the unified `span_accuracy` metric; absent →
  the gold moment's window is used.
- **`talk_vis.en.vtt` / `visual.json`** (M1.1) — the on-screen-only visual fixture;
  EXCLUDED from the scored corpus (`_NON_CORPUS_STEMS`), feeds only the ungated
  `multimodal` block.

## M2.2 additions

- **Multi-part `qa.json` items** carry a `subqueries` array (each with its own
  `q` + `relevant_moment_substrings`); the top-level `relevant_moment_substrings`
  is the union so `hit_rate` still credits the item. The agentic planner
  decomposes the question, retrieves each clause, and merges — `plan.subquery_recall`
  (ungated) is the fraction of clauses covered in the single composed answer.

## M3.4 — backend A/B benchmark

`python -m eval.harness --benchmark` (or `make benchmark`) ranks the available
`BackendConfig`s over the golden text metrics (`hit_rate`, `mrr`, `ndcg`,
`groundedness`, `contradiction.f1`, `synthesis.groundedness`). It **auto-shrinks to
the single FREE row** on a bare machine (the only row CI gates); upgrade rows
(`free+cross-encoder`, `st+deberta`) appear only when their optional deps are
installed. `--benchmark --json` emits machine-readable output; `--benchmark
--assert-no-regression` gates the FREE row on the existing thresholds.

**Scope caveat:** visual/OCR/VLM configs (e.g. `colpali+surya+qwen`) are declared
**unrankable** on this text corpus — they move only the ungated `multimodal` block,
not the ranked text metrics. They are reported with an explicit reason, never
silently scored 0.0.
