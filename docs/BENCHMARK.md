# memovox benchmark methodology

This documents how memovox measures the two claims that distinguish it from
transcript-only "chat with your video" tools. It is deliberately honest about what
is **measured** vs **assumed**, and the runner + protocol are public so anyone can
reproduce or contest the numbers.

> **Status:** methodology + runner are in place (`eval/benchmark.py`, `make
> benchmark-corpus`, fixtures in `eval/benchmark/`). The headline numbers are
> produced by a real run on a configured machine (see *Running it*) — they are a
> dated snapshot and are **not** committed as a CI invariant.

## The two claims under test

1. **Visual fusion adds knowledge the transcript doesn't have.** Measured as the
   **shown-only visual lift**: accuracy on questions whose answer is *on screen but
   never spoken*, with the visual track OFF (transcript-only) vs ON (`--with-video`).
   A large positive delta is the empirical case for fusing OCR/keyframes with speech.
2. **It refuses rather than confabulates.** On adversarial questions whose answers
   are *not in the corpus*, memovox should decline (`low_evidence`) instead of
   fabricating — the rule it never breaks ("if it can't cite it, it won't claim it").

## Ground-truth design

Build a small, license-clean corpus (see [`benchmark/SOURCES.md`](benchmark/SOURCES.md)
— CC-BY / public-domain only) and a QA set where each item is tagged with:

- `expects`: `present` (answer is in the corpus) or `absent` (it isn't);
- `modality`: `speech-only` | `shown-only` | `both` | `none`;
- `answer_substrings`: strings that must appear in a correct answer.

**Shown-only** is the load-bearing tag: the answer must appear on a slide/figure and
**not** be spoken. Those are the only questions that can attribute a lift to the
visual track. Include `speech-only` **present** controls (answerable with video OFF)
so the A/B is honest, and `absent` adversarial items that are *lexically adjacent* to
present content (e.g. ask for a "fine-tuning context length" when only the "retrieval
context length" is stated) — the trap that separates grounded refusal from confident
fabrication.

## Scoring (see `eval/benchmark.py`)

Each (condition, question) response is bucketed:

| | refused (`low_evidence`) | answered, substring hit | answered, no hit |
|---|---|---|---|
| **present** | `refused` (over-cautious) | `correct` | `wrong` |
| **absent**  | `refused` (**correct**) | — | `confabulated` |

Headline metrics:

- **shown-only lift** = `accuracy(with_video, shown-only) − accuracy(audio_only, shown-only)`
- **correct-refusal rate** and **confabulation rate** over `absent` questions
- **present-control accuracy** (proves refusals are discriminating, not blanket "I
  don't know")
- **`ocr_unverified` flag rate** on recovered shown-only answers (should be 1.0 — the
  visual path is honestly marked unverified; see [`../SECURITY.md`](../SECURITY.md))

## Head-to-head refusal protocol (vs incumbents)

To compare refusal behavior against NotebookLM / Otter / Fireflies (with Perplexity as
a *web-grounded reference*, not a peer), follow this pre-registered protocol so it
can't be dismissed as cherry-picked:

1. **Pre-register** the corpus, question list, and rubric (commit them) before running.
2. Build a **salted** corpus: state exactly one of each easily-confused pair, never the
   other; keep a PRESENT/ABSENT ground-truth table.
3. Questions: ≥60/40 absent/present, neutral phrasing, 2–3 paraphrases each.
4. Ingest the **identical** corpus into each tool; verify each can answer the PRESENT
   controls before scoring (a mis-ingested tool isn't "virtuously refusing").
5. Ask under a **context-clearing** discipline (fresh chat/thread per question).
6. Score into the buckets above; for any cited answer, **open the citation** and check
   it supports the claim (a citation to an unsupporting span is a *hallucinated
   citation* — the worst failure).
7. Report each rate with a **95% Wilson CI**, two blind raters + tie-breaker (Cohen's
   κ), and a prompt-sensitivity flip-rate.

**Automatability:** memovox is fully scriptable (`memovox ask --json` → boolean
`low_evidence` + structured citations; refusal needs zero human judgment). Fireflies
(GraphQL `createAskFredThread`) and Perplexity (Sonar API) automate; **NotebookLM has
no consumer API** (manual UI + screenshots) and Otter's free tier is import-capped.
Run memovox in **both** extractive (cannot fabricate by construction) and
generative-LLM modes so it isn't "safest mode vs everyone's only mode."

## Caveats (state these with any result)

- **Small n** → wide CIs; this is an existence demonstration of a behavioral
  difference, not a precise leaderboard.
- **Moving targets** → hosted competitors change without notice; stamp date + model
  versions; keep memovox pinned to a commit.
- **Cherry-picking** → the salted corpus is adversarial by design; publish it, include
  present controls, and have someone outside the project write some questions.
- **Ingestion non-equivalence** → tools chunk/transcribe differently; feed identical
  text where possible and exclude items that fail the PRESENT controls.
- **ToS** → make headline NotebookLM/Otter numbers from legitimate manual runs;
  unofficial automation only as a clearly-labeled side lane.

## Running it

See [`benchmark/README.md`](benchmark/README.md). In short: install ffmpeg + tesseract
+ ASR (`pip install -e ".[asr]"`), download license-vetted media into
`eval/benchmark/media/`, author + **verify** `eval/benchmark/qa.json`, then:

```bash
make benchmark-corpus OUT=bench.json
```

Publish the corpus, the QA set, and `bench.json` alongside any numbers you report.
