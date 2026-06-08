# Real-corpus benchmark

Produces memovox's two headline numbers on **real, license-vetted** videos:

1. **Shown-only visual lift** — accuracy on questions whose answer is on screen but
   never spoken, transcript-only vs `--with-video`.
2. **Refusal vs confabulation** — on out-of-corpus questions, does it refuse rather
   than fabricate?

This is **separate from `make test` / the CI gates** (`eval/harness.py`): it needs a
connected machine, `ffmpeg` + `tesseract`, and real media, and its numbers are a
**dated snapshot**, not a determinism invariant.

## Requirements

- `ffmpeg`/`ffprobe` and `tesseract` on `PATH` (keyframes + OCR).
- ASR for the speech track: `pip install -e ".[asr]"` (faster-whisper). For a clean
  A/B the transcript must be **identical** across conditions, so both conditions
  ingest the same media and differ only in the visual track. Do **not** feed an
  official `.vtt` to one condition and ASR to the other — that confounds transcript
  quality with the visual lift.
- A local LLM via Ollama is optional (richer answers); the free extractive path works.

## 1. Get media (see [`../../docs/benchmark/SOURCES.md`](../../docs/benchmark/SOURCES.md))

Download into `media/` (gitignored). Example — FOSDEM 2025, CC-BY:

```bash
mkdir -p eval/benchmark/media
curl -L -o eval/benchmark/media/fosdem25-riscv.mp4 \
  https://video.fosdem.org/2025/h1309/fosdem-2025-5678-how-good-is-risc-v-comparing-benchmark-results.mp4
```

## 2. Author the QA set (the real work)

```bash
cp eval/benchmark/manifest.example.json eval/benchmark/manifest.json
cp eval/benchmark/qa.example.json eval/benchmark/qa.json
```

Each QA item: `{q, expects: present|absent, modality: speech-only|shown-only|both|none,
answer_substrings}`. For **shown-only** items the answer must appear **on screen and
not be spoken** — watch the moment and confirm. The example answers are research
suggestions and **must be verified** before you publish any number.

## 3. Run

```bash
make benchmark-corpus                 # prints a markdown table
make benchmark-corpus OUT=bench.json  # also writes the full report JSON
# or:
python -m eval.benchmark --manifest eval/benchmark/manifest.json \
    --qa eval/benchmark/qa.json --json bench.json
```

Publish the corpus, the QA set, and the raw report alongside any numbers (see the
refusal-protocol caveats in [`../../docs/BENCHMARK.md`](../../docs/BENCHMARK.md)) so
results are reproducible.
