# memovox — stress-test video corpus

A curated set of **real, verified** YouTube videos for exercising and breaking memovox.
All links resolved live via `yt-dlp` on 2026-06-16 (durations/channels confirmed).
Working doc — **untracked**, not part of the repo.

## Why these / how it's organized

memovox is a pipeline, and each stage fails differently:

```
Stentor (ASR/diarize) → Tessera (keyframes/OCR/VLM) → Escapement (fusion→Moments)
  → Assay (claim extraction/NLI/typing) → Loom (synthesis) → Augur (cited answers)
```

Videos are grouped by **which stage they punish**.

## Setup gotchas (determines what you're actually testing)

- **Captions are a prior by default** (`captions_as_prior=True`). Ingesting a YouTube URL
  tests *YouTube's* transcript, not memovox's ASR. To stress **Stentor/ASR**, force it
  with `--asr faster-whisper` and ingest the audio rather than supplying captions.
- **Visual analysis is off** unless you pass `--with-video`, and OCR/VLM are no-ops on the
  free path. To exercise **Tessera** you need the `tesseract` program (on-screen text) +
  a local Ollama vision model (frame descriptions).
- **Cross-video features need `memovox consolidate` first**, and `find_contradictions` is
  **cross-video only** (same-video pairs excluded) — a single debate won't surface
  contradictions; you need 2+ videos.
- Key caps to watch: `frame_max=1200` (silent frame drop on long/slide-heavy video),
  free-path diarization collapses every speaker to `spk_0`, `find_contradictions`
  `max_claims=600`, consensus/synthesis is token-Jaccard (synonyms don't cluster on the
  free path — clean A/B for the `[embed]` BGE-M3 upgrade).

---

## 1. Sweet spot — show it working
Clear single speaker, structured factual claims, captions exist.

| Video | Len | Channel | Exercises |
|---|---|---|---|
| https://youtu.be/UF8uR6Z6KLc | 15:04 | Stanford | Clean baseline, narrative/quotable claims |
| https://youtu.be/zjkBMFhNj_g | 59:48 | Andrej Karpathy | Flagship good demo: dense claims + slides + jargon |
| https://youtu.be/XTeJ64KD5cg | 9:15 | Numberphile | Definitions + numbers → claim extraction & typing |

## 2. Stress the ASR (Stentor) — `--asr faster-whisper`, no captions
Glossary/decoder-biasing is wired but **empty by default** (jargon misrecognized);
free-path diarization → `spk_0`; language auto-detected.

| Video | Len | Channel | Attack |
|---|---|---|---|
| https://youtu.be/TkPNgw2VUbU | 1:52:54 | (Žižek reupload) | Heavy accent + fast + tics + philosophy jargon |
| https://youtu.be/t3H5D-XxPrI | 12:27 | Wouter Corduwener | 21-language code-switching kills auto-detect |
| https://youtu.be/144uOfr4SYA | 1:47:20 | Munk/PRST | 4 speakers, overlap → `spk_0` collapse, wrong "who said it" |
| https://youtu.be/nkG2SSzPUns | 19:49 | NPR Music | Music beds → `[Music]` filler, lyrics-as-speech |
| https://youtu.be/AmlF6xq2SaQ | 2:56 | Easy Languages | Noisy, mixed accents, short overlapping utterances |

## 3. Stress the vision track (Tessera) — `--with-video` + tesseract + Ollama vision
Frames hard-capped at 1,200 (silent drop). Tests **shown vs. spoken** tagging and whether
it captures what narration omits.

| Video | Len | Channel | Attack |
|---|---|---|---|
| https://youtu.be/J7DzL2_Na80 | 39:49 | MIT OpenCourseWare | Chalkboard **handwriting** → OCR fails; math shown not said |
| https://youtu.be/WUvTyaaNkzM | 17:04 | 3Blue1Brown | Meaning in animation; narration referential ("this area here") |
| https://youtu.be/jbkSRLYSojo | 4:47 | BBC | The **data is the content** — animated chart, numbers on screen |
| https://youtu.be/kCc8FmEb1nY | 1:56:20 | Andrej Karpathy | ~2hr code screen-recording → OCR + frame-cap churn |
| https://youtu.be/jGwO_UgTS7I | 1:15:20 | Stanford Online | Slides + whiteboard, long → OCR + frame cap |

> Rosling / 3B1B / Strang double as a **correctness test**: ask a question whose answer is
> *only on screen* with the visual track **off** — a provenance-honest system should
> decline, not hallucinate.

## 4. Stress duration / scale (Escapement + frame cap)

| Video | Len | Channel | Attack |
|---|---|---|---|
| https://youtu.be/1M3Vdl6DRkU | 2:53:42 | Lex Fridman | ~3hr → many Moments, real ASR cost; with `--with-video` ~10,800 frames → capped to 1,200 (~89% dropped silently) |

(Karpathy GPT 1:56, Munk 1:47, Žižek 1:52 are also long-form.)

---

## 5. Flagship cross-video features (run `memovox consolidate` first)

### `find_contradictions` — same-proposition pairs (near-identical text, opposite polarity → fires the NLI gate)
- Saturated fat **causes** heart disease: https://youtu.be/aqBHXNGKvKU (6:44, Metabolic Mind)
- Saturated fat **doesn't** (Attia/Layman): https://youtu.be/dzOTaNwiFmA (8:29, Peter Attia MD)
- Breakfast **is** most important meal: https://youtu.be/AxIOGqHQqZM (10:08, PBS Origins)
- **Skip** breakfast (Berg): https://youtu.be/2su8e-nhMGw (3:55, Dr. Eric Berg DC)
- Carnivore risks: https://youtu.be/n_Smy5-1cHE (15:26, Adapt Your Life) — looser pair with —
- Vegan/brain: https://youtu.be/SdnEbJZoNg8 (5:03, BBC Global)

### `synthesize_topic "AGI"` — ingest the AI set
Karpathy LLMs (`zjkBMFhNj_g`) + Munk Debate (`144uOfr4SYA`) + the two timeline videos below.
Exposes token-Jaccard weakness: "AGI" / "superintelligence" / "human-level AI" won't
cluster on the free path → A/B against the `[embed]` upgrade.

### `claim_timeline --entity "AGI"` — dated videos on one entity (sorts by `published_at`)
- *We're Not Ready for Superintelligence*: https://youtu.be/5KVDDfAkRgc (34:11, AI In Context)
- *What happened with AGI timelines in 2025*: https://youtu.be/YeRS4TbtZWA (25:39, 80,000 Hours)

---

## 6. Adversarial / "does it over-claim?"
- **Sarcasm / rhetorical setups** ("you'd think X — but no") → polarity errors. Žižek + Munk Debate.
- **Recombination bypass**: ingest Karpathy's LLM talk, then ask a trick question recombining
  real tokens into a false statement — lexical NLI (`entailment_threshold=0.5`) can be fooled;
  the DeBERTa `[nli]` upgrade should catch it.
- **Attribution**: in the debate, ask "who said X?" — free-path `spk_0` collapse can't tell you.
- **Non-YouTube evil inputs**: silent/music-only clip; 10h+ stream (frame cap + Moment
  explosion); transcript-only ingest with no media (visual track must no-op, not crash).

---

## Copy-paste ingest blocks

```bash
# Sweet spot
for u in UF8uR6Z6KLc zjkBMFhNj_g XTeJ64KD5cg; do memovox ingest "https://youtu.be/$u"; done

# ASR stress (force whisper; long ones take a while)
for u in TkPNgw2VUbU t3H5D-XxPrI 144uOfr4SYA nkG2SSzPUns AmlF6xq2SaQ; do
  memovox ingest "https://youtu.be/$u" --asr faster-whisper
done

# Visual stress (needs tesseract + Ollama vision)
for u in J7DzL2_Na80 WUvTyaaNkzM jbkSRLYSojo kCc8FmEb1nY jGwO_UgTS7I; do
  memovox ingest "https://youtu.be/$u" --with-video
done

# Duration / scale
memovox ingest "https://youtu.be/1M3Vdl6DRkU"            # add --with-video to hit the frame cap

# Contradictions (then: memovox consolidate; memovox contradictions --topic "saturated fat")
for u in aqBHXNGKvKU dzOTaNwiFmA AxIOGqHQqZM 2su8e-nhMGw n_Smy5-1cHE SdnEbJZoNg8; do
  memovox ingest "https://youtu.be/$u"
done

# AGI synthesis + timeline (then: memovox consolidate)
for u in zjkBMFhNj_g 144uOfr4SYA 5KVDDfAkRgc YeRS4TbtZWA; do memovox ingest "https://youtu.be/$u"; done
# memovox synthesize "AGI"; memovox evolution --entity "AGI"
```
