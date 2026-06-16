# Stress report — iter8-content-free+fastapi (nli=lexical)

- ingest: **21/21** ok
- moments=4443 committed_claims=5414 trivial=3 (mean ratio **0.001**)
- provenance violations: **0**
- ask hit-rate: **1.0** | refusals_ok: 14/14

## Findings
- **[MED] synthesis_synonyms** — synthesize('AGI') found 0 consensus points (token-Jaccard misses synonyms?)
- **[LOW] contradictions_missed** — 2/2 planted pairs not surfaced (lexical-NLI limit + corpus caveat)
- **[LOW] speaker_collapse** — multi-speaker debate collapsed to a single speaker (no diarization backend)

## Per-video
| id | group | claims | trivial% | prov! | speakers | caps |
|---|---|---|---|---|---|---|
| UF8uR6Z6KLc | sweet | 87 | 0.0 | 0 | 1 | 0 |
| zjkBMFhNj_g | sweet | 392 | 0.0 | 0 | 1 | 0 |
| XTeJ64KD5cg | sweet | 113 | 0.0 | 0 | 4 | 0 |
| t3H5D-XxPrI | asr | 31 | 0.0 | 0 | 1 | 0 |
| 144uOfr4SYA | asr | 657 | 0.0 | 0 | 1 | 0 |
| nkG2SSzPUns | asr | 179 | 0.0 | 0 | 1 | 0 |
| AmlF6xq2SaQ | asr | 14 | 0.0 | 0 | 1 | 0 |
| J7DzL2_Na80 | visual | 326 | 0.0 | 0 | 1 | 0 |
| WUvTyaaNkzM | visual | 114 | 0.0 | 0 | 1 | 0 |
| jbkSRLYSojo | visual | 26 | 0.0 | 0 | 1 | 0 |
| kCc8FmEb1nY | visual | 742 | 0.0 | 0 | 1 | 0 |
| jGwO_UgTS7I | visual | 492 | 0.0 | 0 | 1 | 0 |
| 1M3Vdl6DRkU | duration | 1388 | 0.0 | 0 | 1 | 0 |
| aqBHXNGKvKU | contradiction | 52 | 0.0 | 0 | 1 | 0 |
| dzOTaNwiFmA | contradiction | 52 | 0.0 | 0 | 1 | 0 |
| AxIOGqHQqZM | contradiction | 82 | 0.0 | 0 | 3 | 0 |
| 2su8e-nhMGw | contradiction | 25 | 0.0 | 0 | 1 | 0 |
| n_Smy5-1cHE | contradiction | 97 | 0.0 | 0 | 1 | 0 |
| SdnEbJZoNg8 | contradiction | 30 | 0.0 | 0 | 1 | 0 |
| 5KVDDfAkRgc | timeline | 320 | 0.0 | 0 | 1 | 0 |
| YeRS4TbtZWA | timeline | 195 | 0.015 | 0 | 3 | 0 |
