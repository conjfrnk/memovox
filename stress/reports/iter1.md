# Stress report — iter1-fixes (nli=lexical)

- ingest: **21/21** ok
- moments=4443 committed_claims=5402 trivial=2902 (mean ratio **0.559**)
- provenance violations: **0**
- ask hit-rate: **1.0** | refusals_ok: 3/3

## Findings
- **[HIGH] trivial_claims** — mean trivial-claim ratio 0.559 (>0.25)
- **[MED] contradictions_missed** — 2/2 planted cross-video contradictions not surfaced
- **[MED] synthesis_synonyms** — synthesize('AGI') found 0 consensus points (token-Jaccard misses synonyms?)
- **[LOW] speaker_collapse** — multi-speaker debate collapsed to a single speaker (no diarization backend)

## Per-video
| id | group | claims | trivial% | prov! | speakers | caps |
|---|---|---|---|---|---|---|
| UF8uR6Z6KLc | sweet | 86 | 0.233 | 0 | 1 | 0 |
| zjkBMFhNj_g | sweet | 391 | 0.307 | 0 | 1 | 0 |
| XTeJ64KD5cg | sweet | 113 | 0.761 | 0 | 4 | 0 |
| t3H5D-XxPrI | asr | 30 | 0.933 | 0 | 1 | 0 |
| 144uOfr4SYA | asr | 655 | 0.412 | 0 | 1 | 0 |
| nkG2SSzPUns | asr | 178 | 0.921 | 0 | 1 | 0 |
| AmlF6xq2SaQ | asr | 13 | 0.615 | 0 | 1 | 0 |
| J7DzL2_Na80 | visual | 326 | 0.748 | 0 | 1 | 0 |
| WUvTyaaNkzM | visual | 114 | 0.64 | 0 | 1 | 0 |
| jbkSRLYSojo | visual | 26 | 0.577 | 0 | 1 | 0 |
| kCc8FmEb1nY | visual | 740 | 0.358 | 0 | 1 | 0 |
| jGwO_UgTS7I | visual | 491 | 0.611 | 0 | 1 | 0 |
| 1M3Vdl6DRkU | duration | 1388 | 0.586 | 0 | 1 | 0 |
| aqBHXNGKvKU | contradiction | 52 | 0.596 | 0 | 1 | 0 |
| dzOTaNwiFmA | contradiction | 52 | 0.385 | 0 | 1 | 0 |
| AxIOGqHQqZM | contradiction | 80 | 0.6 | 0 | 3 | 0 |
| 2su8e-nhMGw | contradiction | 25 | 0.36 | 0 | 1 | 0 |
| n_Smy5-1cHE | contradiction | 97 | 0.402 | 0 | 1 | 0 |
| SdnEbJZoNg8 | contradiction | 30 | 0.4 | 0 | 1 | 0 |
| 5KVDDfAkRgc | timeline | 320 | 0.669 | 0 | 1 | 0 |
| YeRS4TbtZWA | timeline | 195 | 0.626 | 0 | 3 | 0 |
