# Stress report — iter0-baseline (nli=lexical)

- ingest: **21/21** ok
- moments=4443 committed_claims=6355 trivial=3584 (mean ratio **0.568**)
- provenance violations: **0**
- ask hit-rate: **1.0** | refusals_ok: 0/3

## Findings
- **[HIGH] trivial_claims** — mean trivial-claim ratio 0.568 (>0.25)
- **[HIGH] fabrication** — 3 out-of-corpus questions answered without low_evidence
- **[MED] contradictions_missed** — 2/2 planted cross-video contradictions not surfaced
- **[MED] synthesis_synonyms** — synthesize('AGI') found 0 consensus points (token-Jaccard misses synonyms?)
- **[LOW] speaker_collapse** — multi-speaker debate collapsed to a single speaker (no diarization backend)

## Per-video
| id | group | claims | trivial% | prov! | speakers | caps |
|---|---|---|---|---|---|---|
| UF8uR6Z6KLc | sweet | 87 | 0.23 | 0 | 1 | 0 |
| zjkBMFhNj_g | sweet | 393 | 0.305 | 0 | 1 | 0 |
| XTeJ64KD5cg | sweet | 131 | 0.756 | 0 | 4 | 0 |
| t3H5D-XxPrI | asr | 33 | 0.939 | 0 | 1 | 0 |
| 144uOfr4SYA | asr | 657 | 0.412 | 0 | 1 | 0 |
| nkG2SSzPUns | asr | 198 | 0.924 | 0 | 1 | 0 |
| AmlF6xq2SaQ | asr | 14 | 0.643 | 0 | 1 | 0 |
| J7DzL2_Na80 | visual | 401 | 0.758 | 0 | 1 | 0 |
| WUvTyaaNkzM | visual | 155 | 0.619 | 0 | 1 | 0 |
| jbkSRLYSojo | visual | 26 | 0.577 | 0 | 1 | 0 |
| kCc8FmEb1nY | visual | 745 | 0.357 | 0 | 1 | 0 |
| jGwO_UgTS7I | visual | 621 | 0.626 | 0 | 1 | 0 |
| 1M3Vdl6DRkU | duration | 1819 | 0.61 | 0 | 1 | 0 |
| aqBHXNGKvKU | contradiction | 75 | 0.627 | 0 | 1 | 0 |
| dzOTaNwiFmA | contradiction | 52 | 0.385 | 0 | 1 | 0 |
| AxIOGqHQqZM | contradiction | 108 | 0.63 | 0 | 3 | 0 |
| 2su8e-nhMGw | contradiction | 25 | 0.36 | 0 | 1 | 0 |
| n_Smy5-1cHE | contradiction | 98 | 0.408 | 0 | 1 | 0 |
| SdnEbJZoNg8 | contradiction | 30 | 0.4 | 0 | 1 | 0 |
| 5KVDDfAkRgc | timeline | 438 | 0.71 | 0 | 1 | 0 |
| YeRS4TbtZWA | timeline | 249 | 0.659 | 0 | 3 | 0 |
