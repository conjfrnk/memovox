# Stress report — iterE-upgraded (nli=deberta-nli)

- ingest: **41/41** ok
- moments=4317 committed_claims=8065 trivial=2 (mean ratio **0.0**)
- provenance violations: **0**
- ask hit-rate: **1.0** | refusals_ok: 13/13

## Findings
- **[MED] synthesis_synonyms** — synthesize('AGI') found 0 consensus points (token-Jaccard misses synonyms?)
- **[LOW] contradictions_missed** — 2/2 planted pairs not surfaced (lexical-NLI limit + corpus caveat)
- **[LOW] speaker_collapse** — multi-speaker debate collapsed to a single speaker (no diarization backend)

## Per-video
| id | group | claims | trivial% | prov! | speakers | caps |
|---|---|---|---|---|---|---|
| UF8uR6Z6KLc | sweet | 50 | 0.0 | 0 | 1 | 0 |
| zjkBMFhNj_g | sweet | 227 | 0.0 | 0 | 1 | 0 |
| XTeJ64KD5cg | sweet | 108 | 0.0 | 0 | 4 | 0 |
| t3H5D-XxPrI | asr | 27 | 0.0 | 0 | 1 | 0 |
| 144uOfr4SYA | asr | 396 | 0.0 | 0 | 1 | 0 |
| nkG2SSzPUns | asr | 182 | 0.0 | 0 | 1 | 0 |
| AmlF6xq2SaQ | asr | 9 | 0.0 | 0 | 1 | 0 |
| J7DzL2_Na80 | visual | 265 | 0.0 | 0 | 1 | 0 |
| WUvTyaaNkzM | visual | 90 | 0.0 | 0 | 1 | 0 |
| jbkSRLYSojo | visual | 14 | 0.0 | 0 | 1 | 0 |
| kCc8FmEb1nY | visual | 387 | 0.0 | 0 | 1 | 0 |
| jGwO_UgTS7I | visual | 371 | 0.0 | 0 | 1 | 0 |
| 1M3Vdl6DRkU | duration | 1178 | 0.0 | 0 | 1 | 0 |
| aqBHXNGKvKU | contradiction | 42 | 0.0 | 0 | 1 | 0 |
| dzOTaNwiFmA | contradiction | 39 | 0.0 | 0 | 1 | 0 |
| AxIOGqHQqZM | contradiction | 68 | 0.0 | 0 | 1 | 0 |
| 2su8e-nhMGw | contradiction | 11 | 0.0 | 0 | 1 | 0 |
| n_Smy5-1cHE | contradiction | 67 | 0.0 | 0 | 1 | 0 |
| SdnEbJZoNg8 | contradiction | 23 | 0.0 | 0 | 1 | 0 |
| 5KVDDfAkRgc | timeline | 294 | 0.0 | 0 | 1 | 0 |
| YeRS4TbtZWA | timeline | 116 | 0.0 | 0 | 2 | 0 |
| 84WIaK3bl_s | vlog | 45 | 0.0 | 0 | 1 | 0 |
| Gnr2k-VvsCc | vlog | 61 | 0.0 | 0 | 1 | 0 |
| jlPQjC-CjNg | watches | 95 | 0.0 | 0 | 1 | 0 |
| sBN_2g0_NE8 | watches | 66 | 0.0 | 0 | 1 | 0 |
| F3OEtmUv5Nc | watches | 65 | 0.0 | 0 | 1 | 0 |
| d5yvnak4x8U | review | 234 | 0.0 | 0 | 1 | 0 |
| n_XlcTBjVgY | review | 76 | 0.0 | 0 | 1 | 0 |
| q0aFOxT6TNw | review | 111 | 0.0 | 0 | 1 | 0 |
| ugpcWk0p4Mk | travel | 142 | 0.0 | 0 | 1 | 0 |
| teN-Y1wAu78 | travel | 94 | 0.0 | 0 | 1 | 0 |
| WRJ1oJ0rfFU | travel | 80 | 0.0 | 0 | 1 | 0 |
| MY3Qy6vAbZQ | food | 256 | 0.0 | 0 | 1 | 0 |
| tj6rNP2p1Yk | food | 533 | 0.0 | 0 | 1 | 0 |
| aeWyp2vXxqA | science | 39 | 0.0 | 0 | 1 | 0 |
| d95dOH-7GHM | science | 130 | 0.008 | 0 | 9 | 0 |
| T4uMfr4dppQ | finance | 178 | 0.0 | 0 | 1 | 0 |
| spr5smxuO5E | law | 117 | 0.0 | 0 | 1 | 0 |
| 8zDwI0Z-VIg | chess | 102 | 0.0 | 0 | 1 | 0 |
| 7ZhdXgRfxHI | nature | 483 | 0.0 | 0 | 1 | 0 |
| VyEINfRMvdc | comedy | 1194 | 0.001 | 0 | 1 | 0 |
