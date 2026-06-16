# Stress report — iter15-regressionfix (nli=lexical)

- ingest: **41/41** ok
- moments=7783 committed_claims=10222 trivial=1 (mean ratio **0.0**)
- provenance violations: **0**
- ask hit-rate: **1.0** | refusals_ok: 13/13

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
| nkG2SSzPUns | asr | 173 | 0.0 | 0 | 1 | 0 |
| AmlF6xq2SaQ | asr | 14 | 0.0 | 0 | 1 | 0 |
| J7DzL2_Na80 | visual | 326 | 0.0 | 0 | 1 | 0 |
| WUvTyaaNkzM | visual | 114 | 0.0 | 0 | 1 | 0 |
| jbkSRLYSojo | visual | 26 | 0.0 | 0 | 1 | 0 |
| kCc8FmEb1nY | visual | 742 | 0.0 | 0 | 1 | 0 |
| jGwO_UgTS7I | visual | 490 | 0.0 | 0 | 1 | 0 |
| 1M3Vdl6DRkU | duration | 1388 | 0.0 | 0 | 1 | 0 |
| aqBHXNGKvKU | contradiction | 52 | 0.0 | 0 | 1 | 0 |
| dzOTaNwiFmA | contradiction | 52 | 0.0 | 0 | 1 | 0 |
| AxIOGqHQqZM | contradiction | 82 | 0.0 | 0 | 1 | 0 |
| 2su8e-nhMGw | contradiction | 25 | 0.0 | 0 | 1 | 0 |
| n_Smy5-1cHE | contradiction | 97 | 0.0 | 0 | 1 | 0 |
| SdnEbJZoNg8 | contradiction | 30 | 0.0 | 0 | 1 | 0 |
| 5KVDDfAkRgc | timeline | 320 | 0.0 | 0 | 1 | 0 |
| YeRS4TbtZWA | timeline | 194 | 0.0 | 0 | 2 | 0 |
| 84WIaK3bl_s | vlog | 51 | 0.0 | 0 | 1 | 0 |
| Gnr2k-VvsCc | vlog | 100 | 0.0 | 0 | 1 | 0 |
| jlPQjC-CjNg | watches | 129 | 0.0 | 0 | 1 | 0 |
| sBN_2g0_NE8 | watches | 79 | 0.0 | 0 | 1 | 0 |
| F3OEtmUv5Nc | watches | 75 | 0.0 | 0 | 1 | 0 |
| d5yvnak4x8U | review | 229 | 0.0 | 0 | 1 | 0 |
| n_XlcTBjVgY | review | 139 | 0.0 | 0 | 1 | 0 |
| q0aFOxT6TNw | review | 117 | 0.0 | 0 | 1 | 0 |
| ugpcWk0p4Mk | travel | 155 | 0.0 | 0 | 1 | 0 |
| teN-Y1wAu78 | travel | 165 | 0.0 | 0 | 1 | 0 |
| WRJ1oJ0rfFU | travel | 127 | 0.0 | 0 | 1 | 0 |
| MY3Qy6vAbZQ | food | 259 | 0.0 | 0 | 1 | 0 |
| tj6rNP2p1Yk | food | 516 | 0.0 | 0 | 1 | 0 |
| aeWyp2vXxqA | science | 58 | 0.0 | 0 | 1 | 0 |
| d95dOH-7GHM | science | 164 | 0.006 | 0 | 10 | 0 |
| T4uMfr4dppQ | finance | 316 | 0.0 | 0 | 1 | 0 |
| spr5smxuO5E | law | 173 | 0.0 | 0 | 1 | 0 |
| 8zDwI0Z-VIg | chess | 148 | 0.0 | 0 | 1 | 0 |
| 7ZhdXgRfxHI | nature | 661 | 0.0 | 0 | 1 | 0 |
| VyEINfRMvdc | comedy | 1156 | 0.0 | 0 | 1 | 0 |
