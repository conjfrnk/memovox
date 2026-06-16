export const meta = {
  name: 'memovox-stress-panel-r2',
  description: 'Adversarial round-2 review: verify the 6 panel fixes + hunt for regressions (41-video corpus)',
  phases: [
    { title: 'Review', detail: '3 skeptics: data-quality, retrieval/relevance, consolidation/synthesis' },
    { title: 'Adjudicate', detail: 'chair decides team_satisfied + blocking_fixable' },
  ],
}

const STORE = '/tmp/mv_stress_iter15'
const REPORT = 'stress/reports/iter15.json'
const PRIOR = 'stress/reports/iter14.json'

const COMMON = `
You are an ADVERSARIAL reviewer of the memovox pipeline (a free/lexical video-knowledge engine).
This is ROUND 4. Prior rounds CONFIRMED H1, H1b, H2, M2, M3 and the two round-2 regression fixes,
and fixed the round-3 HIGH (SYNTH_OOC_STRUCTURED_BYPASS): synthesize()'s out-of-corpus relevance
gate was hoisted ABOVE the consensus/contradiction composition (commit e0b4c30), so OOC topics
refuse even when generic tokens build structure. NOTE: the store ${STORE} was ingested at 7374841;
the round-3 fix is QUERY-TIME ONLY (synthesize.py), so the store's ingest is identical to HEAD and
your LIVE synthesize() queries exercise the fixed code. Your job: (a) confirm the round-3 OOC fix
holds on the structured path (the 4 probes that leaked must now refuse) without breaking in-corpus
synthesis, (b) re-verify nothing else regressed, (c) hunt for any remaining >= MED defect. Be
skeptical; verify with evidence. The bar is high but FAIR — do NOT invent blockers out of
FUNDAMENTAL free-path limits (hashing embed / lexical NLI / no diarization) or pre-existing LOW issues.

Context to read first:
- The post-fix report: ${REPORT} (41 videos, free/captions, nli=lexical, embed=hashing).
- The previous round's report: ${PRIOR}.
- The fixes under review: \`git -C /Users/connor/projects/memovox diff 3c6fbc3 e0b4c30\` (all
  rounds). The newest (round-3 OOC-structured-path) fix: \`git -C ... show e0b4c30\`. Key files:
    src/memovox/loom/consolidate.py   (_candidate_pairs bucket-blocking; raised max_claims=50000;
                                        scope always in universe; topic filter before cap)
    src/memovox/loom/consensus.py     (partition_claims uses the same bucket-blocking)
    src/memovox/augur/answer.py       (advice/transaction verbs added to _COMMON_WORDS/_COVERAGE_FILLER)
    src/memovox/augur/synthesize.py   (salient extractive fallback when no structure)
    src/memovox/stentor/transcript.py (residual-bracket strip; speaker casefold; discourse openers)

The 6 fixes claimed (all must be CONFIRMED, and none may have caused a regression):
  H1 consolidation no longer 94% blind: the offline pass + consensus now scan the WHOLE corpus via
     inverted-index blocking with a per-bucket cap. VERIFY all claims are scanned (report
     aggregate.consolidate.claims_scanned ~= total claims, capped=False) and consolidate wall-time is
     affordable.
  H1b PRECISION GATE (e25193d): the full scan first emitted 534 garbage CONTRADICTS + 2256 SUPPORTS
     (unrelated short claims sharing 2 generic tokens + a negation/agreement cue). A near-mirror gate
     (>=3 shared content tokens AND Jaccard >=0.5; consensus min_shared 2->3) was added. VERIFY the
     resulting edge graph is now SMALL and CLEAN: open the db, count CONTRADICTS/SUPPORTS, sample ALL
     CONTRADICTS and many SUPPORTS, and judge each as real vs false-positive. CRITICALLY also check the
     gate did NOT over-suppress: golden-style near-mirror contradictions ("X is harmful"/"X is not
     harmful", >=3 shared, J>=0.5) must still be found — confirm via the new regression tests and by
     reasoning about the diff. Report if edges are still garbage (precision too low) OR if real
     contradictions are now missed (recall regression).
  H2 incremental new-vs-ALL holds past the cap (scope always in the universe).
  H3 relevance gate: generic advice verbs no longer leak OOC queries. VERIFY the home-purchase leak now
     refuses, AND that no LEGITIMATE in-corpus query is now over-refused by the added verbs (e.g. a watch
     or car review query that legitimately uses "recommend"/"buy").
  M1 synthesize emits a salient summary (low_evidence=False) when citations exist but no structure;
     VERIFY the summary sentences are all [n]-cited and grounded (not confabulated), and the genuine
     zero-citation case still reports low-evidence.
  M2 bracket annotations stripped from claims; VERIFY none remain AND that code/math ([i],[b,t]) and
     legitimate content were NOT over-stripped (sample claims from CS/math videos: Karpathy, Strang, 3b1b).
  M3 case-duplicate speaker collapsed (YANJAA/Yanjaa -> one); VERIFY no remaining case-dup speakers.

Consolidated store at ${STORE}. To run live queries WITHOUT lock contention, COPY first:
  cp -r ${STORE} /tmp/panelr2_<label> && in python (cd /Users/connor/projects/memovox; source .venv/bin/activate):
    import sys; sys.path.insert(0,'src'); from memovox import Memovox
    mv = Memovox(store='/tmp/panelr2_<label>', embed_backend='hashing', nli_backend='lexical',
                 llm_backend='none', rerank_backend='identity', entity_backend='none')
    print(mv.ask('...')); print(mv.synthesize('saturated fat').to_dict())
  For the persistent edge graph, open the sqlite db directly (read-only): /tmp/panelr2_<label>/memovox.db,
  table 'edges' (columns src,rel,dst,video_id,confidence). Resolve claim text via the 'claims' table.

Classify EVERY finding as FIXABLE / FUNDAMENTAL (free-path limit; justify) / ENV. For a claimed fix give
verdict CONFIRMED or REFUTED (with the counterexample). Flag any REGRESSION explicitly (severity >= MED).
Work from /Users/connor/projects/memovox.
`

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['fix_verdicts', 'findings', 'summary'],
  properties: {
    fix_verdicts: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['fix', 'verdict', 'evidence'],
        properties: {
          fix: { type: 'string' },
          verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'PARTIAL', 'NOT_APPLICABLE'] },
          evidence: { type: 'string' },
        },
      },
    },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['code', 'severity', 'classification', 'is_regression', 'title', 'evidence', 'fix_suggestion'],
        properties: {
          code: { type: 'string' },
          severity: { type: 'string', enum: ['CRIT', 'HIGH', 'MED', 'LOW'] },
          classification: { type: 'string', enum: ['FIXABLE', 'FUNDAMENTAL', 'ENV'] },
          is_regression: { type: 'boolean', description: 'true if introduced by the 078d56a fixes' },
          title: { type: 'string' },
          evidence: { type: 'string' },
          fix_suggestion: { type: 'string' },
        },
      },
    },
    summary: { type: 'string' },
  },
}

phase('Review')
const lenses = [
  {
    key: 'data-quality',
    prompt: `${COMMON}

YOUR LENS: caption cleaning & claim data-quality. Verify M2 (bracket strip) and M3 (speaker casefold).
  - Scan ALL committed claim text in the store db for leftover [bracket] spans, &-entities, raw U+00A0,
    stray tags, '>>' markers. Report any survivors.
  - REGRESSION CHECK for M2: sample claims from code/math videos (Karpathy kCc8FmEb1nY/zjkBMFhNj_g,
    Strang J7DzL2_Na80, 3Blue1Brown WUvTyaaNkzM) — was any legitimate bracketed token ([i], [b,t], [0])
    or real content wrongly stripped? Count claims that lost content.
  - Re-scan distinct_speakers in ${REPORT}: any remaining sentence-fragment OR case-duplicate speaker?
  - Try to break the new _strip_bracket / speaker casefold with plausible caption patterns.
Return findings with concrete evidence.`,
  },
  {
    key: 'retrieval-relevance',
    prompt: `${COMMON}

YOUR LENS: retrieval, relevance gate, synthesize. Re-verify the round-2 regression fixes (7374841).
  - H3 + its round-2 fix: 'what is the best way to recommend a first home purchase?' and 'how do I
    recommend a good real estate agent?' must REFUSE (low_evidence=True). AND the over-refusal fix must
    hold: 'which Rolex should I buy?' must ANSWER (low_evidence=False, citations) — rolex is a real
    48-claim topic. Try other legit in-corpus 'buy/recommend a <topic>' queries (iphone, watch — note
    'watch' the noun collides with 'watch' the verb, a PRE-EXISTING FUNDAMENTAL limit, not a regression).
  - M1 + round-3 hoisted gate (THE KEY ONE): mv.synthesize('saturated fat')/('diet')/('AGI') must
    return low_evidence=False with a cited, grounded summary. The 4 structured-path OOC probes that
    leaked in round 3 — synthesize('what is the population of Brazil?'), ('what is the chemical formula
    for table salt?'), ('how do volcanoes form?'), ('how do I knit a wool sweater?') — must NOW return
    low_evidence=True with 0 citations. Also confirm 'capital of Mongolia' / 'underwater basket weaving'
    still refuse. Try 3 MORE of your own generic-token OOC synthesize probes to confirm no residual leak.
  - Re-run 5 fresh out-of-corpus probes on BOTH ask() and synthesize() to confirm no fabrication leak.
Return findings with concrete evidence (run the queries).`,
  },
  {
    key: 'consolidation-synthesis',
    prompt: `${COMMON}

YOUR LENS: consolidation correctness, scale, and edge QUALITY. Verify H1, H1b, H2.
  - VERIFY H1: from ${REPORT} aggregate.consolidate — claims_scanned ~= total committed, capped=False,
    and the contradictions/consensus stage wall-times (should be a few seconds, topics dominates).
  - VERIFY H1b (THE KEY ONE): open the store db; count CONTRADICTS and SUPPORTS edges. They should now be
    FEW (was 534/2256). Print EVERY CONTRADICTS pair's two claim texts and classify real vs false-positive.
    Sample many SUPPORTS likewise. Decide: is the graph now clean? Is the precision gate too lenient
    (garbage remains -> FIXABLE) or too strict (you can argue a real near-mirror contradiction in this
    corpus is now missed -> recall regression)? Note: the planted diet pairs are differently-phrased and
    remain a genuine [nli]-only FUNDAMENTAL limit (NOT a precision-gate bug).
  - VERIFY H2 conceptually from the diff + the new regression tests (new scope claim past the cap is
    paired against prior).
Return findings with concrete evidence (open the db; print pair texts).`,
  },
]

const reviews = await parallel(lenses.map(l => () =>
  agent(l.prompt, { label: `review:${l.key}`, phase: 'Review', schema: FINDINGS_SCHEMA })
))

phase('Adjudicate')
const CHAIR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['team_satisfied', 'blocking_fixable', 'accepted_limits', 'rationale'],
  properties: {
    team_satisfied: { type: 'boolean' },
    blocking_fixable: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['code', 'severity', 'why_blocking', 'fix'],
        properties: {
          code: { type: 'string' }, severity: { type: 'string' },
          why_blocking: { type: 'string' }, fix: { type: 'string' },
        },
      },
    },
    accepted_limits: { type: 'array', items: { type: 'string' } },
    rationale: { type: 'string' },
  },
}

const chair = await agent(`${COMMON}

YOU ARE THE CHAIR. Three adversarial reviewers returned the JSON findings below. Adjudicate.

REVIEWS:
${JSON.stringify(reviews.map((r, i) => ({ lens: lenses[i].key, ...r })), null, 2)}

Decide team_satisfied. Rules:
 - SATISFIED only when (a) all prior fixes still hold (H1, H1b, H2, M2, M3, the two round-2 regression
   fixes) AND the round-3 OOC fix holds (synthesize refuses OOC topics on BOTH the structured and the
   fallback paths), AND (b) no FIXABLE finding of severity >= MED is a genuine correctness/UX defect or
   a NEW regression (garbage edges, real contradictions newly missed, over-stripped content, over-
   refusal, ungrounded or confabulated synthesis). FUNDAMENTAL free-path limits (hashing embed / lexical
   NLI / no diarization / vocabulary gaps) and pre-existing LOW issues NEVER block — name them as
   accepted_limits. Do not manufacture a blocker to avoid declaring satisfaction; if the pipeline is
   genuinely sound on the free path, SAY SO.
 - A REGRESSION (e.g. over-stripped content, new over-refusal, garbage contradiction edges that mislead
   users) of severity >= MED blocks even if technically "fixable later".
 - FUNDAMENTAL (free-path) and ENV findings never block. Correct any reviewer misclassification.
 - blocking_fixable must be concrete + actionable. Keep it minimal and real.
Return the verdict.`, { label: 'chair', phase: 'Adjudicate', schema: CHAIR_SCHEMA })

return { reviews: reviews.map((r, i) => ({ lens: lenses[i].key, ...r })), chair }
