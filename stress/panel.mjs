export const meta = {
  name: 'memovox-stress-panel-r2',
  description: 'Adversarial round-2 review: verify the 6 panel fixes + hunt for regressions (41-video corpus)',
  phases: [
    { title: 'Review', detail: '3 skeptics: data-quality, retrieval/relevance, consolidation/synthesis' },
    { title: 'Adjudicate', detail: 'chair decides team_satisfied + blocking_fixable' },
  ],
}

const STORE = '/tmp/mv_stress_iter14'
const REPORT = 'stress/reports/iter14.json'
const PRIOR = 'stress/reports/iter13.json'

const COMMON = `
You are an ADVERSARIAL reviewer of the memovox pipeline (a free/lexical video-knowledge engine).
This is ROUND 2. A prior panel found 6 FIXABLE defects; they were fixed (commit 078d56a), and a
7th fix (commit e25193d) was added because fixing H1 EXPOSED a latent lexical-NLI precision problem.
Your job: CONFIRM or REFUTE each fix on the fresh 41-video run, AND — critically — hunt for any
REGRESSION the fixes introduced, plus any remaining/new defect. Be skeptical; verify with evidence.

Context to read first:
- The post-fix report: ${REPORT} (41 videos, free/captions, nli=lexical, embed=hashing).
- The previous round's report: ${PRIOR}.
- The fixes under review: \`git -C /Users/connor/projects/memovox diff 3c6fbc3 e25193d\` (both
  the 6 fixes AND the precision gate). Key files:
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

YOUR LENS: retrieval, relevance gate, synthesize. Verify H3 (advice-verb leak) and M1 (synthesize fallback).
  - VERIFY H3: mv.ask('what is the best way to recommend a first home purchase?') now refuses
    (low_evidence=True). Then REGRESSION-test the added verbs against legitimate in-corpus queries that
    use them, e.g. 'what watch does Teddy recommend for a first luxury watch?',
    'which airline does Jeb Brooks recommend?', 'should I buy the iPhone 17 Pro?' — these SHOULD still be
    answered (the topic word — watch/airline/iphone — carries topicality). Report any new over-refusal.
  - VERIFY M1: mv.synthesize('saturated fat') and ('diet') now return low_evidence=False with a non-empty
    summary; CHECK every summary sentence carries an [n] marker and the cited claim actually contains that
    text (grounded, not confabulated). Confirm a junk topic ('quantum chromodynamics') still low-evidence.
  - Re-run 5 fresh out-of-corpus probes to confirm no new fabrication leak.
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
 - SATISFIED only when (a) all 7 fixes (the 6 round-1 fixes + the H1b precision gate) are CONFIRMED
   (no REFUTED), AND (b) no FIXABLE finding of severity >= MED is a genuine correctness/UX defect or a
   REGRESSION introduced by the fixes (garbage edges remaining, real contradictions newly missed,
   over-stripped content, new over-refusal, ungrounded synthesis text).
 - A REGRESSION (e.g. over-stripped content, new over-refusal, garbage contradiction edges that mislead
   users) of severity >= MED blocks even if technically "fixable later".
 - FUNDAMENTAL (free-path) and ENV findings never block. Correct any reviewer misclassification.
 - blocking_fixable must be concrete + actionable. Keep it minimal and real.
Return the verdict.`, { label: 'chair', phase: 'Adjudicate', schema: CHAIR_SCHEMA })

return { reviews: reviews.map((r, i) => ({ lens: lenses[i].key, ...r })), chair }
