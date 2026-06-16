export const meta = {
  name: 'memovox-stress-panel',
  description: 'Adversarial multi-agent review of the 41-video stress run + the iter11 fixes',
  phases: [
    { title: 'Review', detail: '3 skeptics: data-quality, retrieval/relevance, consolidation/synthesis' },
    { title: 'Adjudicate', detail: 'chair decides team_satisfied + blocking_fixable' },
  ],
}

const STORE = '/tmp/mv_stress_iter12'
const REPORT = 'stress/reports/iter12.json'
const PRIOR = 'stress/reports/iter10.json'

const COMMON = `
You are an ADVERSARIAL reviewer of the memovox pipeline (a free/lexical video-knowledge engine).
The goal is to find REAL defects, confirm or REFUTE claimed fixes, and honestly classify each
finding. Be skeptical and concrete — verify with evidence, do not take prior claims on faith.

Context you must read first:
- The post-fix stress report: ${REPORT} (41 videos, free/captions path, nli=lexical, embed=hashing).
- The pre-fix report for comparison: ${PRIOR}.
- The uncommitted/recent fixes under review: run \`git -C /Users/connor/projects/memovox diff HEAD~0 --stat\`
  and \`git -C /Users/connor/projects/memovox diff\` plus inspect:
    src/memovox/stentor/transcript.py  (speaker-label heuristic + HTML-entity decode)
    src/memovox/loom/consolidate.py    (topic filter BEFORE the max_claims cap)
    tests/test_caption_cleaning_and_consolidate.py (the new regression tests)

The three iter11 fixes claimed:
  (1) Speaker-label false positives: sentence fragments ("But caveat", "For example",
      "And that worked great", "I have three world records") were being parsed as speaker IDs;
      now gated by _looks_like_speaker (ALLCAPS or Title-case proper names, <=3 words).
  (2) HTML entities (&nbsp;, &amp;, &#39;) survived into committed claim text; now html.unescape'd.
  (3) find_contradictions(topic=...) truncated to the first 600 committed claims BEFORE applying
      the topic filter, so topics whose claims arrive late in ingest order (e.g. saturated fat,
      ~#14-15 of 41) returned ZERO candidates — misattributed to the "lexical-NLI limit". Now the
      topic filter runs before the cap.

A consolidated store exists at ${STORE}. To run live queries WITHOUT lock contention, COPY it first:
  cp -r ${STORE} /tmp/panel_<yourlabel> && then in python:
    import sys; sys.path.insert(0, 'src')
    from memovox import Memovox
    mv = Memovox(store='/tmp/panel_<yourlabel>', embed_backend='hashing', nli_backend='lexical',
                 llm_backend='none', rerank_backend='identity', entity_backend='none')
    print(mv.ask('...'))            # read-only; cited Answer
    print([p.to_dict() for p in mv.contradictions(topic='saturated fat')])
    print(mv.synthesize('saturated fat').to_dict())
Work from /Users/connor/projects/memovox. Run the venv python: \`source .venv/bin/activate\`.

Classify EVERY finding as exactly one of:
  FIXABLE   — a real defect in memovox code that can be fixed on the free/lexical path now.
  FUNDAMENTAL — a genuine limit of the free path (hashing embedder / lexical NLI / no diarization),
              only resolvable by an optional backend ([embed]/[nli]/[pyannote]/LLM). Justify WHY.
  ENV       — a local-environment artifact (e.g. the broken-pyarrow dlopen), not a memovox bug.
For a claimed fix, also state verdict: CONFIRMED (works + correct) or REFUTED (with the counterexample).
`

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['fix_verdicts', 'findings', 'summary'],
  properties: {
    fix_verdicts: {
      type: 'array',
      description: 'Verdict on each claimed fix you were asked to verify.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['fix', 'verdict', 'evidence'],
        properties: {
          fix: { type: 'string' },
          verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'PARTIAL', 'NOT_APPLICABLE'] },
          evidence: { type: 'string', description: 'Concrete evidence: command output, report values, counterexample.' },
        },
      },
    },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['code', 'severity', 'classification', 'title', 'evidence', 'fix_suggestion'],
        properties: {
          code: { type: 'string', description: 'short_snake_case id' },
          severity: { type: 'string', enum: ['CRIT', 'HIGH', 'MED', 'LOW'] },
          classification: { type: 'string', enum: ['FIXABLE', 'FUNDAMENTAL', 'ENV'] },
          title: { type: 'string' },
          evidence: { type: 'string', description: 'How you verified it; concrete data.' },
          fix_suggestion: { type: 'string', description: 'If FIXABLE, the concrete change. Else "n/a".' },
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

YOUR LENS: caption cleaning & claim data-quality. Adversarially verify fixes (1) and (2), and hunt
for OTHER residual data-quality defects across all 41 videos. Specifically:
  - Re-inspect every video's distinct_speakers in ${REPORT}: are there STILL sentence-fragment or
    duplicate-cased speaker IDs? (pre-fix examples: "But caveat", "For example",
    "And that worked great", "I have three world records", "Yanjaa"/"YANJAA".)
  - Sample real committed claim text from the store for leftover artifacts: &nbsp;, &amp;, &#39;,
    stray <tags>, '>>' turn markers, musical-note lyrics committed as claims, ultra-short fragments.
  - Try to BREAK _looks_like_speaker with plausible real caption patterns (false pos AND false neg):
    e.g. "Dr. Smith:", "MR. T:", "Anderson Cooper:", "well, here's the thing:", "JAY-Z:".
  - Note the Yanjaa/YANJAA case-duplicate speaker: is it still split? is that FIXABLE (case-fold) and worth it?
Return findings with concrete evidence.`,
  },
  {
    key: 'retrieval-relevance',
    prompt: `${COMMON}

YOUR LENS: retrieval, the out-of-corpus relevance gate, and CITATION PRECISION. Specifically:
  - From ${REPORT} asks/refusals: any fabrication (out-of-corpus answered without low_evidence) or
    over-refusal (an in-corpus probe wrongly refused)? Verify against the actual store.
  - Citation precision: for short queries the dense (hashing-embed) RRF leg injects zero-lexical-overlap
    moments into top-k (e.g. iter10 cited the Lunchables/Attenborough/Trevor-Noah videos for
    "what is AGI?", "Graham's number", "Steve Jobs"). Quantify on iter12: for each ask, how many of the
    8 citations share NO distinctive query token with the query? Is this a FIXABLE retrieval defect or a
    FUNDAMENTAL hashing-embed limit? NOTE: a retrieval-ranking change would break the frozen
    eval/golden/parity.json gate (see eval/harness.py ~line 776) — weigh that.
  - Invent 6 NEW adversarial out-of-corpus questions whose generic tokens scatter across the now-richer
    41-genre corpus (watches/cars/chess/law/food/travel) and check they are REFUSED. Report any leak.
Return findings with concrete evidence (run the queries).`,
  },
  {
    key: 'consolidation-synthesis',
    prompt: `${COMMON}

YOUR LENS: consolidation, contradictions, synthesis, scaling. Specifically:
  - Verify fix (3): run mv.contradictions(topic='saturated fat') and (topic='breakfast') on the store.
    Do the planted cross-video pairs [aqBHXNGKvKU vs dzOTaNwiFmA] / [AxIOGqHQqZM vs 2su8e-nhMGw] now
    surface? If NOT, prove WHY by printing the candidate claim texts and the LexicalNLI verdict on the
    actual mirror sentences — distinguish "cap artifact" (now fixed) from "genuine lexical-NLI semantic
    limit" (FUNDAMENTAL). Do not just trust the count.
  - synthesize('saturated fat') / ('diet') return low_evidence=true WITH 29-31 citations. Is reporting
    low_evidence while emitting 30 citations misleading to a user? FIXABLE or FUNDAMENTAL?
  - The GLOBAL consolidate pass still caps at max_claims=600 of ~10,182 claims (consolidate.py line ~71,
    no-topic path) — 94% of claims never scanned for cross-video contradictions/consensus at this scale.
    Is that a real correctness-at-scale defect or acceptable cost control? Propose the concrete fix if FIXABLE.
Return findings with concrete evidence (run the queries; print intermediate claim texts).`,
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
      description: 'Finding codes that MUST be fixed before the team is satisfied. A finding blocks only if classification=FIXABLE, severity>=MED, and it is a genuine correctness/UX defect (NOT a known free-path limit, NOT ENV).',
      items: {
        type: 'object', additionalProperties: false,
        required: ['code', 'severity', 'why_blocking', 'fix'],
        properties: {
          code: { type: 'string' }, severity: { type: 'string' },
          why_blocking: { type: 'string' }, fix: { type: 'string' },
        },
      },
    },
    accepted_limits: { type: 'array', items: { type: 'string' }, description: 'Findings accepted as FUNDAMENTAL/ENV with one-line justification.' },
    rationale: { type: 'string' },
  },
}

const chair = await agent(`${COMMON}

YOU ARE THE CHAIR. Three adversarial reviewers (data-quality, retrieval-relevance,
consolidation-synthesis) returned the JSON findings below. Adjudicate.

REVIEWS:
${JSON.stringify(reviews.map((r, i) => ({ lens: lenses[i].key, ...r })), null, 2)}

Decide team_satisfied. Rules:
 - The team is SATISFIED only when there is no FIXABLE finding of severity >= MED that is a genuine
   correctness or user-facing-quality defect. FUNDAMENTAL (free-path) and ENV findings never block.
 - Be adversarial about the reviewers too: if a reviewer mislabeled a FUNDAMENTAL limit as FIXABLE
   (or vice-versa), correct it in your rationale. If a claimed fix was REFUTED, that blocks.
 - blocking_fixable must be ACTIONABLE: each needs a concrete fix. Keep it minimal and real.
Return the verdict.`, { label: 'chair', phase: 'Adjudicate', schema: CHAIR_SCHEMA })

return { reviews: reviews.map((r, i) => ({ lens: lenses[i].key, ...r })), chair }
