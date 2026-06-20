export const meta = {
  name: 'memovox-panel-upgraded',
  description: 'Adversarial review of the clean UPGRADED path (dense retrieval + cross-encoder + DeBERTa near-mirror)',
  phases: [
    { title: 'Review', detail: '3 skeptics: retrieval/relevance, contradiction/consensus, data-quality/regressions' },
    { title: 'Adjudicate', detail: 'chair: team_satisfied — fixable defects block; rigorously-PROVEN impossibilities do not' },
  ],
}

const REPORT = 'stress/reports/iterH.json'
const PROBES = 'stress/reports/iterM_probes.json'
const STORE = '/tmp/mv_embed4'
const VENV = '/Users/connor/projects/memovox/.venv/bin/python'

const COMMON = `
You are an ADVERSARIAL reviewer of memovox's UPGRADED path: BGE-M3 dense retrieval +
cross-encoder rerank + DeBERTa NLI (judging near-mirror candidates) + lexical relevance gate.
The optional backends are installed and used. Read:
  - ${REPORT} — the iterG upgraded stress run (41-video corpus).
  - ${PROBES} — a pre-generated probe battery (ask in-corpus / OOC / hard edge cases,
    synthesize in/OOC, contradictions) run with the REAL backends, so you can judge actual
    behavior WITHOUT loading models. (You MAY run extra probes via the venv if needed:
    \`MEMOVOX_MODELS_DIR=/tmp/mv_models_cache PYTHONPATH=src ${VENV}\` + cp ${STORE} to a temp dir
    and open Memovox(embed_backend='sentence-transformers', nli_backend='deberta-nli',
    rerank_backend='cross-encoder', llm_backend='none', entity_backend='none', consensus_cosine=0.7).)
  - History: \`git -C /Users/connor/projects/memovox log --oneline -8\`.

ROUND 7: round 6 found ONE blocking MED defect, now FIXED (commit e582ddf) and verified. VERIFY it
holds with NO regression, then make a genuine final attempt to find ANY remaining >= MED fixable
defect. (All prior-round fixes also still hold: bare-speaker ALLCAPS-only strip; the distinctive-
token near-mirror gate on BOTH the contradiction and consensus paths incl. contraction artifacts +
discourse frames + CTA/'but'; watch/car topicality; consensus NLI-confirmed; cluster_claims default
write_edges=False.)
  PRESIDENT_ROLE_WORD_OOC_LEAK — a generic political/leadership ROLE word ('president' df=10) alone
  cleared the topicality gate while the real subject was below the df floor, so 'who is the president
  of Brazil?' ('brazil' df=3) / 'what is the vice president?' fabricated confident OOC answers. FIX:
  added zero-domain role words (president/vice/minister/senator/governor/mayor/chancellor/premier/
  dictator) to _COMMON_WORDS in src/memovox/augur/answer.py, DELIBERATELY EXCLUDING king/queen/prince/
  emperor (this corpus discusses king/queen AS chess subjects, df=40/37). VERIFY on the real store
  (see PROBES): 'who is the president of Brazil?' / 'what is the vice president?' now REFUSE; chess
  'what is the best move for the king?' still ANSWERS; watch/car still answer.
  ALREADY-ACCEPTED TRULY_IMPOSSIBLE (chair-ruled in rounds 5-6 with direct measurement — do NOT
  re-raise as fixable without exhibiting a NEW separating signal that does not regress OOC refusal or
  in-corpus answering): (a) an embedding-aware (BGE-M3 cosine / cross-encoder) OOC gate — measured
  legit answers score BELOW must-refuse cases, no threshold separates; (b) WATCH_VERB_SENSE_OOC_LEAK
  word-sense collision; (c) 'president of the UNITED STATES' residual (united/states are genuine
  high-df corpus tokens); (d) cross-video differently-phrased contradiction recall (no clean lexical
  near-mirror; dropping the gate floods 8k-32k DeBERTa-hallucinated garbage); (e) synonym consensus
  (corpus-limited, max cross-video cosine 0.62 < 0.7).

RIGOROUSLY-ESTABLISHED EVIDENCE you must factor in (measured this session, not assumed):
  (1) Cross-video CONTRADICTION via per-pair NLI is [nli]-UNSOLVABLE on this corpus. Dropping
      the lexical near-mirror gate and letting DeBERTa judge topic-cluster / token-blocking /
      cosine candidates produced 8,069–31,960 GARBAGE contradictions (DeBERTa hallucinates
      high-confidence contradictions on decontextualized cross-video fragments, e.g. "the ML
      world is mature" vs "they never talk" → 0.98). It was tried (commit 9dc70d5) and REVERTED
      (7e461e7). The high-precision lexical near-mirror gate (→ ~0 on this corpus, since the
      corpus has no clean near-mirror cross-video contradictions) is the CORRECT behavior.
  (2) Synonym CONSENSUS is corpus-limited: only ~9 substantive AGI claims across 2 videos, max
      cross-video cosine 0.62 (AGI-related, not paraphrase AGREEMENTS) — so consensus=0 is correct.
  (3) The clean semantic win is RETRIEVAL (dense ranking) + rerank; these need no candidate-gen.

Classify each finding: FIXABLE (a real defect/regression fixable in code now — garbage output,
fabrication, OOC leak, over-refusal regression, an embedding-aware-gate improvement that clearly
helps WITHOUT breaking OOC refusal), or TRULY_IMPOSSIBLE (genuinely unsolvable even with
BGE-M3+DeBERTa — justify with the evidence above or your own measurement), or ENV. Do NOT
hand-wave; but a RIGOROUSLY-PROVEN impossibility is NOT a defect. Work from /Users/connor/projects/memovox.
`

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['findings', 'summary'],
  properties: {
    findings: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['code', 'severity', 'classification', 'title', 'evidence', 'fix_suggestion'],
      properties: {
        code: { type: 'string' }, severity: { type: 'string', enum: ['CRIT', 'HIGH', 'MED', 'LOW'] },
        classification: { type: 'string', enum: ['FIXABLE', 'TRULY_IMPOSSIBLE', 'ENV'] },
        title: { type: 'string' }, evidence: { type: 'string' }, fix_suggestion: { type: 'string' } } } },
    summary: { type: 'string' },
  },
}

phase('Review')
const lenses = [
  { key: 'retrieval-relevance', prompt: `${COMMON}

YOUR LENS: dense retrieval + the relevance/OOC gate.
  - From ${PROBES}: in-corpus asks should ANSWER with on-topic citations; OOC asks should REFUSE
    (low_evidence). Flag any FABRICATION (OOC answered) or OVER-REFUSAL (in-corpus refused).
  - Cross-encoder rerank + dense retrieval: is the top citation on-topic? better than lexical free path?
  - The relevance GATE is still LEXICAL. Examine the hard edge cases (save energy / Steve Jobs death /
    walk my dog / BMW M3 / superintelligence). Is an EMBEDDING-AWARE gate a clear FIXABLE win, or would
    it break OOC refusal? Be concrete (would semantic coverage fix word-sense WITHOUT leaking OOC?).` },
  { key: 'contradiction-consensus', prompt: `${COMMON}

YOUR LENS: contradiction + consensus on the upgraded path.
  - Confirm the persistent graph is CLEAN (no garbage edges): open ${STORE}/memovox.db 'edges'.
    Report CONTRADICTS/SUPPORTS counts; sample SUPPORTS for quality. (The 28k-garbage dense
    contradiction was reverted — verify it's gone.)
  - ${PROBES} contradictions + synthesize: are the (few) surfaced pairs real? is consensus correct?
  - Adjudicate the contradiction/consensus limits against EVIDENCE (1)(2): are they genuinely
    TRULY_IMPOSSIBLE on this corpus, or did the reviewers/I miss a FIXABLE high-precision approach?` },
  { key: 'data-quality', prompt: `${COMMON}

YOUR LENS: data-quality + regressions from the revert.
  - Confirm the free-path hardening still holds (no leaked brackets/entities/speaker dups) — sample
    claims in ${STORE}/memovox.db.
  - moments dropped vs free (BGE-M3 merges more): sample claims — coherent or fragments? Does fragment
    quality cause any USER-FACING defect on the upgraded path (vs just limiting contradiction)?
  - Any regression introduced by the dense-feature revert (7e461e7)? Full suite is 617 pass/3 skip.` },
]
const reviews = await parallel(lenses.map(l => () =>
  agent(l.prompt, { label: `review:${l.key}`, phase: 'Review', schema: FINDINGS_SCHEMA })))

phase('Adjudicate')
const CHAIR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['team_satisfied', 'blocking', 'accepted_truly_impossible', 'rationale'],
  properties: {
    team_satisfied: { type: 'boolean' },
    blocking: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['code', 'severity', 'why_blocking', 'fix'],
      properties: { code: { type: 'string' }, severity: { type: 'string' }, why_blocking: { type: 'string' }, fix: { type: 'string' } } } },
    accepted_truly_impossible: { type: 'array', items: { type: 'string' } },
    rationale: { type: 'string' },
  },
}
const chair = await agent(`${COMMON}

YOU ARE THE CHAIR. Reviews below. Adjudicate.
REVIEWS:
${JSON.stringify(reviews.map((r, i) => ({ lens: lenses[i].key, ...r })), null, 2)}

team_satisfied=true ONLY if there is NO FIXABLE finding of severity >= MED (genuine defect or
regression on the upgraded path: garbage edges, fabrication, OOC leak, over-refusal regression,
or a clearly-worthwhile embedding-gate fix that won't break OOC refusal). A finding may be set
aside as accepted_truly_impossible ONLY with rigorous justification (the measured evidence that
even BGE-M3+DeBERTa cannot resolve it). Be fair: a high-precision system that emits NOTHING false
and whose only residuals are proven-impossible is SATISFIED — do not invent blockers, but do not
hand-wave a real fixable defect either. Every blocking item needs a concrete fix.`,
  { label: 'chair', phase: 'Adjudicate', schema: CHAIR_SCHEMA })

return { reviews: reviews.map((r, i) => ({ lens: lenses[i].key, ...r })), chair }
