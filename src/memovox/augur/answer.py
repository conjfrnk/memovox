"""Augur answer synthesis (spec §5).

Composes a grounded answer **strictly from retrieved Moments**, with every
sentence carrying a citation to ``(video, timestamp, modality)``. The default
synthesizer is extractive (free, deterministic); a generative LLM is used when
configured. Low-evidence queries are flagged rather than confabulated.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..backends.base import Embedder, LLMBackend, Reranker
from ..config import Settings
from ..loom.models import make_provenance
from ..loom.store import LoomStore
from ..observe import Tracer
from ..util import split_sentences, tokenize, truncate
from .planner import decompose, llm_decompose
from .retrieve import retrieve
from .stitch import stitch_clips
from .types import Answer, Citation

_LOW_EVIDENCE_MSG = (
    "I don't have enough indexed evidence to answer that confidently. "
    "Try ingesting more sources or rephrasing."
)


# W5.1/W5.4 relevance gate. The signal asks: do the query's DISTINCTIVE terms appear,
# jointly, in a single cited moment? "Distinctive" = not a function word and not a
# common English word — so a generic word that is merely absent from a niche corpus
# ("arrive", "capital") neither vetoes a real question nor fakes relevance for an
# out-of-corpus one. The COMMON_WORDS list is deliberately generic (no corpus topic
# terms). NOTE: English-only; a non-English corpus falls back to df-based weighting.
_REL_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "as", "by", "with", "from", "we", "you", "they", "i", "not", "no",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "say", "said", "tell", "about", "their", "his", "her",
}

# Common English content words (verbs/nouns/adjectives) that are NOT topic-distinctive.
# Dropped from the relevance signal so they cannot veto a real question (a df=0 generic
# verb like "arrive") or fake relevance for an absent topic (an incidentally-rare word
# like "capital"/"boiling"). Curated to contain zero domain/topic terms — a word the actual
# corpus discusses AS a subject ("watch" -> luxury-watch reviews, "car" -> car reviews)
# must NOT live here, or a watch/car question whose only other tokens are framing words
# ("what watch is best for a first purchase?") loses its sole distinctive token and is
# wrongly refused. The df-topicality gate + per-moment coverage still hold the OOC line for
# an incidental verb use ("where can I watch the game?" — the absent subject 'game/football'
# is below min_df, so it still refuses).
_COMMON_WORDS = frozenset("""
able about above across act actual add after again against age ago agree air all
allow almost alone along already also although always among amount another answer any
anyone anything appear area around arrive ask available away back bad base because
become been before begin behind believe below best better between big bit body both
break bring build business call came can cannot care carry case catch cause center
certain chance change check child choose city claim class clear close come common
community company complete consider continue control cost could country couple course
cover create current cut data day deal decide deep develop die difference different
discuss done door down draw drive drop during each early easy eat effect either else
end enough enter even ever every example expect experience explain face fact fall family
far feel few field figure fill final find fine first follow food foot force form found
free friend full game general get give glad goes going gone good got great group grow
guess guy had half hand happen happy hard has have head hear heart held help here high
hold home hope hour house however huge idea important include increase inside instead
interest into issue item job join just keep kind knew know land large last late later
lead learn least leave left less let level life light like line list little live local
long look lose lot love low made main make man many matter may mean meet member might
mind minute miss moment money month more most move much must name near need never new
next nice night nor note nothing now number off offer office often okay old once only
open order other our out over own page part pass past pay people perhaps person pick
piece place plan play point possible power present press pretty probably problem process
provide pull purpose push put question quick quite rather reach read ready real reason
receive recent record remember report rest result return right rise room round run safe
said same save saw school season second see seem seen sell send sense set several shall
share short should show side simple since single small social some something sometimes
soon sort sound space speak special spend stand start state stay step still stop story
street strong study stuff such sure system take talk teach team tell thank their them
then there thing think this those though thought three through time today together told
too took top total touch toward tried true try turn type under understand until upon
use used usual very view visit wait walk want water way week well went were what
when where whether while white whole why will win window wish within without word work
world would write wrong year yes yet young
recommend recommended recommends recommendation recommendations suggest suggests
suggested suggestion suggestions advise advice purchase purchases purchased purchasing
buy buys buying bought sell sells selling sold choose chooses choosing chose pick picks
won wins winner winners winning lost lose loses losing beat beats beaten beating
happen happened happens happening become became becomes consider considered considers
invest invests investing invested investment investments worth value valued values
president presidents vice minister ministers senator senators governor governors mayor
mayors chancellor premier dictator
""".split())
# Generic political/leadership ROLE words are zero-domain-term for a tech/diet/watches/cars
# corpus: a query like "who is the president of Brazil?" must NOT clear topicality on the
# bare role word "president" (df=10, a recurring incidental mention) while its actual
# subject "brazil" (df=3, below floor) is absent — the round-4 "where can I watch the
# game?" leak class applied to role words. NB king/queen/prince/emperor are DELIBERATELY
# EXCLUDED: this corpus discusses them AS subjects (chess pieces — "king on G1", "queen on
# C7", df=40/37), so stripping them would over-refuse legitimate chess queries.


def _rel_tokens(text: str) -> set:
    """DISTINCTIVE tokens (topic candidates): function + common words removed."""
    return {t for t in tokenize(text)
            if t not in _REL_STOP and t not in _COMMON_WORDS and len(t) > 2}


# Generic filler verbs + query-framing meta-words. Dropped from COVERAGE so they
# neither fake coverage for an out-of-corpus query ("how do plants MAKE energy" — the
# absent subject 'plants' must not be rescued by the filler 'make') nor demand coverage
# that over-refuses a real one ("how does gravity WORK", "what do the SOURCES say about
# X"). Distinct from _COMMON_WORDS: context nouns (save, home, plants, dog) are KEPT.
_COVERAGE_FILLER = frozenset("""
make makes made making get gets got getting take takes took taking use uses used using
work works worked working go goes going come comes coming give gives gave given giving
find finds finding put puts putting keep keeps keeping let lets letting want wants wanting
need needs needing like likes look looks looking seem seems become becomes happen happens
mean means meant call calls called turn turns help helps start starts begin begins
show shows showed showing tell explain explains explaining describe describes mention
mentions discuss discusses discussing cover covers covering talk talks talking
source sources video videos clip clips talk speaker speakers lecture episode podcast
thing things way ways kind sort lot stuff topic topics idea ideas point points part parts
good bad great best better worse worst nice fine invest invests investing invested
investment investments worth value valued values
""".split())
# Value-judgment framing ('a GOOD INVESTMENT', 'WORTH it') is dropped from COVERAGE too —
# else "are watches a good investment?" over-refuses (cov_q {good,investment,watches} -> a
# watch moment covers only {watches}=1/3 < floor) despite the corpus being watch-purchase
# advice. df-based topicality is still the OOC guard (crypto/bitcoin df=0 -> still refuse).
# NOTE: generic advice/transaction verbs (recommend/suggest/buy/purchase...) live ONLY in
# _COMMON_WORDS (the TOPICALITY signal), NOT here in _COVERAGE_FILLER. Keeping them out of
# topicality closes the OOC leak ("recommend a first home purchase?" — no distinctive topic
# token); adding them to COVERAGE too would over-refuse a legitimate in-corpus query whose
# real subject IS a topic word ("which Rolex should I buy?" — rolex is a real 48-claim topic).


def _coverage_tokens(text: str) -> set:
    """CONTENT tokens for coverage: function/question words, plus generic filler verbs
    and query-framing meta-words, removed — but real CONTEXT nouns ("save", "home",
    "plants", "dog") are KEPT so the gate checks the cited moment shares the query's
    context (the polysemy defense) without being fooled or blocked by filler."""
    return {t for t in tokenize(text)
            if t not in _REL_STOP and t not in _COVERAGE_FILLER and len(t) > 2}


# A query token counts as a genuine corpus TOPIC (vs an incidental hapax) when it
# appears in at least this fraction of moments (with a small absolute floor). Used by
# the topicality gate in _relevance_coverage.
_RELEVANCE_TOPIC_DF_FRAC = 0.0009


def _relevance_coverage(store, query: str, citations: List["Citation"],
                        min_moments: int = 0) -> float:
    """Relevance of the CITED evidence to the query, in [0,1] (the W5 gate signal).
    Two robust, count-based checks (NOT IDF mass, which a single token could dominate):

      1. TOPICALITY — at least one DISTINCTIVE query token must be a genuine corpus
         topic: it recurs across moments (df >= min_df) OR names a cited video's title.
         This separates "what is AGI?" (agi in 14 moments) from "what time does the
         BANK open?" (bank in 1 moment, an incidental hapax) -> refuse.
      2. CONTEXT COVERAGE — the fraction of the query's CONTENT words (context words
         kept, not just topic words) that ONE cited moment covers. This is the polysemy
         defense: "how do I save ENERGY at home?" shares only "energy" with the physics
         moments (save/home absent) -> low coverage -> refuse, even though "energy"
         recurs (df=119) in a different sense.

    FUNDAMENTAL LIMITS (free/lexical path), all resolved by the optional sentence-
    transformers [embed] backend (dense query/moment similarity), none by bag-of-words:
      - WORD-SENSE collision: a topic word that genuinely co-occurs with its context
        words in a DIFFERENT sense ("train my DOG to sit" vs the CS229 'training a dog'
        RL analogy; "speed limit" vs "speed of light") cannot be disambiguated lexically.
      - VOCABULARY gap: a subject the corpus discusses under a DIFFERENT word
        ("superintelligence" spoken as "superhuman AI"; "backpropagation" as "back
        propagation") has doc_freq 0, so the topicality gate over-refuses it. A title
        carries the word but indexing titles into moment-FTS inflates title-word df and
        regresses retrieval, so it is intentionally NOT done here."""
    distinctive = _rel_tokens(query)
    cov_q = _coverage_tokens(query)
    try:
        n = max(1, int(store.stats().get("moments", 1)))
    except Exception:  # noqa: BLE001 - stats is best-effort; never break an answer
        n = 1
    if n < min_moments:
        return 1.0  # corpus too small for the df signal to be reliable
    if not cov_q:
        # A content-free query (only function/filler/framing words: "how does it
        # work?", "what do the sources say?") names no topic to answer -> refuse.
        # This MUST come after the small-corpus exemption but BEFORE coverage, else
        # the empty-coverage short-circuit would wrongly return full relevance.
        return 0.0

    min_df = max(2, round(_RELEVANCE_TOPIC_DF_FRAC * n))
    title_tokens: set = set()
    for c in citations:
        title_tokens |= _rel_tokens(c.title or "")
    topical = (any(store.doc_freq(t) >= min_df for t in distinctive)
               or bool(distinctive & title_tokens))
    if not topical:
        return 0.0  # no genuine corpus topic in the query -> refuse

    best = 0.0
    for c in citations:
        ctoks = _coverage_tokens((c.source_text or c.snippet or "") + " " + (c.title or ""))
        # The covering citation must actually contain a DISTINCTIVE topic token, not only
        # generic verb/framing words. Otherwise a verb-dominated query about a thin topic
        # ("what should I recommend for DINNER?", "who WON the world cup?") clears the
        # floor purely on shared "recommend/should/won" while the real subject is absent
        # from every cited moment — a confident answer to evidence that never mentions it.
        if not (distinctive & ctoks):
            continue
        covered = sum(1 for t in cov_q if t in ctoks)
        frac = covered / len(cov_q)
        if frac > best:
            best = frac
    return best


def _best_sentence(text: str, query: str) -> str:
    q_tokens = set(tokenize(query))
    best, best_score = "", -1
    for sentence in split_sentences(text) or [text]:
        overlap = sum(1 for w in tokenize(sentence) if w in q_tokens)
        if overlap > best_score:
            best, best_score = sentence, overlap
    return best.strip()


def _citation_text(moment) -> str:
    """The answerable CONTENT of a moment for snippet selection + LLM synthesis:
    the spoken transcript and any literal on-screen OCR text. The VLM's prose
    *description* of the frame (``visual_caption``) is deliberately excluded — it is
    a retrieval aid, not content, and a verbose caption ("The image shows a man
    wearing sunglasses…") otherwise wins the snippet and makes the synthesizer
    reason about the picture instead of what was said/shown. Falls back to the
    caption only for a pure-visual moment with no transcript/OCR, so such a moment
    still yields a non-empty citation."""
    parts = [p for p in (getattr(moment, "transcript", None),
                         getattr(moment, "ocr_text", None)) if p]
    if parts:
        return "\n".join(parts).strip()
    return (getattr(moment, "visual_caption", None) or "").strip()


def _includes_unverified_visual(moment) -> bool:
    """True when a moment's answerable content includes on-screen/visual material
    that did NOT pass the entailment gate. Claims are extracted from the spoken
    transcript only (assay.claims), so OCR text and a pure-visual caption are never
    verify-before-commit checked — a poisoned slide could otherwise reach an answer
    indistinguishable from vetted speech. Citations carrying such content are flagged
    ``ocr_unverified`` so clients can mark it lower-trust. Mirrors _citation_text's
    content rule: content = transcript + OCR, falling back to the caption only when
    there is no transcript and no OCR."""
    transcript = (getattr(moment, "transcript", None) or "").strip()
    ocr = (getattr(moment, "ocr_text", None) or "").strip()
    caption = (getattr(moment, "visual_caption", None) or "").strip()
    return bool(ocr) or (not transcript and bool(caption))


def _synthesize_extractive(citations: List[Citation], *, limit: int = 4) -> str:
    parts = []
    for c in citations[:limit]:
        snippet = c.snippet.strip()
        if not snippet:
            continue
        if snippet[-1] not in ".!?":
            snippet += "."
        parts.append(f"{snippet} [{c.index}]")
    return " ".join(parts)


_LLM_SYSTEM = (
    "Answer the question using ONLY the numbered sources. Cite every sentence "
    "with the matching [n]. If the sources don't answer it, say so. Be concise."
)


_CITE_MARKER_RE = re.compile(r"\[(\d+)\]")
#: WITHIN-LINE clause boundary: a terminator (. ! ? ;) + same-line whitespace. PHYSICAL
#: line breaks are handled separately via ``str.splitlines()``, which covers EVERY unicode
#: line/paragraph separator (\n \r \r\n \v \f \x1c-\x1e \x85    ) — so no exotic
#: separator can smuggle an uncited line past the gate (closing the \n-only whack-a-mole).
#: Commas/colons/dashes are deliberately NOT boundaries: they join clauses WITHIN one
#: sentence ("Ribosomes, which are organelles, make proteins [1]"), so splitting on them
#: would reject legitimate single-citation sentences. Sentence boundaries within a line are
#: detected in code (see :func:`_is_sentence_boundary` / :func:`_gate_clauses`), not a single
#: regex, so short words, accents, CJK, acronyms, decimals and abbreviations are all handled.
#: A 2+ letter run = real prose (so "[1]", ".", ", " alone are not "prose"). Unicode-aware,
#: so accented / CJK / non-Latin words count as prose, like the boundary detector below.
_GATE_WORD_RE = re.compile(r"[^\W\d_]{2,}")
#: A citation marker AFTER a terminator ("acids. [1]", incl. CJK 。！？) cites the PRECEDING
#: sentence (the extractive synthesizer writes "acids. [1]"). Pull it BEFORE the terminator
#: before clause-splitting — only across SAME-LINE whitespace ([^\S\n]), so a marker on the
#: next line never binds backward across a line break.
_MARKER_BIND_RE = re.compile(r"([.!?;。！？…]+)([^\S\n]+)(\[\d+\](?:[^\S\n]*\[\d+\])*)")
#: Sentence terminators: ASCII . ! ? ; the CJK ideographic/fullwidth 。 ！ ？, and the unicode
#: ellipsis … (U+2026) — many LLMs emit "…" for "...", so it must gate like ASCII "...".
_GATE_TERMINATORS = ".!?;。！？…"
#: KNOWN abbreviations whose trailing period is NOT a sentence boundary. An ALLOWLIST (not
#: "any short Capitalized word"): a short proper noun ("Pope", "God", "Rome", "King") is a
#: real word that DOES end a sentence — treating it as an abbreviation let an uncited
#: sentence ending in one merge into the next cited clause and leak. A rare true abbreviation
#: outside this set (e.g. "Sun."/"Mon.") merely over-refuses to the extractive synthesizer.
_GATE_ABBREVIATIONS = frozenset(
    "mr mrs ms dr st jr sr vs etc inc ltd co corp fig no vol pp ed al eg ie eg "
    "prof gen rev hon sen rep dept est mt ave blvd approx esp dr".split())


def _is_sentence_boundary(line: str, i: int) -> bool:
    """True iff the terminator at ``line[i]`` ends a sentence (vs. an abbreviation /
    initialism / decimal). Done in code, not one mega-regex: lexical sentence segmentation
    has too many interacting cases (short words, accents, CJK, acronyms, decimals,
    abbreviations) to keep correct as a single pattern. Bias is FAIL-CLOSED — a plain
    word/number/percent/acronym/marker end IS a boundary (so an uncited sentence gets split
    off and rejected); only a clear abbreviation/initialism/decimal is NOT. Worst case, an
    abbreviation-bearing generative answer is downgraded to the faithful extractive
    synthesizer, which never leaks an uncited assertion."""
    ch = line[i]
    if ch in "。！？…":
        return True  # CJK/fullwidth terminator or unicode ellipsis: always a boundary
    nxt = line[i + 1] if i + 1 < len(line) else ""
    if ch == "." and nxt.isdigit():
        return False  # decimal / version (5.99, v2.0, 3.14)
    prev = line[i - 1] if i > 0 else ""
    if prev == "]":
        return True   # sentence ending at a citation marker ("... [1]. Next")
    if prev == "%" or prev.isdigit():
        return True   # ends in a percent ("90%.") or a number/year ("2019.")
    j = i  # the alphabetic token ending just before the terminator
    while j > 0 and (line[j - 1].isalpha() or line[j - 1] in "'’"):
        j -= 1
    token = line[j:i]
    if len(token) <= 1:
        return False  # single letter -> initialism ("U.S.", "e.g.") or no real token
    if token.lower() in _GATE_ABBREVIATIONS:
        return False  # KNOWN abbreviation (Dr, Mr, Inc, Fig, etc.) — allowlist, not "short Cap"
    return True  # plain word (incl. a short proper noun like "Pope"/"God"), acronym, or number


def _gate_clauses(line: str):
    """Split one physical line into clauses at sentence boundaries (see _is_sentence_boundary)."""
    clauses, start, i, n = [], 0, 0, len(line)
    while i < n:
        if line[i] in _GATE_TERMINATORS and _is_sentence_boundary(line, i):
            while i < n and line[i] in _GATE_TERMINATORS:  # consume the terminator run
                i += 1
            clauses.append(line[start:i])
            start = i
        else:
            i += 1
    if start < n:
        clauses.append(line[start:])
    return clauses


def _llm_citations_valid(text: str, citations: List[Citation]) -> bool:
    """True iff a GENERATED answer is safe to surface as-is: every assertion is ATTRIBUTED
    to a real citation (the never-break invariant). Requires:
      * at least one ``[n]``, and every ``[n]`` resolves to a real citation index, and
      * no prose follows the final ``[n]`` (no uncited trailing clause), and
      * every physical line (``str.splitlines()`` — all unicode line breaks) and every
        sentence-clause within it (see :func:`_gate_clauses`) carries a ``[n]``.
    On any failure the caller falls back to the verified extractive synthesizer.

    SCOPE: this enforces citation STRUCTURE — nothing is asserted without pointing at a
    source. It does NOT verify the cited text is semantically ENTAILED by that source:
    faithfulness of a generative paraphrase is the model's responsibility (constrained by
    the system prompt) and is reliably checkable only with an NLI backend, not lexically
    (a faithful synonym paraphrase shares no tokens with its source, while a topic-matched
    fabrication shares the subject word — so token overlap separates neither). The default
    free path never hits this — it uses the extractive synthesizer, faithful by construction."""
    text = (text or "").strip()
    if not text:
        return False
    valid = {c.index for c in citations}
    markers = list(_CITE_MARKER_RE.finditer(text))
    if not markers or any(int(m.group(1)) not in valid for m in markers):
        return False  # no citation at all, or a dangling marker -> reject
    bound = _MARKER_BIND_RE.sub(r"\3\1\2", text)
    # no uncited prose may follow the final marker (uncited tail)
    if _GATE_WORD_RE.search(bound[list(_CITE_MARKER_RE.finditer(bound))[-1].end():]):
        return False
    # every physical line, and every sentence-clause within it, must carry a marker
    for line in bound.splitlines():
        for clause in _gate_clauses(line):
            if _GATE_WORD_RE.search(clause) and not _CITE_MARKER_RE.search(clause):
                return False
    return True


def _synthesize_llm(llm: LLMBackend, query: str, citations: List[Citation]) -> str:
    # Give the LLM each citation's FULL content (source_text) rather than the
    # one-sentence display snippet, so an answer-bearing sentence with no
    # query-token overlap is still visible. Falls back to the snippet if unset.
    sources = "\n".join(f"[{c.index}] {c.source_text or c.snippet}" for c in citations)
    prompt = f"SOURCES:\n{sources}\n\nQUESTION: {query}\n\nANSWER (with [n] citations):"
    return llm.complete(prompt, system=_LLM_SYSTEM, temperature=0.0).strip()


def _apply_rerank(reranker, query, fused, store):
    """Rerank a fused (id, score) list (no-op for identity / empty / None)."""
    if reranker is None or not fused:
        return fused
    texts = None
    if reranker.needs_text:
        texts = {m.moment_id: m.text_for_embedding()
                 for m in store.get_moments([mid for mid, _ in fused])}
    return reranker.rerank(query, fused, texts=texts)


def _merge_round_robin(legs, top_k):
    """Interleave per-sub-query fused lists (rank-0 of each, then rank-1, …),
    de-duplicating moments and capping at ``top_k`` — so every clause contributes
    its top result(s) before any clause's deeper results (per-clause coverage WHEN
    ``top_k >= len(legs)``; with fewer slots than clauses the later clauses are
    intentionally dropped, the deterministic cost of a small top_k)."""
    merged: List[Tuple[str, float]] = []
    seen = set()
    depth = 0
    while len(merged) < top_k and any(depth < len(leg) for leg in legs):
        for leg in legs:
            if depth < len(leg):
                mid, score = leg[depth]
                if mid not in seen:
                    seen.add(mid)
                    merged.append((mid, score))
                    if len(merged) >= top_k:
                        break
        depth += 1
    return merged


def ask(
    store: LoomStore,
    query: str,
    *,
    embedder: Embedder,
    llm: Optional[LLMBackend] = None,
    settings: Optional[Settings] = None,
    video_id: Optional[str] = None,
    tracer: Optional[Tracer] = None,
    modality: str = "any",
    visual_query_vec: Optional[List[float]] = None,
    reranker: Optional["Reranker"] = None,
) -> Answer:
    settings = settings or Settings()
    tracer = tracer or Tracer("ask", otel_enabled=settings.otel_enabled)
    # Decompose the query (single-clause => one verbatim sub-query). The LLM
    # decomposer is opt-in (planner_agentic + a generative LLM) with a guaranteed
    # deterministic fallback; the free path is always the deterministic decompose.
    if getattr(settings, "planner_agentic", False) and llm is not None and \
            getattr(llm, "is_generative", False):
        qp = llm_decompose(llm, query)
    else:
        qp = decompose(query)
    multi = len(qp.subqueries) > 1
    plan_dicts = [sq.to_dict() for sq in qp.subqueries]
    # The VISUAL leg (M1.1) turns on when the plan routes to visual OR the caller
    # explicitly asks for modality="visual"; it only fires if a visual query vector
    # is supplied (e.g. an image query), so a plain text ask is byte-identical.
    use_visual = (settings.visual_retrieval
                  and (qp.modality == "visual" or qp.strategy == "visual"
                       or modality == "visual"))
    # Consume the plan: the strategy chooses the retrieval mode (and, below, the
    # citation ordering) — it is NOT decorative. Only the contradiction route
    # turns on the graph leg today, walking CONTRADICTS/SUPPORTS edges to surface
    # the OTHER side of a disagreement (a moment that shares no query terms). We
    # deliberately do NOT follow ELABORATES: it is emitted intra-moment only, so
    # following it never reaches a new moment. hybrid/procedure/visual keep the
    # dense+lexical baseline — which is what the (factual) eval queries route to,
    # so the retrieval gates stay green.
    # SUPPORTS is included alongside CONTRADICTS on purpose: a contradiction answer
    # can then surface both the disagreement AND corroborating context. The
    # extractive synthesizer cites every surfaced moment neutrally, so adding
    # SUPPORTS only widens evidence. (Per-clause graph routing is derived inside the
    # multi branch from each sub-query's strategy, NOT from qp.strategy.)
    with tracer.span("retrieve") as _sp:
        if not multi:
            # SINGLE-CLAUSE: the literal today's path (retrieve over the full query),
            # so the output stays byte-identical.
            use_graph = qp.strategy == "contradiction"
            graph_rels = ["CONTRADICTS", "SUPPORTS"] if use_graph else None
            fused = retrieve(
                store, query, embedder=embedder, settings=settings, video_id=video_id,
                use_graph=use_graph, graph_rels=graph_rels, span=_sp,
                use_visual=use_visual, visual_query_vec=visual_query_vec,
            )
        else:
            # MULTI-PART (spec §5): retrieve + rerank EACH sub-query (the rerank sees
            # a focused clause), then merge round-robin so every clause is covered.
            legs = []
            for sq in qp.subqueries:
                sq_graph = sq.strategy == "contradiction"
                sq_rels = ["CONTRADICTS", "SUPPORTS"] if sq_graph else None
                sq_visual = settings.visual_retrieval and (
                    sq.modality == "visual" or sq.strategy == "visual")
                leg = retrieve(
                    store, sq.text, embedder=embedder, settings=settings, video_id=video_id,
                    use_graph=sq_graph, graph_rels=sq_rels,
                    use_visual=sq_visual, visual_query_vec=visual_query_vec,
                )
                legs.append(_apply_rerank(reranker, sq.text, leg, store))
            fused = _merge_round_robin(legs, settings.top_k)
        _sp.add_counter("results", len(fused))
    if not fused:
        return Answer(text=_LOW_EVIDENCE_MSG, citations=[], strategy=qp.strategy,
                      low_evidence=True, metrics=tracer.to_dict(), plan=plan_dicts)

    # M2.1 rerank stage (spec §5/§3): the single-clause path reranks the fused set
    # here (the multi-part path already reranked each sub-query above). The free
    # identity reranker is a no-op -> [n] indices/deep links re-derive contiguously.
    if reranker is not None and not multi:
        with tracer.span("rerank") as _rsp:
            fused = _apply_rerank(reranker, query, fused, store)
            _rsp.add_counter("candidates", len(fused))

    with tracer.span("synthesize") as _sp:
        moment_ids = [mid for mid, _ in fused]
        score_by_id = dict(fused)
        moments = store.get_moments(moment_ids)

        citations: List[Citation] = []
        video_cache: dict = {}
        for i, moment in enumerate(moments, start=1):
            video = video_cache.get(moment.video_id)
            if video is None:
                video = store.get_video(moment.video_id)
                video_cache[moment.video_id] = video
            prov = make_provenance(
                video, moment.t_start_s, moment.t_end_s,
                modality=moment.modality, speaker=moment.speaker_id,
                confidence=round(min(1.0, score_by_id.get(moment.moment_id, 0.0) * 30), 4),
            ) if video else None
            content = _citation_text(moment)
            snippet = _best_sentence(content, query)
            citations.append(
                Citation(
                    index=i,
                    video_id=moment.video_id,
                    moment_id=moment.moment_id,
                    t_start_s=moment.t_start_s,
                    t_end_s=moment.t_end_s,
                    modality=moment.modality,
                    speaker=moment.speaker_id,
                    title=video.title if video else None,
                    deep_link=prov.deep_link if prov else None,
                    snippet=truncate(snippet, 240),
                    score=round(score_by_id.get(moment.moment_id, 0.0), 6),
                    ocr_unverified=_includes_unverified_visual(moment),
                    source_text=truncate(content, 600),
                )
            )

        if qp.strategy == "temporal":
            # Multi-hop temporal synthesis: order the cited moments chronologically
            # by their video's published_at (ascending), missing dates last, then
            # RE-INDEX so the [n] markers stay aligned with the new order. This must
            # happen BEFORE synthesis, because the extractive synthesizer emits
            # [c.index] and iterates citations in list order — both must reflect it.
            def _published_at(c: Citation) -> str:
                video = video_cache.get(c.video_id)
                return (video.published_at or "") if video else ""

            citations.sort(key=lambda c: (_published_at(c) == "", _published_at(c)))
            for new_index, c in enumerate(citations, start=1):
                c.index = new_index

        if llm is not None and getattr(llm, "is_generative", False):
            try:
                text = _synthesize_llm(llm, query, citations)
            except Exception as exc:  # noqa: BLE001 - graceful fallback, but visible
                import sys
                print(f"memovox: LLM answer synthesis failed ({type(exc).__name__}: "
                      f"{exc}); using the extractive synthesizer.", file=sys.stderr)
                text = _synthesize_extractive(citations)
            else:
                # GROUNDING GATE: a well-formed but uncited / fabricated-quote / dangling-
                # marker LLM answer is discarded for the verified extractive synthesizer —
                # we never surface an assertion we can't tie to a citation.
                if not _llm_citations_valid(text, citations):
                    text = _synthesize_extractive(citations)
        else:
            text = _synthesize_extractive(citations)

        # W5.1: a synthesized answer is only trustworthy if the cited evidence
        # actually covers what was asked. Gate on IDF-weighted query coverage so an
        # out-of-corpus question ("capital of Mongolia") is refused rather than
        # answered with the nearest-but-irrelevant moments. Withhold the citations
        # too — presenting them would itself be an unsupported claim.
        floor = getattr(settings, "answer_relevance_floor", 0.0)
        relevance = _relevance_coverage(
            store, query, citations,
            min_moments=getattr(settings, "answer_relevance_min_moments", 0),
        ) if floor > 0 else 1.0
        low_evidence = (not text.strip()) or relevance < floor
        if low_evidence:
            text = _LOW_EVIDENCE_MSG
            citations = []
        _sp.add_counter("relevance", round(relevance, 4))
        _sp.add_counter("citations", len(citations))
    # M2.3 clip stitching (spec §5/§8): a strictly ADDITIVE final step reading the
    # finalized citations — it widens cited spans into deep-linked watch windows and
    # never mutates text/citations, so the gates stay byte-identical.
    clips = stitch_clips(citations, videos=video_cache,
                         merge_gap_s=settings.clip_merge_gap_s)
    return Answer(text=text, citations=citations, strategy=qp.strategy,
                  low_evidence=low_evidence, metrics=tracer.to_dict(), plan=plan_dicts,
                  clips=clips)
