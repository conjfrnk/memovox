"""Claim extraction, epistemic typing, and salience (spec §5).

Default extractor is rule-based and dependency-free: it turns declarative
sentences of a Moment into atomic claims, tied to the Moment's source span. When
a generative LLM backend is configured it is used for richer extraction, but the
NLI gate (verify.py) still guards everything that lands in the graph.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from ..backends.base import LLMBackend
from ..loom.models import Claim, Moment
from ..util import make_claim_id, split_sentences, tokenize
from .spans import locate_span

# Ordered (subject, keyword) predicate cues for naive S-P-O splitting.
_PREDICATE_CUES = [
    "is defined as", "refers to", "is a", "is an", "are a", "are the",
    "recommends", "recommend", "requires", "uses", "use", "shows", "means",
    "equals", "causes", "improves", "reduces", "increases", "decreases",
    "enables", "provides", "is", "are", "was", "were", "has", "have",
]

# Epistemic-type cue lists, checked in priority order.
_TYPE_CUES = [
    ("CORRECTION", ["actually", "i misspoke", "correction", "i meant", "let me correct", "to correct"]),
    ("OPINION", ["i think", "i believe", "in my opinion", "i feel", "arguably", "imo",
                 "personally", "i'd say", "i would say", "probably", "it seems"]),
    ("PREDICTION", ["will ", "going to", "i expect", "we expect", "i predict",
                    "in the future", "by 20", "soon"]),
    ("PROCEDURE", ["first,", "firstly", "secondly", "step ", "then ", "next,",
                   "you should", "install", "run ", "click", "type ", "open the"]),
    ("EXAMPLE", ["for example", "for instance", "e.g", "such as"]),
    ("DEFINITION", ["is defined as", "means ", "refers to", " is a ", " is an ", " are a ", "definition"]),
]


# Entity-mention surface-form recognizers (W2.1). Acronyms: a run of >=2
# uppercase letters/digits, optionally hyphenated (BERT, GPT, RAG, GPT-4), with
# an optional trailing lowercase ``s`` plural (LLMs, GPUs) captured but stripped
# so the bare acronym is emitted and unifies with the singular. Title-case runs:
# one or more consecutive capitalized words (Transformer, Geoffrey Hinton,
# New York) treated as a single mention.
_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]+(?:-[A-Z0-9]+)?)s?\b")
_TITLECASE_RUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# Common sentence-initial / function words: a single Title-case word matching
# one of these is dropped (catches sentence-initial capitalization, which is NOT
# evidence of proper-noun-hood). Acronyms are exempt. Conversational speech (e.g.
# a YouTube vlog) opens far more sentences with capitalized common words than the
# academic register the original list targeted, so this is deliberately broad —
# but it omits words that frequently LEAD a real multi-word entity run (months,
# common first names like Will/May/Mark), since _strip_leading_common would shear
# them off the front of e.g. "Will Smith".
_COMMON_CAPS = {
    # determiners / articles / quantifiers
    "The", "This", "That", "These", "Those", "A", "An", "All", "Both", "Each",
    "Every", "Some", "Any", "Many", "Much", "Most", "Few", "Several", "Either",
    "Neither", "Another", "Other", "Such", "Enough", "More", "Less", "Half", "No",
    # pronouns / possessives
    "It", "We", "You", "They", "He", "She", "I", "Me", "Him", "Them", "Us", "My",
    "Your", "His", "Her", "Its", "Our", "Their", "Mine", "Yours", "Ours", "Theirs",
    "Who", "Whom", "Whose", "What", "Which", "Whatever", "Whoever", "Anyone",
    "Everyone", "Someone", "Nobody", "Everybody", "Somebody", "Anybody", "Nothing",
    "Something", "Everything", "Anything",
    # connectives / ordinals / discourse markers
    "First", "Second", "Third", "Then", "Next", "Finally", "Also", "However",
    "Therefore", "In", "On", "At", "For", "And", "Or", "But", "So", "If", "When",
    "While", "Here", "There", "Now", "Today", "Tonight", "Tomorrow", "Yesterday",
    "Yet", "Still", "Even", "Just", "Only", "Once", "Twice", "Again", "Always",
    "Never", "Often", "Sometimes", "Usually", "Maybe", "Perhaps", "Probably",
    "Actually", "Basically", "Honestly", "Literally", "Really", "Truly", "Clearly",
    "Obviously", "Certainly", "Definitely", "Absolutely", "Anyway", "Anyways",
    "Besides", "Instead", "Otherwise", "Meanwhile", "Thus", "Hence", "Moreover",
    "Furthermore", "Although", "Though", "Because", "Since", "Before", "After",
    "Until", "Unless", "Whereas", "Plus", "Why", "How", "Where", "Whether",
    # interjections / conversational openers
    "Yes", "Not", "Nope", "Yeah", "Yep", "Okay", "OK", "Oh", "Ah", "Uh", "Um",
    "Hey", "Hi", "Hello", "Wow", "Well", "Sure", "Right", "Look", "Listen", "See",
    "Watch", "Wait", "Hold", "Come", "Let", "Like", "Please", "Thanks", "Thank",
    "Sorry", "Welcome",
    # common auxiliary / light verbs (very common sentence starters in speech)
    "Is", "Are", "Was", "Were", "Be", "Been", "Being", "Am", "Do", "Does", "Did",
    "Done", "Have", "Has", "Had", "Can", "Could", "Would", "Should", "Might",
    "Must", "Get", "Got", "Want", "Need", "Think", "Know", "Said", "Says", "Say",
    "Go", "Goes", "Going", "Went", "Make", "Made", "Take", "Took", "Give", "Gave",
    "Tell", "Told", "Want", "Want",
}


def _strip_leading_common(run: str) -> str:
    """Strip leading stoplisted words from a Title-case run.

    A common word leading a multi-word run (sentence-initial "The Transformer
    architecture", "In New York", "This Chinchilla model") must not be emitted
    glued to the real entity, or W2.3's surface-form unification fragments it
    ("The Transformer" never collapses onto "Transformer"). Iteratively drop
    leading words in ``_COMMON_CAPS`` and return the remainder; returns "" when
    nothing is left (or the lone survivor is itself stoplisted), signalling drop.
    """
    words = run.split()
    while words and words[0] in _COMMON_CAPS:
        words.pop(0)
    if len(words) == 1 and words[0] in _COMMON_CAPS:
        return ""
    return " ".join(words)


def extract_mentions(claim: Claim) -> List[str]:
    """Surface candidate entity mentions from a claim (W2.1).

    Scans ``claim.text`` and the (often-overlapping) ``claim.subject``/
    ``claim.object`` fields for two kinds of surface form: all-caps acronyms
    (always kept, even sentence-initial; a trailing plural ``s`` is stripped so
    "LLMs" emits "LLM") and Title-case runs (a multi-word run is one mention).
    Common words are dropped via a stoplist -- both a standalone single capital
    and any leading common word(s) of a run ("The Transformer architecture" ->
    "Transformer", "In New York" -> "New York"). Returns DISTINCT surface forms
    in first-seen order; W2.3 canonicalizes them into graph entities.

    Known limits (free heuristic; out of scope here): does NOT capture
    CamelCase / internal-capital names ("OpenAI", "DeepMind", "iPhone") or
    accented tails ("Schoelkopf"/"Schölkopf"). These are left for the optional
    W2.2 Wikidata linker to recover.
    """
    seen: set = set()
    out: List[str] = []

    def _add(surface: str) -> None:
        if surface and surface not in seen:
            seen.add(surface)
            out.append(surface)

    for field_text in (claim.text, claim.subject, claim.object):
        if not field_text:
            continue
        # Acronyms first: emit the bare acronym (group 1 drops any plural "s").
        acronyms = set()
        for m in _ACRONYM_RE.finditer(field_text):
            bare = m.group(1)
            acronyms.add(bare)
            _add(bare)
        for m in _TITLECASE_RUN_RE.finditer(field_text):
            surface = _strip_leading_common(m.group(0))
            if not surface:
                continue
            # Skip a remaining standalone common-word capital ("The", etc.).
            if " " not in surface and surface in _COMMON_CAPS:
                continue
            # Defensive: never let a title-case match shadow an acronym already
            # taken from the same field (the patterns are disjoint, but guard).
            if surface in acronyms:
                continue
            _add(surface)

    return out


def epistemic_type(sentence: str) -> str:
    low = " " + sentence.lower().strip() + " "
    for label, cues in _TYPE_CUES:
        if any(cue in low for cue in cues):
            return label
    return "FACT"


def _spo(sentence: str):
    low = sentence.lower()
    for cue in _PREDICATE_CUES:
        idx = low.find(f" {cue} ")
        if idx > 0:
            subject = sentence[:idx].strip()
            obj = sentence[idx + len(cue) + 2:].strip().rstrip(".!")
            return subject, cue, obj
    return sentence.strip().rstrip(".!"), "", ""


def salience_score(claim: Claim) -> float:
    toks = tokenize(claim.text)
    if not toks:
        return 0.0
    length_score = min(1.0, len(toks) / 25.0)
    has_number = any(any(ch.isdigit() for ch in t) for t in toks)
    proper = sum(1 for w in claim.text.split()[1:] if w[:1].isupper())
    density = min(1.0, proper / 5.0)
    score = 0.5 * length_score + 0.25 * (1.0 if has_number else 0.0) + 0.25 * density
    return round(min(1.0, score), 4)


# W5.2 — conservative low-value-claim filter (precision over recall: better to keep
# a borderline claim than drop a real one). Demotes, never deletes.
_GREETING_STARTS = (
    "my name is", "my name's", "hello", "hi ", "hey ", "hey,", "welcome",
    "good morning", "good afternoon", "good evening",
    "thanks for", "thank you for", "thanks everyone", "thank you everyone",
)
# Anchored at the start so a content claim that merely MENTIONS sponsorship
# ("studies are sponsored by food companies") is not mistaken for an ad read.
_AD_RE = re.compile(
    r"^(support for |sponsored by |brought to you by "
    r"|this (episode|video|podcast|segment) is (sponsored|brought))")
_URL_RE = re.compile(r"(https?://|www\.|\.com\b|\.org\b|\.net\b|\.io\b)")
# Navigational/promotional imperatives only — NOT content verbs (look/see/watch),
# which routinely lead real assertions ("Look at the data showing…").
_IMPERATIVE_VERBS = {"subscribe", "click", "visit", "follow", "download", "sign"}


def is_non_claim(text: str) -> bool:
    """True for greetings, self-introductions, ad/sponsor reads, bare URLs, and
    short navigational imperatives — utterances that carry no verifiable assertion."""
    t = (text or "").strip()
    if len(t) < 2:
        return True
    low = t.lower()
    toks = tokenize(t)
    # Greeting/self-intro: only when it is essentially the WHOLE short utterance, so a
    # greeting PREFIX on a substantive sentence ("Hello everyone, today saturated fat
    # ...") is kept.
    if len(toks) <= 8 and any(low.startswith(g) for g in _GREETING_STARTS):
        return True
    if _AD_RE.match(low):
        return True
    if any(k in low for k in ("learn more", "check out", "go to ")) and _URL_RE.search(low):
        return True
    if toks and toks[0].lower() in _IMPERATIVE_VERBS and len(toks) <= 6:
        return True
    return False


def is_sentence_fragment(text: str) -> bool:
    """True for a continuation fragment — text starting lowercase, which a real
    sentence/claim never does. These are produced when a sentence is split across a
    Moment boundary, leaving a dangling tail ("number in your head, then…").
    ONLY meaningful for punctuated/cased transcripts — see is_low_value_claim."""
    t = (text or "").strip()
    return bool(t) and t[0].islower()


_SENTENCE_PUNCT_RE = re.compile(r"[.!?][\"')\]]?\s+[A-Z0-9\"'(]")


def transcript_is_punctuated(text: str) -> bool:
    """Does this transcript use real sentence punctuation (manual/cased captions),
    vs. unpunctuated all-lowercase auto-captions ("so today we want to talk about")?
    Continuation-fragment detection (which keys off leading case) is only valid for
    the former; on auto-captions EVERY sentence starts lowercase, so applying it
    would demote the entire video's claims."""
    t = (text or "").strip()
    if not t:
        return False
    # Multi-sentence cased text: a terminator followed by a capitalized next sentence.
    if _SENTENCE_PUNCT_RE.search(t):
        return True
    # A single cased sentence: starts uppercase AND ends with terminal punctuation.
    # The leading-case requirement avoids misreading an all-lowercase auto-caption
    # sentence (which merely ends in ".") as a fragment-checkable cased transcript.
    return t[0].isupper() and t.endswith((".", "!", "?"))


def is_low_value_claim(text: str, *, punctuated: bool = True) -> bool:
    """Demotion predicate (W5.2): a claim not worth committing to the trusted layer.
    Non-claim utterances always qualify; continuation fragments only when the source
    transcript is punctuated (else leading-case is not a fragment signal)."""
    if is_non_claim(text):
        return True
    return punctuated and is_sentence_fragment(text)


def extract_claims(
    moment: Moment, *, llm: Optional[LLMBackend] = None, min_words: int = 4
) -> List[Claim]:
    """Extract atomic claims from a Moment (LLM if available, else rule-based)."""
    if llm is not None and getattr(llm, "is_generative", False):
        try:
            claims = _extract_with_llm(llm, moment)
            if claims:
                return claims
        except Exception:
            pass  # fall back to deterministic extraction
    return _extract_rule_based(moment, min_words=min_words)


def _extract_rule_based(moment: Moment, *, min_words: int) -> List[Claim]:
    claims: List[Claim] = []
    for sentence in split_sentences(moment.transcript):
        sentence = sentence.strip()
        if sentence.endswith("?"):
            continue
        if len(tokenize(sentence)) < min_words:
            continue
        subject, predicate, obj = _spo(sentence)
        t0, t1 = locate_span(
            sentence, moment.segments,
            default=(moment.t_start_s, moment.t_end_s),
        )
        claim = Claim(
            claim_id=make_claim_id(moment.moment_id, len(claims)),
            moment_id=moment.moment_id,
            video_id=moment.video_id,
            text=sentence,
            subject=subject,
            predicate=predicate,
            object=obj,
            claim_type=epistemic_type(sentence),
            t_start_s=t0,
            t_end_s=t1,
            speaker_id=moment.speaker_id,
        )
        claim.salience = salience_score(claim)
        claims.append(claim)
    return claims


_LLM_SYSTEM = (
    "You extract atomic factual claims from a transcript span. Return ONLY a JSON "
    "array; each item: {\"text\": str, \"subject\": str, \"predicate\": str, "
    "\"object\": str, \"type\": one of FACT|DEFINITION|OPINION|PROCEDURE|EXAMPLE|"
    "PREDICTION|CORRECTION}. Extract only claims stated in the text; do not infer. "
    "The \"text\" field MUST be an EXACT VERBATIM substring of the transcript "
    "(copy the words as written; do not paraphrase, rephrase, or summarize) so it "
    "stays grounded in its source span; subject/predicate/object may be normalized."
)


def _extract_with_llm(llm: LLMBackend, moment: Moment) -> List[Claim]:
    prompt = f"TRANSCRIPT SPAN:\n{moment.transcript}\n\nJSON array of claims:"
    raw = llm.complete(prompt, system=_LLM_SYSTEM, temperature=0.0)
    data = _parse_json_array(raw)
    claims: List[Claim] = []
    for item in data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        t0, t1 = locate_span(
            text, moment.segments,
            default=(moment.t_start_s, moment.t_end_s),
        )
        claim = Claim(
            claim_id=make_claim_id(moment.moment_id, len(claims)),
            moment_id=moment.moment_id,
            video_id=moment.video_id,
            text=text,
            subject=str(item.get("subject", "")),
            predicate=str(item.get("predicate", "")),
            object=str(item.get("object", "")),
            claim_type=str(item.get("type", "FACT")).upper(),
            t_start_s=t0,
            t_end_s=t1,
            speaker_id=moment.speaker_id,
        )
        claim.salience = salience_score(claim)
        claims.append(claim)
    return claims


def _parse_json_array(raw: str) -> list:
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except ValueError:
        return []
