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
        claim = Claim(
            claim_id=make_claim_id(moment.moment_id, len(claims)),
            moment_id=moment.moment_id,
            video_id=moment.video_id,
            text=sentence,
            subject=subject,
            predicate=predicate,
            object=obj,
            claim_type=epistemic_type(sentence),
            t_start_s=moment.t_start_s,
            t_end_s=moment.t_end_s,
            speaker_id=moment.speaker_id,
        )
        claim.salience = salience_score(claim)
        claims.append(claim)
    return claims


_LLM_SYSTEM = (
    "You extract atomic factual claims from a transcript span. Return ONLY a JSON "
    "array; each item: {\"text\": str, \"subject\": str, \"predicate\": str, "
    "\"object\": str, \"type\": one of FACT|DEFINITION|OPINION|PROCEDURE|EXAMPLE|"
    "PREDICTION|CORRECTION}. Extract only claims stated in the text; do not infer."
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
        claim = Claim(
            claim_id=make_claim_id(moment.moment_id, len(claims)),
            moment_id=moment.moment_id,
            video_id=moment.video_id,
            text=text,
            subject=str(item.get("subject", "")),
            predicate=str(item.get("predicate", "")),
            object=str(item.get("object", "")),
            claim_type=str(item.get("type", "FACT")).upper(),
            t_start_s=moment.t_start_s,
            t_end_s=moment.t_end_s,
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
