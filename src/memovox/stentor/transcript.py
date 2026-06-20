"""Transcript parsing and cleaning (Stentor).

Parses VTT / SRT / JSON / plain-text transcripts into :class:`Segment` objects,
strips filler tokens and bracketed audio events from the knowledge text while
retaining the events as timeline markers (spec §4 stage 2).
"""

from __future__ import annotations

import html
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from ..backends.base import Segment, Word
from ..util import split_sentences

FILLERS = {"um", "uh", "erm", "uhh", "umm", "mm", "hmm", "mhm", "uhm", "ah", "er"}
EVENT_RE = re.compile(
    r"\[\s*(music|applause|laughs?|laughter|laughing|silence|inaudible|noise|"
    r"crosstalk|cheering|chuckles?|singing|humming|instrumental|sighs?|groans?|"
    r"coughs?|gasps?|clears throat|beep|static|whistling|ticking|sizzling|"
    r"ringing|rings|dings?|bells?|footsteps|wind|rain|thunder)[^\]]{0,160}\]",
    re.IGNORECASE,
)  # bounded body {0,160}: caps the inner scan (O(n) linear) so "[music"-repeated with no
#   "]" can't drive O(n^2) backtracking, while still matching a long "[speaking in ...]" body
#: After the whitelist pass, a RESIDUAL ``[...]`` span in caption text is a non-speech
#: annotation the whitelist did not name — sound effects ([clock ticking]), foreign-
#: language markers ([speaking in Thai]), bracketed speaker labels ([Mark Wiens]), or
#: censored profanity ([ __ ]). Strip it so it never lands in a claim. Guard against
#: eating legitimate code/math brackets ([i], [b,t], [0]) by only stripping spans whose
#: inner text is a WORD annotation (a 3+ letter run) or all-underscore censorship.
_RESIDUAL_BRACKET_RE = re.compile(r"\[[^\]\n]{1,60}\]")
_BRACKET_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_BRACKET_CENSOR_RE = re.compile(r"^[\s_]+$")
#: Markdown link ``[label](url)`` (captions sometimes embed article links): keep the
#: visible label, drop the URL. Must run BEFORE residual-bracket stripping so the label
#: survives. ``_URL_PAREN_RE`` mops up an orphaned ``](url)`` / bare ``(url)`` left when
#: the sentence splitter broke inside a link (e.g. "…crazy short](https://…) post").
_MD_LINK_RE = re.compile(r"\[([^\]\n]{0,200})\]\((https?://[^)\s(]+)\)")  # [^)\s(]: ReDoS-safe (see _URL_PAREN_RE)
#: Excludes ``(`` from the URL body so a fresh ``(`` terminates the run instead of
#: letting the body span the whole string — without it, ``"(http://x"`` repeated makes
#: the outer ``\(`` retry at every paren offset, an O(n²) ReDoS (a ~600KB cue hangs for
#: minutes). A URL with a literal ``(`` (rare in captions) is simply left unstripped.
_URL_PAREN_RE = re.compile(r"\]?\((https?://[^)\s(]+)\)")
#: A cue timing line is ``<ts> --> <ts>`` with each side purely numeric. Used to reject a
#: prose line that merely CONTAINS ``-->`` (an arrow in speech) as a fake timestamp line.
_TS_SIDE_RE = re.compile(r"^[\d:.,]+$")
#: Musical-note markers wrap sung lyrics in captions. A line carrying one is music
#: (a song), NOT spoken video content, so it becomes a timeline event rather than a
#: claim. Unambiguous: these glyphs appear only for music, so there is no false
#: positive. (Sung lyrics transcribed as PLAIN text with no marker -- e.g. some
#: YouTube auto-captions -- can't be told apart from speech without audio.)
_MUSIC_NOTE_RE = re.compile(r"[♪♫♬\U0001f3b5\U0001f3b6]")
#: Bounded body ({1,200}, single-line) so a long unclosed "<" run can't drive O(n^2)
#: backtracking in _decode_entities — real caption tags (<v Name>, <00:00:01.199>, <i>) are
#: short and single-line.
_TAG_RE = re.compile(r"<[^>\n]{1,200}>")
_SPEAKER_VTT_RE = re.compile(r"<v\s+([^>\n]{1,200})>")  # bounded body: ReDoS-safe on a long "<v "-run
_SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 ._'-]{0,30}):\s+")
#: A leading ``Name:`` is only a speaker label if it *looks* like a name, not a
#: sentence that happens to contain a colon. Real caption labels are ALL-CAPS
#: ("NEWSCASTER", "BRADY HARAN") or Title-Case proper names ("Rob Wiblin",
#: "Molaison"), ≤3 words. Sentence fragments ("But caveat:", "For example:",
#: "And that worked great:", "I have three world records:") carry a lowercase
#: content word or run long — those are content, not a speaker, and must NOT be
#: hoisted into a bogus speaker id (the richer-corpus false positives).
_SPEAKER_PARTICLES = {"van", "von", "de", "del", "der", "da", "di", "la", "le",
                      "bin", "al", "of", "the", "y"}
_SPEAKER_DISCOURSE = {"but", "and", "for", "so", "well", "okay", "ok", "now", "yes",
                      "no", "oh", "then", "also", "because", "however", "i", "we",
                      "they", "he", "she", "it", "this", "that", "there", "here",
                      "what", "when", "why", "how", "anyway", "right",
                      # single-word section/interjection openers that end in a colon but
                      # are sentence framing, not a speaker label ("Note:", "Today:")
                      "note", "today", "look", "listen", "wait", "see", "first",
                      "second", "third", "next", "finally", "warning", "breaking",
                      "remember", "update", "tip", "summary", "conclusion", "basically",
                      "honestly", "actually", "literally", "meanwhile", "again",
                      "maybe", "perhaps", "alright", "hey", "welcome", "thanks", "plus"}


def _looks_like_speaker(name: str) -> bool:
    """True iff ``name`` looks like a caption speaker label, not a sentence fragment."""
    words = name.split()
    if not words or len(words) > 3:
        return False
    if len(words) == 1 and words[0].lower() in _SPEAKER_DISCOURSE:
        return False  # a lone discourse word ("Well:", "So:", "I:") is punctuation
    for w in words:
        alpha = re.sub(r"[^A-Za-z]", "", w)
        if not alpha:
            continue  # punctuation/initials-only token
        if alpha.isupper() or (alpha[0].isupper() and not alpha.islower()):
            continue  # ALLCAPS or Capitalized (incl. McDonald/DiCaprio) -> name-like
        if alpha.lower() in _SPEAKER_PARTICLES:
            continue  # nobiliary particle: de / van / von ...
        return False  # an all-lowercase content word -> this is a sentence
    return True


def _speaker_prefix(text: str) -> Optional[re.Match]:
    """Return the ``Name:`` prefix match only when it is a plausible speaker label."""
    m = _SPEAKER_PREFIX_RE.match(text)
    if m and _looks_like_speaker(m.group(1)):
        return m
    return None


#: A speaker change can occur MID-cue (parse_cues joins all of a cue's content lines into
#: one string), e.g. "...the question. ADDIS :When we go to retrieve...". The leading-only
#: _speaker_prefix never sees it, so the label leaks into claim text + answer snippets.
#: Two forms leak, and stripping each safely needs the document's CONFIRMED-speaker set
#: (collected by :func:`_collect_speakers`) so we never eat ordinary prose:
#:   * COLON form ("ADDIS:", "ADDIS :", "Rob Wiblin:") — captured by ``_MID_SPEAKER_RE``
#:     with group(2) holding any whitespace before the colon. A "Subject: predicate" prose
#:     appositive ("Steve Jobs: a visionary") is ALSO a `_looks_like_speaker` colon match,
#:     so we strip only when the name is a confirmed speaker OR uses the WebVTT "Name :"
#:     space-before-colon convention (which prose never does) — preserving the appositive.
#:   * BARE form ("ADDIS So, the fact...") — a cue/sentence-initial occurrence of a
#:     confirmed speaker with no colon at all. Gated strictly on the confirmed set so an
#:     arbitrary sentence-initial proper noun is never eaten.
_MID_SPEAKER_RE = re.compile(
    r"([A-Z][A-Za-z0-9.'’-]{0,40}(?:\s+[A-Z][A-Za-z0-9.'’-]{0,40}){0,2})(\s*):\s*")  # {0,40}: ReDoS-safe


def _collect_speakers(segments: List[Segment]) -> "frozenset[str]":
    """Casefolded set of names CONFIRMED as speakers somewhere in the document:
    a ``<v Name>`` tag, a leading ``Name:`` label, or a mid-cue ``Name :`` (space before
    the colon — the WebVTT convention that distinguishes a label from a prose colon)."""
    names: set = set()
    for seg in segments:
        if seg.speaker:
            names.add(seg.speaker.strip().casefold())
        raw = seg.text or ""
        m = _speaker_prefix(raw)
        if m:
            names.add(m.group(1).strip().casefold())
        for sm in _MID_SPEAKER_RE.finditer(raw):
            if sm.group(2) and _looks_like_speaker(sm.group(1)):  # space before colon
                names.add(sm.group(1).casefold())
    return frozenset(names)


@lru_cache(maxsize=128)
def _bare_speaker_re(speakers: "frozenset[str]") -> "Optional[re.Pattern]":
    """Match a cue/sentence-initial bare (no-colon) occurrence of a confirmed speaker,
    written in the ALLCAPS broadcast-label convention. The match is CASE-SENSITIVE on the
    uppercased name: prose writes "Reagan was the president" (Title case) but a caption
    speaker change writes "REAGAN ..." (ALLCAPS), so matching only ALLCAPS strips the
    leaked label without ever eating a sentence-initial CONTENT use of a name that happens
    to coincide with a speaker (Mark/Will/Hope/Reagan/Armstrong/...)."""
    if not speakers:
        return None
    alts = sorted({re.escape(n.upper()) for n in speakers}, key=len, reverse=True)
    return re.compile(r"(^|[.!?]\s+)(?:" + "|".join(alts) + r")\s+(?=\S)")


def _strip_mid_speaker_labels(text: str, speakers: "frozenset[str]" = frozenset()) -> str:
    def colon_repl(m: re.Match) -> str:
        name = m.group(1)
        if not _looks_like_speaker(name):
            return m.group(0)
        if m.group(2) or name.casefold() in speakers:  # "Name :" convention, or confirmed
            return " "
        return m.group(0)  # bare-colon prose appositive ("Steve Jobs:") -> keep the subject

    text = _MID_SPEAKER_RE.sub(colon_repl, text)
    bare = _bare_speaker_re(speakers)
    if bare is not None:
        text = bare.sub(lambda m: m.group(1), text)
    return text
#: ``>>`` (optionally ``>>>``) marks a speaker change in CEA-608 / broadcast-style
#: captions. We strip it from the knowledge text and use it to start a new
#: (anonymous) speaker turn so multi-speaker captions don't collapse onto spk_0.
_TURN_START_RE = re.compile(r"^\s*>>+")
_TURN_ANYWHERE_RE = re.compile(r">>+\s*")
#: Inline word-level timestamp tag, e.g. ``<00:00:01.199>`` — the fingerprint of
#: YouTube auto-generated "rolling" captions (Kind: captions). Their cues repeat
#: the previous line as a carried-over scroll-up line plus a new line carrying
#: these inline tags; the carry-over lines (and tiny 10ms scroll cues) hold no new
#: content and must be dropped or ~half the transcript duplicates.
_INLINE_TS_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{3}>")


def _parse_ts(value: str) -> float:
    s = value.strip().replace(",", ".")
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        h, m, sec = 0.0, nums[0], nums[1]
    else:
        h, m, sec = 0.0, 0.0, nums[0]
    # A huge all-digit field (e.g. a corrupt exporter) makes float() overflow to inf; an
    # inf/nan timestamp mis-sorts moments and crashes the deep-link / H:MM:SS formatters
    # downstream. Clamp to 0.0 so a malformed timing degrades rather than poisoning ingest
    # (mirrors the JSON path's _to_float guard).
    total = h * 3600 + m * 60 + sec
    return total if math.isfinite(total) else 0.0


def _is_timestamp_line(line: str) -> bool:
    """True iff ``line`` is a real cue timing line (``<ts> --> <ts>``, each side numeric).

    Guards against a malformed block whose only ``-->`` is an arrow inside speech: such
    a line would otherwise be accepted as the timestamp (``_parse_ts`` swallows the parse
    error to 0.0) and the real content before the arrow would be silently dropped."""
    left, _, right = line.partition("-->")
    right_ts = right.strip().split(" ")[0] if right.strip() else ""
    return bool(_TS_SIDE_RE.match(left.strip())) and bool(_TS_SIDE_RE.match(right_ts))


def parse_cues(text: str) -> List[Segment]:
    """Parse a VTT or SRT document into raw (uncleaned) segments.

    YouTube auto-generated "rolling" captions are de-duplicated: when the document
    carries inline word timestamps (``<00:00:01.199>``), each cue's carried-over
    scroll-up line is dropped and only the lines bearing inline timing (the newly
    revealed content) are kept. Without this, consecutive cues share ~half their
    text and downstream moment-merging triplicates it. A normal VTT/SRT (no inline
    timestamps) is parsed unchanged — every content line of every cue is kept.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    rolling = bool(_INLINE_TS_RE.search(text))
    # Rolling captions put a whitespace-only line *inside* a cue (an empty caption
    # row) and separate cues with a truly-blank line, so split only on truly-blank
    # lines — otherwise the intra-cue space-line orphans the new inline-timed line
    # from its timestamp. A normal transcript keeps the lenient ``\n\s*\n`` split.
    blocks = re.split(r"\n\n+" if rolling else r"\n\s*\n", text)
    segments: List[Segment] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        time_idx = next(
            (i for i, ln in enumerate(lines) if "-->" in ln and _is_timestamp_line(ln)),
            None,
        )
        if time_idx is None:
            continue
        left, _, right = lines[time_idx].partition("-->")
        start = _parse_ts(left)
        end = _parse_ts(right.strip().split(" ")[0])
        content_lines = lines[time_idx + 1:]
        if rolling:
            # Keep only lines that carry inline word timing (the new content);
            # pure carry-over lines (and the tiny 10ms scroll cues that hold only
            # carry-over) have none and are skipped — that is the dedup.
            content_lines = [ln for ln in content_lines if _INLINE_TS_RE.search(ln)]
            if not content_lines:
                continue
        raw = " ".join(content_lines).strip()
        if not raw:
            continue
        speaker = None
        m = _SPEAKER_VTT_RE.search(raw)
        if m:
            speaker = m.group(1).strip()
        segments.append(Segment(start=start, end=end, text=raw, speaker=speaker))
    return segments


# VTT and SRT share the same block parser.
parse_vtt = parse_cues
parse_srt = parse_cues


def _to_float(value, default: float = 0.0) -> float:
    """Coerce a JSON timing to float, tolerating null / list / junk so a malformed-but-
    valid export degrades instead of raising mid-ingest. Non-finite values (NaN / inf,
    including a huge literal like ``1e400`` that float() rounds to inf) also collapse to
    ``default``: an inf/nan timestamp otherwise mis-orders moments (every NaN comparison
    is False) and hard-crashes the deep-link / H:MM:SS formatters downstream. Mirrors the
    serving boundary's ``math.isfinite`` guard (routes._finite_float)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def parse_json(data) -> List[Segment]:
    """Parse a JSON transcript: a list of cues or ``{"segments": [...]}``.

    Tolerant of a syntactically-valid-but-wrong-shape document (a list of strings, a
    ``null`` item, non-numeric timings — all plausible third-party exports): non-dict
    items are skipped and bad timings coerce to 0.0, so an unexpected shape yields fewer
    (or zero) segments rather than an uncaught AttributeError/TypeError crashing ingest."""
    if isinstance(data, dict):
        data = data.get("segments", [])
    if not isinstance(data, list):
        return []
    segments: List[Segment] = []
    for item in data:
        if not isinstance(item, dict):
            continue  # a bare string / null / number is not a cue -> skip
        start = _to_float(item.get("start", item.get("t_start", 0.0)), 0.0)
        end = _to_float(item.get("end", item.get("t_end", start)), start)
        # Only a real string is content. A null/list/number "text" must be dropped, not
        # str()-coerced — str(None) -> the literal word "None", which would survive the
        # empty-guard and become a citable one-word speech claim.
        raw_text = item.get("text", "")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        if not text:
            continue
        # Optional per-word timings (M0.3): a free-path fixture can carry word
        # precision via a "words": [{"word","start","end"}, ...] array per cue.
        words = [
            Word(word=str(w.get("word", "")), start=_to_float(w.get("start"), 0.0),
                 end=_to_float(w.get("end"), 0.0))  # tolerates null/absent/junk timings
            for w in (item.get("words") or [])
            if isinstance(w, dict) and w.get("word")
        ]
        speaker = item.get("speaker")
        segments.append(Segment(start=start, end=end, text=text,
                                speaker=speaker if isinstance(speaker, str) else None,
                                words=words))
    return segments


def parse_plain(text: str, *, duration: Optional[float] = None) -> List[Segment]:
    """Parse untimed text into sentence segments with synthetic timestamps.

    Without real timing, citations can only preserve *order*; this is a
    last-resort path (real use should supply VTT/SRT or run ASR).
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    # Guard a non-finite duration (a corrupt probe/metadata value) so synthetic timings
    # stay finite — an inf/nan per-segment span crashes the deep-link / hms formatters.
    per = (duration / len(sentences)) if (duration and math.isfinite(duration)) else 3.0
    segments = []
    t = 0.0
    for s in sentences:
        segments.append(Segment(start=round(t, 3), end=round(t + per, 3), text=s))
        t += per
    return segments


def load_transcript(path: "str | Path", *, duration: Optional[float] = None) -> List[Segment]:
    """Load + parse a transcript file, dispatching on extension/content."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()
    if ext == ".vtt" or raw.lstrip().upper().startswith("WEBVTT"):
        return parse_cues(raw)
    if ext == ".srt":
        return parse_cues(raw)
    if ext == ".json":
        try:
            return parse_json(json.loads(raw))
        except ValueError:
            return parse_plain(raw, duration=duration)
    if "-->" in raw:  # unlabeled but looks like cues
        return parse_cues(raw)
    return parse_plain(raw, duration=duration)


def _decode_entities(raw: str) -> str:
    """Strip markup tags, THEN decode HTML entities. Tag-stripping first keeps a decoded
    ``&lt;``/``&gt;`` from being re-read as a tag; decoding before the event/bracket/music
    passes means an entity-encoded annotation (YouTube/WebVTT emit ``&#91;applause&#93;``
    and ``&#9834;``) is recognized rather than surviving still-escaped into claim text."""
    return html.unescape(_TAG_RE.sub(" ", raw or ""))


def clean_text(raw: str, speakers: "frozenset[str]" = frozenset()) -> Tuple[str, List[str]]:
    """Return (cleaned speech text, list of audio-event keywords).

    ``speakers`` is the document's confirmed-speaker set (see :func:`_collect_speakers`),
    used to strip mid-cue/bare interior speaker labels without eating prose appositives.
    """
    # Tags off + entities decoded UP FRONT (see _decode_entities) so every strip pass below
    # operates on real glyphs; the trailing ``\s+`` collapse folds any U+00A0 (&nbsp;) out.
    raw = _decode_entities(raw)
    events = [e.lower() for e in EVENT_RE.findall(raw)]
    stripped = EVENT_RE.sub(" ", raw)
    # Markdown links first (keep label, drop URL), then any orphaned ](url)/(url) — both
    # BEFORE residual-bracket stripping so a [label](url) is not deleted as an annotation.
    stripped = _MD_LINK_RE.sub(r"\1", stripped)
    stripped = _URL_PAREN_RE.sub(" ", stripped)

    def _strip_bracket(m: re.Match) -> str:
        inner = m.group(0)[1:-1].strip()
        if _BRACKET_CENSOR_RE.match(inner):
            return " "  # censored profanity "[ __ ]" -> drop, not a timeline event
        if _BRACKET_WORD_RE.search(inner):
            events.append(inner.lower()[:40])  # non-speech annotation -> timeline marker
            return " "
        return m.group(0)  # code/math like [i], [b,t], [0] -> keep

    stripped = _RESIDUAL_BRACKET_RE.sub(_strip_bracket, stripped)
    stripped = _TURN_ANYWHERE_RE.sub(" ", stripped)  # drop ">>" turn markers
    m = _speaker_prefix(stripped)
    if m:
        stripped = stripped[m.end():]
    stripped = _strip_mid_speaker_labels(stripped, speakers)  # interior speaker changes
    words = [w for w in stripped.split() if _normalize_word(w) not in FILLERS]
    text = re.sub(r"\s+", " ", " ".join(words)).strip()
    return text, events


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def clean_segments(segments: List[Segment]) -> List[Segment]:
    """Strip fillers/events; emit cleaned speech segments + event markers.

    ``>>`` turn markers (when present) start a new anonymous speaker that PERSISTS
    onto following unmarked lines, so a dialogue surfaces as multiple speakers
    instead of a single ``spk_0``. A transcript with no markers/names is untouched
    (speaker stays None -> :func:`assign_speakers` applies the single-speaker
    default), so the free auto-caption path is unchanged."""
    out: List[Segment] = []
    # First pass: collect every name confirmed as a speaker anywhere in the document, so
    # the per-segment cleaner can strip a bare/no-space label that is confirmed elsewhere
    # (e.g. "ADDIS So, the fact...") without eating a prose appositive ("Steve Jobs:").
    speakers = _collect_speakers(segments)
    current: Optional[str] = None  # persistent speaker across ">>"-delimited turns
    anon_idx = 0
    saw_turn = False
    # Case-insensitive canonicalization of named speakers WITHIN a video: captions
    # often spell the same person two ways ("YANJAA" in a label, "Yanjaa" in a <v> tag),
    # which otherwise splits into two speaker ids with duplicated turns. Collapse to the
    # first-seen casing.
    speaker_canon: dict = {}

    def _canon(name: str) -> str:
        return speaker_canon.setdefault(name.casefold(), name)

    for seg in segments:
        # A musical-note-marked line is song/lyrics, not spoken content — record it
        # as a music event and emit no speech (so it never becomes a claim).
        if seg.kind != "event" and _MUSIC_NOTE_RE.search(_decode_entities(seg.text)):
            out.append(Segment(start=seg.start, end=seg.start, text="[music]", kind="event"))
            continue
        text, events = clean_text(seg.text, speakers)
        speaker = seg.speaker
        if not speaker:
            m = _speaker_prefix(seg.text)
            if m:
                speaker = m.group(1).strip()
        if speaker:
            speaker = _canon(speaker)
            current = speaker
        elif _TURN_START_RE.match(seg.text or ""):
            # A ">>" with no name starts a new anonymous turn (the first one keeps
            # spk_0 so a single-speaker ">>"-prefixed caption isn't bumped to spk_1).
            if current is None:
                current = "spk_0"
            else:
                anon_idx += 1
                current = f"spk_{anon_idx}"
            saw_turn = True
        # Only propagate a tracked speaker once turns/names have actually appeared;
        # otherwise leave None for assign_speakers' default (no behavior change).
        speaker = current if (speaker or saw_turn) else None
        if text:
            out.append(
                Segment(
                    start=seg.start, end=max(seg.end, seg.start), text=text,
                    speaker=speaker, words=seg.words, kind="speech",
                )
            )
        for ev in events:
            out.append(Segment(start=seg.start, end=seg.start, text=f"[{ev}]", kind="event"))
    out.sort(key=lambda s: (s.start, 0 if s.kind == "speech" else 1))
    return out
