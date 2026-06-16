"""Transcript parsing and cleaning (Stentor).

Parses VTT / SRT / JSON / plain-text transcripts into :class:`Segment` objects,
strips filler tokens and bracketed audio events from the knowledge text while
retaining the events as timeline markers (spec §4 stage 2).
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..backends.base import Segment, Word
from ..util import split_sentences

FILLERS = {"um", "uh", "erm", "uhh", "umm", "mm", "hmm", "mhm", "uhm", "ah", "er"}
EVENT_RE = re.compile(
    r"\[\s*(music|applause|laughs?|laughter|laughing|silence|inaudible|noise|"
    r"crosstalk|cheering|chuckles?|singing|humming|instrumental|sighs?|groans?|"
    r"coughs?|gasps?|clears throat|beep|static|whistling|ticking|sizzling|"
    r"ringing|rings|dings?|bells?|footsteps|wind|rain|thunder)[^\]]*\]",
    re.IGNORECASE,
)
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
_MD_LINK_RE = re.compile(r"\[([^\]\n]{0,200})\]\((https?://[^)\s]+)\)")
_URL_PAREN_RE = re.compile(r"\]?\((https?://[^)\s]+)\)")
#: Musical-note markers wrap sung lyrics in captions. A line carrying one is music
#: (a song), NOT spoken video content, so it becomes a timeline event rather than a
#: claim. Unambiguous: these glyphs appear only for music, so there is no false
#: positive. (Sung lyrics transcribed as PLAIN text with no marker -- e.g. some
#: YouTube auto-captions -- can't be told apart from speech without audio.)
_MUSIC_NOTE_RE = re.compile(r"[♪♫♬\U0001f3b5\U0001f3b6]")
_TAG_RE = re.compile(r"<[^>]+>")
_SPEAKER_VTT_RE = re.compile(r"<v\s+([^>]+)>")
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
    return h * 3600 + m * 60 + sec


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
        time_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if time_idx is None:
            continue
        try:
            left, right = lines[time_idx].split("-->")
            start = _parse_ts(left)
            end = _parse_ts(right.strip().split(" ")[0])
        except ValueError:
            continue
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


def parse_json(data) -> List[Segment]:
    """Parse a JSON transcript: a list of cues or ``{"segments": [...]}``."""
    if isinstance(data, dict):
        data = data.get("segments", [])
    segments: List[Segment] = []
    for item in data or []:
        start = float(item.get("start", item.get("t_start", 0.0)) or 0.0)
        end = float(item.get("end", item.get("t_end", start)) or start)
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        # Optional per-word timings (M0.3): a free-path fixture can carry word
        # precision via a "words": [{"word","start","end"}, ...] array per cue.
        words = [
            Word(word=str(w.get("word", "")), start=float(w.get("start") or 0.0),
                 end=float(w.get("end") or 0.0))  # `or 0.0` tolerates null/absent timings
            for w in (item.get("words") or [])
            if w.get("word")
        ]
        segments.append(Segment(start=start, end=end, text=text,
                                speaker=item.get("speaker"), words=words))
    return segments


def parse_plain(text: str, *, duration: Optional[float] = None) -> List[Segment]:
    """Parse untimed text into sentence segments with synthetic timestamps.

    Without real timing, citations can only preserve *order*; this is a
    last-resort path (real use should supply VTT/SRT or run ASR).
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    per = (duration / len(sentences)) if duration else 3.0
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


def clean_text(raw: str) -> Tuple[str, List[str]]:
    """Return (cleaned speech text, list of audio-event keywords)."""
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
    stripped = _TAG_RE.sub(" ", stripped)
    # Decode HTML entities AFTER tag stripping (so a decoded ``&lt;`` is not re-read
    # as a tag): WebVTT escapes ``&`` ``<`` ``>`` and emits ``&nbsp;``/``&#39;`` —
    # left raw, these survive into claim text ("By 1920,&nbsp;&nbsp;"). The trailing
    # ``\s+`` collapse folds the resulting U+00A0 into a normal space.
    stripped = html.unescape(stripped)
    stripped = _TURN_ANYWHERE_RE.sub(" ", stripped)  # drop ">>" turn markers
    m = _speaker_prefix(stripped)
    if m:
        stripped = stripped[m.end():]
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
        if seg.kind != "event" and _MUSIC_NOTE_RE.search(seg.text or ""):
            out.append(Segment(start=seg.start, end=seg.start, text="[music]", kind="event"))
            continue
        text, events = clean_text(seg.text)
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
