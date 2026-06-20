"""Core data model (spec §6): Video, Moment, Claim, Entity, Speaker, Topic.

Plus the Provenance object attached to every retrievable fact — the heart of the
"provenance is sacred" principle (spec §2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional

from ..util import deep_link


class SegmentRef(NamedTuple):
    """A source-segment span a Moment was fused from (W1, spec §6).

    A build-time artifact retained on each Moment so later stages (Assay) can
    bind a claim to its EXACT source span. ``words`` (M0.3) optionally carries the
    per-word ``(start, end, word)`` timings ASR emitted; it is the LAST field with
    an empty-tuple default, so positional ``t0, t1, text = ref[:3]`` /
    ``(t0, t1, text, *_) = ref`` unpacking stays valid and the free (captions) path
    carries ``words == ()``.
    """

    t_start_s: float
    t_end_s: float
    text: str
    words: tuple = ()

# Claim epistemic types (spec §5).
CLAIM_TYPES = (
    "FACT", "DEFINITION", "OPINION", "PROCEDURE", "EXAMPLE", "PREDICTION", "CORRECTION",
)
# Claim lifecycle status.
STATUS_COMMITTED = "committed"
STATUS_UNSUPPORTED = "unsupported"
STATUS_SUPERSEDED = "superseded"


@dataclass
class Video:
    video_id: str
    source_url: Optional[str]
    title: str
    channel: Optional[str] = None
    published_at: Optional[str] = None
    duration_s: Optional[float] = None
    lang: Optional[str] = None
    content_hash: Optional[str] = None
    ingested_at: Optional[str] = None
    pipeline_version: Optional[str] = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        # Coerce a non-finite duration (inf/nan — e.g. from an old pre-clamp ingest) to None
        # so json.dumps never emits invalid `Infinity`/`NaN` that strict clients reject.
        dur = d.get("duration_s")
        if dur is not None and not math.isfinite(dur):
            d["duration_s"] = None
        return d


@dataclass
class Moment:
    moment_id: str
    video_id: str
    t_start_s: float
    t_end_s: float
    transcript: str
    speaker_id: Optional[str] = None
    visual_caption: Optional[str] = None
    ocr_text: Optional[str] = None
    topic_id: Optional[str] = None
    index: int = 0
    #: Source-segment spans this Moment was fused from. A build-time artifact
    #: consumed by Assay in the same pipeline run; NOT persisted to the store
    #: (Moments reloaded from the store have ``segments == []``).
    segments: List[SegmentRef] = field(default_factory=list)

    @property
    def modality(self) -> str:
        mods = ["speech"]
        if self.ocr_text or self.visual_caption:
            mods.append("slide")
        return "+".join(mods)

    def text_for_embedding(self) -> str:
        # Speech transcript + literal on-screen OCR text are answerable CONTENT and
        # belong in the searchable text embedding. The VLM's prose DESCRIPTION of
        # the frame (visual_caption) is intentionally excluded: it is a retrieval
        # aid, and a verbose caption ("The image shows a man wearing sunglasses…")
        # otherwise dominates the embedding and buries the transcript for text
        # queries. The caption still drives the separate visual embedding
        # (visual_vectors) and the human-readable digest.
        parts = [self.transcript]
        if self.ocr_text:
            parts.append(self.ocr_text)
        return "\n".join(p for p in parts if p).strip()

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["modality"] = self.modality
        return d


@dataclass
class Claim:
    claim_id: str
    moment_id: str
    video_id: str
    text: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    claim_type: str = "FACT"
    salience: float = 0.0
    entailment_score: float = 0.0
    status: str = STATUS_COMMITTED
    superseded_by: Optional[str] = None
    t_start_s: float = 0.0
    t_end_s: float = 0.0
    speaker_id: Optional[str] = None
    qualifiers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Speaker:
    speaker_id: str
    label: str
    voiceprint_ref: Optional[str] = None
    resolved_name: Optional[str] = None
    #: Cross-video canonical identity (``spk:<slug>``), set by
    #: :func:`memovox.loom.resolve.resolve_speakers`. ``None`` for an
    #: unresolved / self-canonical speaker.
    canonical_id: Optional[str] = None


@dataclass
class Entity:
    entity_id: str
    canonical_name: str
    type: str = "concept"
    wikidata_qid: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class Topic:
    topic_id: str
    label: str
    moment_count: int = 0


@dataclass
class Provenance:
    video_id: str
    t_start_s: float
    t_end_s: float
    modality: str = "speech"
    speaker: Optional[str] = None
    confidence: float = 1.0
    deep_link: Optional[str] = None

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def make_provenance(
    video: Video,
    t_start_s: float,
    t_end_s: float,
    *,
    modality: str = "speech",
    speaker: Optional[str] = None,
    confidence: float = 1.0,
) -> Provenance:
    return Provenance(
        video_id=video.video_id,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        modality=modality,
        speaker=speaker,
        confidence=confidence,
        deep_link=deep_link(video.source_url, t_start_s),
    )
