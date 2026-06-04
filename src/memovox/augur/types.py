"""Answer + Citation value types for the query layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Citation:
    index: int
    video_id: str
    moment_id: str
    t_start_s: float
    t_end_s: float
    modality: str = "speech"
    speaker: Optional[str] = None
    title: Optional[str] = None
    deep_link: Optional[str] = None
    snippet: str = ""
    score: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Clip:
    """A stitched, deep-linked watch window over one or more cited spans (M2.3)."""

    video_id: str
    t_start_s: float
    t_end_s: float
    title: Optional[str] = None
    deep_link: Optional[str] = None  # RANGED (watch?v=…&start=&end=)
    citation_indices: List[int] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "duration_s": self.duration_s,
            "title": self.title,
            "deep_link": self.deep_link,
            "citation_indices": list(self.citation_indices),
        }


@dataclass
class Answer:
    text: str
    citations: List[Citation] = field(default_factory=list)
    strategy: str = "hybrid"
    low_evidence: bool = False
    metrics: dict = field(default_factory=dict)  # M0.1 per-stage trace (volatile wall_ms)
    plan: List[dict] = field(default_factory=list)  # M2.2 decomposed sub-queries
    clips: List[Clip] = field(default_factory=list)  # M2.3 stitched clip windows

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "strategy": self.strategy,
            "low_evidence": self.low_evidence,
            "citations": [c.to_dict() for c in self.citations],
            "metrics": self.metrics,
            "plan": self.plan,
            "clips": [c.to_dict() for c in self.clips],
        }
