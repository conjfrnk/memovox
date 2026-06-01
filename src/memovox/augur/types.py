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
class Answer:
    text: str
    citations: List[Citation] = field(default_factory=list)
    strategy: str = "hybrid"
    low_evidence: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "strategy": self.strategy,
            "low_evidence": self.low_evidence,
            "citations": [c.to_dict() for c in self.citations],
        }
