"""Configuration: the on-disk knowledge store and tunable settings.

A memovox *store* is a single directory (default ``~/.memovox``) holding one
SQLite database plus media, frames, and human-readable Markdown digests::

    <store>/
        memovox.db          relational + lexical(FTS5) + vectors + graph
        media/              demuxed audio, downloaded video, source files
        frames/             extracted keyframes (Tessera)
        digests/            per-video Markdown digests (human-readable substrate)
        models/             local model cache (HF, Vosk, ...)
        subscriptions.json  channels/playlists for `memovox sync`
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PIPELINE_VERSION = "0.1.0-phase0"
ENV_STORE = "MEMOVOX_STORE"


@dataclass
class Settings:
    """Tunable knobs for the pipeline and retrieval (spec §4, §5)."""

    # backend selection ("auto" picks the best available, else a free fallback)
    asr_backend: str = "auto"
    embed_backend: str = "auto"
    nli_backend: str = "auto"
    llm_backend: str = "auto"
    vlm_backend: str = "auto"
    ocr_backend: str = "auto"
    entity_backend: str = "auto"
    # OPTIONAL voiceprint backend (W4.2). "auto" uses pyannote ONLY if installed,
    # else None (free path); "none" disables the voice-based speaker merge.
    voiceprint_backend: str = "auto"

    # ASR device placement (M0.3) — fail loud if a heavy model lands on CPU.
    asr_device: str = "auto"            # auto | cpu | cuda
    asr_compute_type: str = "default"   # faster-whisper compute_type
    asr_allow_cpu: bool = False         # escape hatch for the DevicePlacementError guard
    captions_as_prior: bool = True      # §9 cost lever: captions win over Whisper on auto

    embed_dim: int = 256  # dimensionality of the hashing fallback embedder

    # Tessera — visual track (spec §4 stage 3)
    visual_enabled: bool = True
    frame_sample_fps: float = 1.0       # candidate frames sampled per second
    frame_side: int = 16                # downscaled signature is frame_side**2 long
    frame_max: int = 1200               # hard cap on sampled frames per video
    scene_threshold: float = 0.3        # frame-diff above this => scene/slide cut
    keyframe_min_gain: float = 0.12     # info-gain floor to keep another keyframe
    keyframe_per_scene_cap: int = 8     # max keyframes kept per scene
    visual_embed_backend: str = "signature"  # M1.1: visual embedder (signature/colpali)
    visual_workers: int = 1             # M1.1: per-keyframe OCR/VLM pool (1 = serial/deterministic)

    # Escapement — Moment boundaries
    moment_max_sec: float = 90.0
    moment_min_sec: float = 8.0
    moment_gap_sec: float = 2.5
    boundary_similarity: float = 0.45  # below this similarity => topic boundary

    # Assay — verification gate
    entailment_threshold: float = 0.5
    salience_floor: float = 0.0

    # Augur — retrieval
    rrf_k: int = 60
    top_k: int = 8
    contradiction_threshold: float = 0.55
    rerank_backend: str = "auto"  # M2.1: "auto" -> cross-encoder iff installed, else identity
    vector_prefilter_fts: bool = False  # M0.2: restrict vector candidates to FTS hits (opt-in)
    visual_retrieval: bool = False      # M1.1: master switch for the VISUAL retrieval leg in ask()

    # Loom — synthesis (Phase 3, spec §4.7)
    topic_similarity: float = 0.5   # cosine floor to merge moments into one topic
    topic_min_size: int = 1         # drop induced topics with fewer members
    consensus_jaccard: float = 0.5  # content-token Jaccard floor to cluster equivalent claims

    # Observability (M0.1, spec §7/§9) — all default to the free, no-output-change path
    budget_mode: str = "soft"       # "soft" records overage; "hard" raises BudgetExceeded
    otel_enabled: bool = False      # mirror spans to OpenTelemetry (opt-in [otel] extra)

    @classmethod
    def from_env(cls, base: "Settings | None" = None) -> "Settings":
        data = asdict(base) if base else {}
        valid = {f.name: f.type for f in fields(cls)}
        for name in valid:
            env_key = f"MEMOVOX_{name.upper()}"
            if env_key in os.environ:
                raw = os.environ[env_key]
                data[name] = _coerce(getattr(cls, name), raw)
        return cls(**data)

    def merged(self, overrides: dict) -> "Settings":
        data = asdict(self)
        for k, v in (overrides or {}).items():
            if k in data and v is not None:
                data[k] = v
        return Settings(**data)


def _coerce(default, raw: str):
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


class Config:
    """Resolved paths and settings for one memovox store."""

    def __init__(self, store=None, settings: "Settings | None" = None) -> None:
        raw = store or os.environ.get(ENV_STORE) or "~/.memovox"
        self.store = Path(raw).expanduser()
        # Layer: defaults < store/config.json < env < explicit settings arg.
        base = self._load_file_settings()
        base = Settings.from_env(base)
        self.settings = settings or base

    def _load_file_settings(self) -> Settings:
        cfg = self.store / "config.json"
        if cfg.is_file():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
                return Settings().merged(data)
            except (ValueError, OSError):
                pass
        return Settings()

    # -- paths -------------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self.store / "memovox.db"

    @property
    def media_dir(self) -> Path:
        return self.store / "media"

    @property
    def frames_dir(self) -> Path:
        return self.store / "frames"

    @property
    def digests_dir(self) -> Path:
        return self.store / "digests"

    @property
    def models_dir(self) -> Path:
        return self.store / "models"

    @property
    def subscriptions_path(self) -> Path:
        return self.store / "subscriptions.json"

    def ensure(self) -> "Config":
        for d in (self.store, self.media_dir, self.frames_dir, self.digests_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(store={self.store!r})"
