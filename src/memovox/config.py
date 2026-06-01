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

    embed_dim: int = 256  # dimensionality of the hashing fallback embedder

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
