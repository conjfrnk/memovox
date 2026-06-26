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
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .errors import ConfigError

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
    # W5.1: refuse confident answers when the cited evidence does not cover the
    # query's distinctive (IDF-weighted) terms — the "no citation, no claim" promise.
    # Coverage in [0,1]; below this floor the answer is flagged low_evidence and the
    # (irrelevant) citations are withheld. 0.0 disables the gate.
    answer_relevance_floor: float = 0.55
    # The IDF coverage signal needs a corpus large enough to tell a rare topic term
    # from a generic term that is merely absent from a tiny store. Below this many
    # moments the gate is disabled (any real single video already has far more).
    answer_relevance_min_moments: int = 50
    planner_agentic: bool = False  # M2.2: use the LLM query decomposer (opt-in; deterministic default)
    clip_merge_gap_s: float = 2.5  # M2.3: merge cited spans <= this gap into one stitched clip
    vector_prefilter_fts: bool = False  # M0.2: restrict vector candidates to FTS hits (opt-in)
    visual_retrieval: bool = False      # M1.1: master switch for the VISUAL retrieval leg in ask()
    decay_enabled: bool = False         # M3.1: recency re-weight + superseded demotion (opt-in)
    decay_halflife_days: float = 365.0  # M3.1: recency half-life (reuses the consensus model)

    # Loom — synthesis (Phase 3, spec §4.7)
    topic_similarity: float = 0.5   # cosine floor to merge moments into one topic
    topic_min_size: int = 1         # drop induced topics with fewer members
    consensus_jaccard: float = 0.5  # content-token Jaccard floor to cluster equivalent claims
    # W5.6: opt-in embedding-cosine fallback for consensus clustering — groups
    # paraphrases/synonyms that token-Jaccard misses. 0.0 = off (free path default);
    # a no-op with the lexical hashing embedder, meaningful with sentence-transformers.
    consensus_cosine: float = 0.0

    # Observability (M0.1, spec §7/§9) — all default to the free, no-output-change path
    budget_mode: str = "soft"       # "soft" records overage; "hard" raises BudgetExceeded
    otel_enabled: bool = False      # mirror spans to OpenTelemetry (opt-in [otel] extra)

    # Serving / privacy (M3.3, §12) — private-by-default posture
    local_only: bool = False        # refuse all network acquisition (air-gapped use)

    @classmethod
    def from_env(cls, base: "Settings | None" = None) -> "Settings":
        data = asdict(base) if base else {}
        valid = {f.name: f.type for f in fields(cls)}
        for name in valid:
            env_key = f"MEMOVOX_{name.upper()}"
            if env_key in os.environ:
                data[name] = _coerce_value(getattr(cls, name), os.environ[env_key], name)
        return cls(**data)

    def merged(self, overrides: dict) -> "Settings":
        # Coerce EVERY override to the field's declared type — applies to both the config.json
        # layer (untyped native JSON values) and the SDK/CLI kwargs layer. Without this, a JSON
        # string "false" for a bool field stayed a non-empty (truthy) string — silently flipping
        # a privacy/safety flag (local_only, asr_allow_cpu) — and a JSON string "8" for an int
        # field crashed the query path later. (env values already coerce via from_env; this
        # closes the parallel gap the config.py docstring warns about.)
        data = asdict(self)
        for k, v in (overrides or {}).items():
            if k in data and v is not None:
                data[k] = _coerce_value(getattr(Settings, k), v, k)
        return Settings(**data)


def _coerce_value(default, value, name: str):
    """Coerce a config override (an env STRING or a native JSON value) to ``default``'s type.

    Raises :class:`ConfigError` (a MemovoxError, so the CLI surfaces a clean ``error: ...``
    instead of a raw traceback) on a value that cannot be coerced or is non-finite — closing
    two gaps: a bad env value (``MEMOVOX_TOP_K=abc``) used to crash Config construction with a
    raw ValueError, and a non-finite float (``MEMOVOX_ANSWER_RELEVANCE_FLOOR=nan``) used to
    sail through and silently DISABLE the out-of-corpus refusal gate (NaN defeats ``floor>0``
    and ``relevance<floor``)."""
    # bool FIRST: bool is a subclass of int, so the int branch below would otherwise swallow it.
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(value, (int, float)):
            return bool(value)
        raise ConfigError(f"setting {name!r}: cannot interpret {value!r} as a boolean")
    if isinstance(default, int):
        try:
            return int(value)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"setting {name!r}: {value!r} is not an integer ({exc})") from exc
    if isinstance(default, float):
        try:
            f = float(value)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"setting {name!r}: {value!r} is not a number ({exc})") from exc
        if not math.isfinite(f):
            raise ConfigError(f"setting {name!r}: {value!r} is not a finite number")
        return f
    # str-typed field: accept a string as-is; stringify a stray native scalar (e.g. a JSON
    # number given for a backend-name field) so the dataclass still holds the declared type.
    return value if isinstance(value, str) else str(value)


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
            except (ValueError, OSError) as exc:
                # A malformed config.json silently reverting to defaults can flip
                # privacy-relevant settings (e.g. local_only back to False). Warn on
                # stderr so the operator notices instead of running with surprise
                # defaults. (stderr, not stdout — MCP owns stdout.)
                import sys
                print(f"memovox: ignoring malformed {cfg} ({exc}); using defaults.",
                      file=sys.stderr)
                return Settings()
            if not isinstance(data, dict):
                import sys
                print(f"memovox: {cfg} is not a JSON object; using defaults.",
                      file=sys.stderr)
                return Settings()
            return Settings().merged(data)
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
        # MEMOVOX_MODELS_DIR separates the (multi-GB, reusable) model cache from the data
        # store, so wiping/replacing a store doesn't force a re-download. Default unchanged.
        override = os.environ.get("MEMOVOX_MODELS_DIR")
        return Path(override).expanduser() if override else self.store / "models"

    @property
    def subscriptions_path(self) -> Path:
        return self.store / "subscriptions.json"

    def ensure(self) -> "Config":
        for d in (self.store, self.media_dir, self.frames_dir, self.digests_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(store={self.store!r})"
