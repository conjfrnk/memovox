"""Exception hierarchy for memovox."""

from __future__ import annotations


class MemovoxError(Exception):
    """Base class for all memovox errors."""


class ConfigError(MemovoxError):
    """Invalid configuration."""


class AcquisitionError(MemovoxError):
    """Failed to acquire a source (download / locate media)."""


class DemuxError(MemovoxError):
    """ffmpeg/ffprobe demux or validation failed."""


class BackendUnavailable(MemovoxError):
    """A requested model backend's dependencies are not installed."""


class IngestionError(MemovoxError):
    """A pipeline stage failed irrecoverably."""


class NotFoundError(MemovoxError):
    """A requested entity does not exist."""


class BudgetExceeded(MemovoxError):
    """A per-video token/compute budget was exceeded in hard mode (spec §9)."""


class DevicePlacementError(MemovoxError):
    """A heavy ASR model would silently run on CPU (10-50x slower, spec §9).

    Escape via ``--allow-cpu`` / ``MEMOVOX_ASR_ALLOW_CPU=1`` / ``asr_allow_cpu``.
    """


class VectorSpaceError(MemovoxError):
    """A vector search crossed embedding spaces (e.g. text query vs visual signature).

    Text and visual signatures can be the same length by coincidence; cosining
    across spaces is meaningless, so a mismatch raises rather than silently scoring.
    """
