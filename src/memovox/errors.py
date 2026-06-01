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
