"""memovox — a multimodal video-to-knowledge engine.

Voice in, queryable and *cited* memory out. memovox ingests video at the level
of meaning — fusing speech, on-screen content, and speakers onto a single
timeline — and distills it into a verified, provenance-stamped knowledge base
you can query and synthesize across.

The core is local-first and runs on the Python standard library alone: every
model slot (ASR, embedder, NLI, LLM) has a deterministic fallback, with real
backends (faster-whisper, sentence-transformers, DeBERTa-NLI, Ollama, yt-dlp)
available as optional, swappable upgrades.

Subsystem codenames (spec §3):
    Stentor    acquisition + ASR + diarization        (memovox.stentor)
    Tessera    visual track (keyframes, OCR, VLM)      (memovox.tessera)
    Escapement temporal fusion into Moments           (memovox.escapement)
    Assay      claim extraction + NLI verification     (memovox.assay)
    Loom       indices + knowledge graph + synthesis   (memovox.loom)
    Augur      agentic retrieval + cited answers       (memovox.augur)
"""

__version__ = "0.1.0"

__all__ = ["Memovox", "Config", "__version__"]


def __getattr__(name):
    # Lazy imports keep `import memovox` cheap and avoid import cycles during
    # partial builds / optional-dependency probing.
    if name == "Memovox":
        from .sdk import Memovox

        return Memovox
    if name == "Config":
        from .config import Config

        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
