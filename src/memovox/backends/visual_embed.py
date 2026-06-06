"""Visual embedders (Tessera, spec §4.3): the free signature + opt-in ColPali.

``SignatureVisualEmbedder`` is the always-available free path — it reuses the
grayscale signature (``bytes_to_signature``) that Tessera already computes, so a
keyframe becomes a retrievable visual vector with no model and no new dependency.
``ColPaliVisualEmbedder`` is an opt-in multi-vector (MaxSim) upgrade, gated to
slide/doc/diagram frames; it reports unavailable (→ ``BackendUnavailable``) on a
bare machine, exactly like the other optional backends.
"""

from __future__ import annotations

import importlib.util
from typing import List

from .base import VisualEmbedder


class SignatureVisualEmbedder(VisualEmbedder):
    name = "signature"
    space = "visual_sig"

    def __init__(self, config=None, **options) -> None:
        super().__init__(config, **options)
        self.side = int(options.get("side", 16))
        self.dim = self.side * self.side

    @classmethod
    def is_available(cls) -> bool:
        return True

    def embed_image(self, image) -> List[float]:
        """A precomputed signature (list of floats in [0,1]) passes through; raw
        8-bit grayscale ``bytes`` are converted via the free signature function."""
        if isinstance(image, (bytes, bytearray)):
            from ..tessera.frames import bytes_to_signature

            return bytes_to_signature(bytes(image))
        return list(image)


class ColPaliVisualEmbedder(VisualEmbedder):
    """Opt-in ColPali multi-vector late-interaction embedder (skeleton, M1.1 W4)."""

    name = "colpali"
    space = "colpali"

    @classmethod
    def is_available(cls) -> bool:
        # Unimplemented skeleton (embed_image() raises NotImplementedError): report
        # UNAVAILABLE so explicit selection fails clean at the factory rather than
        # crashing mid-ingest. Restore find_spec("colpali_engine") when wired.
        return False

    def embed_image(self, image) -> List[float]:  # pragma: no cover - needs colpali_engine
        raise NotImplementedError(
            "ColPaliVisualEmbedder is a skeleton; wire colpali_engine page embeddings "
            "+ MaxSim scoring here (gated to slide/doc/diagram frames; benchmarked in M3.4)."
        )
