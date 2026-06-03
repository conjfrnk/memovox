"""Optional voiceprint backend for voice-based speaker resolution (W4.2, §12).

The FREE path never touches this: cross-video speakers are merged by NAME (W4.1,
:func:`memovox.loom.resolve.resolve_speakers`), and anonymous diarization labels
stay unmerged. This backend is an OPTIONAL upgrade — when ``pyannote.audio`` is
installed it extracts a per-speaker embedding (a "voiceprint") from a span of
audio, so anonymous same-voice speakers can be clustered conservatively above a
cosine threshold (the clustering itself lives in ``resolve.py`` as a pure,
dependency-free function — only the embedding model needs pyannote).

``is_available()`` guards every use, and pyannote is lazy-imported inside
:meth:`embed`, so importing this module on the free path pulls in nothing beyond
the standard library.
"""

from __future__ import annotations

import importlib.util
from typing import List, Optional

from .base import Backend


class PyannoteVoiceprint(Backend):
    """Speaker-embedding extractor backed by ``pyannote.audio`` (optional).

    Never imports pyannote unless :meth:`embed` is actually called, and only
    after :meth:`is_available` has confirmed the package is installed.
    """

    name = "pyannote"

    def __init__(self, config=None, model: Optional[str] = None, **options) -> None:
        super().__init__(config, **options)
        #: Pretrained speaker-embedding model id (override via options/env if needed).
        self.model = model or "pyannote/embedding"
        self._inference = None  # lazily-constructed pyannote Inference

    @classmethod
    def is_available(cls) -> bool:
        # find_spec raises (not returns None) when a *parent* package is missing,
        # so guard exactly like stentor.diarize.pyannote_available().
        try:
            return importlib.util.find_spec("pyannote.audio") is not None
        except (ImportError, ValueError):
            return False

    def _load(self):  # pragma: no cover - requires pyannote + model weights
        if self._inference is None:
            from pyannote.audio import Inference, Model

            model = Model.from_pretrained(self.model)
            self._inference = Inference(model, window="whole")
        return self._inference

    def embed(self, audio_path: str, t_start: float, t_end: float) -> List[float]:
        """Return a speaker-embedding vector for ``audio_path[t_start:t_end]``.

        Only runs under pyannote (guarded by :meth:`is_available`); raises
        :class:`RuntimeError` if the backend is somehow used while unavailable so
        the failure is loud rather than a silent free-path import.
        """
        if not self.is_available():  # pragma: no cover - defensive
            raise RuntimeError(
                "PyannoteVoiceprint.embed called but pyannote.audio is not installed; "
                "install with: pip install 'memovox[diarize]'"
            )
        # Lazy imports — never executed on the free path.
        from pyannote.core import Segment  # pragma: no cover - needs pyannote

        inference = self._load()  # pragma: no cover - needs pyannote
        excerpt = inference.crop(audio_path, Segment(t_start, t_end))  # pragma: no cover
        # pyannote returns a numpy array; coerce to a plain Python float list so
        # the rest of the stack stays numpy-free.
        return [float(x) for x in excerpt.reshape(-1)]  # pragma: no cover
