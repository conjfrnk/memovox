"""Offline ASR via faster-whisper (Stentor stage 2 upgrade).

Requires ``faster-whisper`` (``pip install "memovox[asr]"``). Model weights
download once from a public mirror on first use — free, no API key. Produces
word-level timestamps for precise citation.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from .base import ASRBackend, ASRResult, Segment, Word

DEFAULT_MODEL = "large-v3"


class WhisperASR(ASRBackend):
    name = "whisper"
    _model_cache: dict = {}

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def _load(self):
        model_size = self.options.get("model") or DEFAULT_MODEL
        device = self.options.get("device", "auto")
        compute_type = self.options.get("compute_type", "default")
        key = (model_size, device, compute_type)
        cached = self._model_cache.get(key)
        if cached is not None:
            return cached
        from faster_whisper import WhisperModel  # type: ignore

        download_root = str(self.config.models_dir) if self.config is not None else None
        model = WhisperModel(
            model_size, device=device, compute_type=compute_type, download_root=download_root
        )
        self._model_cache[key] = model
        return model

    def transcribe(
        self,
        audio_path: Optional[str] = None,
        *,
        captions_path: Optional[str] = None,
        language: Optional[str] = None,
    ) -> ASRResult:
        if not audio_path:
            raise ValueError("WhisperASR requires an audio_path.")
        model = self._load()
        # Domain glossary biasing via initial_prompt (spec §4 stage 2).
        initial_prompt = self.options.get("glossary_prompt")
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            initial_prompt=initial_prompt,
        )
        segments = []
        for seg in segments_iter:
            words = [
                Word(word=w.word, start=float(w.start), end=float(w.end))
                for w in (seg.words or [])
                if w.start is not None and w.end is not None
            ]
            segments.append(
                Segment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                )
            )
        return ASRResult(
            segments=segments,
            language=getattr(info, "language", None),
            duration=getattr(info, "duration", None),
            backend=self.name,
        )
