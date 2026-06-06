"""Test package bootstrap: make ``src/`` importable without installing, and
pin every model slot to its deterministic, dependency-free fallback so the
whole suite is hermetic (never resolves to a network-backed model download).

This module is imported by BOTH ``python -m unittest discover`` (``tests`` is a
package) and ``pytest``, so it is the single source of truth for the hermetic
environment — there is intentionally no duplicate ``conftest.py`` env block.
``Settings.from_env`` reads ``MEMOVOX_<FIELD>`` at call-time, which always
happens after this package is imported, so setting it here is early enough.
``setdefault`` is used so an explicit override (e.g. a benchmark opting into a
real backend) still wins.
"""

import os
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Hermetic defaults: free fallbacks only, and forbid any HF/transformers
# network access even if an optional package happens to be installed.
os.environ.setdefault("MEMOVOX_EMBED_BACKEND", "hashing")
os.environ.setdefault("MEMOVOX_NLI_BACKEND", "lexical")
os.environ.setdefault("MEMOVOX_ASR_BACKEND", "captions")
os.environ.setdefault("MEMOVOX_LLM_BACKEND", "none")
os.environ.setdefault("MEMOVOX_VLM_BACKEND", "none")
os.environ.setdefault("MEMOVOX_OCR_BACKEND", "none")
# Slots added by later tracks (M1.1 visual_embed, M2.1 rerank, plus entity +
# voiceprint) — pin them too so a machine that HAS the optional ML deps installed
# (cross-encoder / colpali / wikidata / pyannote) but is offline/uncached stays
# hermetic instead of attempting a model download. Must cover EVERY *_backend slot.
os.environ.setdefault("MEMOVOX_RERANK_BACKEND", "none")
os.environ.setdefault("MEMOVOX_VISUAL_EMBED_BACKEND", "signature")
os.environ.setdefault("MEMOVOX_ENTITY_BACKEND", "none")
os.environ.setdefault("MEMOVOX_VOICEPRINT_BACKEND", "none")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
