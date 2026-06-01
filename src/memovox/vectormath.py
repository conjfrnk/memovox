"""Tiny pure-Python vector helpers (no numpy dependency).

Used by the hashing embedder path, the embedded vector index, and retrieval, so
the whole stack runs on the standard library. If numpy is present it is *not*
required; these are correct (if modest) reference implementations.
"""

from __future__ import annotations

import math
import struct
from typing import List, Sequence


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def pack_floats(vec: Sequence[float]) -> bytes:
    """Pack a float vector into a compact little-endian float32 blob."""
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_floats(blob: bytes) -> List[float]:
    n = len(blob) // 4
    if n == 0:
        return []
    return list(struct.unpack(f"<{n}f", blob[: n * 4]))
