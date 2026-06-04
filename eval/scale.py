"""Opt-in scale harness (M0.2 W7) — synthetic-N p95 latency + ANN recall@k.

NOT part of the CI gate: this module is never imported by ``eval/harness.py`` and
never run by ``make test``. Run it on demand to measure the free path (and any
opt-in ANN backend) at a scale the 2-3 video golden corpus cannot reach::

    python -m eval.scale --n 2000 --k 10
    python -m eval.scale --n 2000 --ann lance     # recall@k of an ANN vs exact

Everything is deterministic (seeded RNG) so runs are reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from memovox.config import Config, Settings
from memovox.errors import BackendUnavailable
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.backends import get_vector_index
from memovox.observe import Tracer
from memovox.vectormath import normalize


def _rand_vec(rng: random.Random, dim: int):
    return normalize([rng.gauss(0.0, 1.0) for _ in range(dim)])


def _synthesize(store: LoomStore, n: int, dim: int, rng: random.Random) -> None:
    store.upsert_video(Video(video_id="syn", source_url="syn", title="syn", content_hash="syn"))
    for i in range(n):
        mid = f"syn#m{i:06d}"
        store.add_moment(
            Moment(mid, "syn", float(i), float(i) + 1.0, f"synthetic moment {i}", "spk", index=i),
            _rand_vec(rng, dim),
        )


def _p95(values):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def recall_at_k(approx, exact, k: int) -> float:
    """|approx_topk ∩ exact_topk| / k — 1.0 when the exact index is compared to itself."""
    if not exact:
        return 1.0
    inter = len({a for a, _ in approx[:k]} & {e for e, _ in exact[:k]})
    return inter / min(k, len(exact))


def run(*, n: int = 200, dim: int = 64, k: int = 10, queries: int = 50,
        seed: int = 1234, ann_backend: str = "exact") -> dict:
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(store=str(Path(tmp) / "store"), settings=Settings()).ensure()
        with LoomStore(cfg) as store:
            _synthesize(store, n, dim, rng)
            qs = [_rand_vec(rng, dim) for _ in range(queries)]

            # exact free-path p95 (reusing the M0.1 tracer for per-query wall_ms)
            tracer = Tracer("scale")
            for qv in qs:
                with tracer.span("vector_search") as sp:
                    sp.add_counter("results", len(store.vector_search(qv, k)))
            walls = [s.wall_ms for s in tracer.spans]

            # recall@k: ANN vs exact (exact-vs-exact == 1.0 when no ANN is available)
            try:
                if ann_backend == "exact":
                    index = store.vector_index
                else:
                    index = get_vector_index(ann_backend, conn=store.conn)
                backend = index.name
                recalls = [
                    recall_at_k(index.search(qv, k), store.vector_search(qv, k), k)
                    for qv in qs
                ]
                recall = sum(recalls) / len(recalls) if recalls else 1.0
            except (BackendUnavailable, NotImplementedError):
                backend = f"{ann_backend} (unavailable -> exact)"
                recall = 1.0

    return {
        "n": n, "dim": dim, "k": k, "queries": queries, "seed": seed,
        "p95_ms": round(_p95(walls), 4),
        "recall_at_k": round(recall, 6),
        "ann_backend": backend,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.scale",
        description="Opt-in synthetic-scale latency + ANN recall harness (not a CI gate).",
    )
    p.add_argument("--n", type=int, default=2000, help="synthetic moments.")
    p.add_argument("--dim", type=int, default=64, help="embedding dimensionality.")
    p.add_argument("--k", type=int, default=10, help="top-k cutoff.")
    p.add_argument("--queries", type=int, default=50, help="number of probe queries.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ann", default="exact",
                   help="vector backend for recall@k (exact/lance/qdrant).")
    args = p.parse_args(argv)
    print(json.dumps(run(n=args.n, dim=args.dim, k=args.k, queries=args.queries,
                         seed=args.seed, ann_backend=args.ann), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
