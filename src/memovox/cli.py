"""Command-line interface (spec §8)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .errors import MemovoxError
from .loom import LoomStore
from .sdk import Memovox
from .util import format_span, seconds_to_hms, truncate

BACKEND_FLAGS = {"asr": "asr_backend", "embed": "embed_backend", "nli": "nli_backend",
                 "llm": "llm_backend", "entity_link": "entity_backend"}


def _resolved_video(args) -> Optional[str]:
    """Resolve the target video_id from either the positional or the legacy
    --video flag (export/forget accept both, like show/extract take a positional)."""
    return getattr(args, "video", None) or getattr(args, "video_opt", None)


def _make_memovox(args) -> Memovox:
    overrides = {key: getattr(args, flag) for flag, key in BACKEND_FLAGS.items()
                 if getattr(args, flag, None)}
    if getattr(args, "allow_cpu", False):
        overrides["asr_allow_cpu"] = True
    return Memovox(store=args.store, **overrides)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_ingest(args, mv: Memovox) -> int:
    report = mv.ingest(
        args.source, source_url=args.source_url, title=args.title, captions=args.captions,
        cookies=args.cookies, language=args.lang, glossary=args.glossary, force=args.force,
        with_video=args.with_video,
    )
    print(f"[{report.status}] {report.video_id}  —  {report.title}")
    print(f"  moments: {report.n_moments}   claims: {report.n_claims_committed} committed, "
          f"{report.n_claims_unsupported} unsupported")
    print(f"  asr: {report.asr_backend}   embed: {report.embed_backend}   nli: {report.nli_backend}")
    if report.visual_available:
        print(f"  visual: {report.n_visual_events} events   "
              f"vlm: {report.vlm_backend}   ocr: {report.ocr_backend}")
    else:
        print("  visual: none (no video stream)")
    return 0


def cmd_ask(args, mv: Memovox) -> int:
    answer = mv.ask(" ".join(args.query), video_id=args.video)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(answer.text or "(no answer)")
    if answer.low_evidence:
        print("\n  (low evidence)")
    if answer.citations:
        print("\nCitations:")
        for c in answer.citations:
            ts = seconds_to_hms(c.t_start_s)
            head = f"  [{c.index}] ({ts}) {c.title or c.video_id}"
            if c.speaker:
                head += f" — {c.speaker}"
            print(head)
            if c.deep_link:
                print(f"      {c.deep_link}")
            if c.snippet:
                print(f"      “{truncate(c.snippet, 160)}”")
            if getattr(c, "ocr_unverified", False):
                print("      ⚠ includes on-screen text (not entailment-verified)")
    if answer.clips:
        print("\nClips:")
        for clip in answer.clips:
            span = format_span(clip.t_start_s, clip.t_end_s)
            print(f"  ({span}) {clip.title or clip.video_id}")
            if clip.deep_link:
                print(f"      {clip.deep_link}")
    return 0


def cmd_sync(args, mv: Memovox) -> int:
    if not mv.list_subscriptions():
        print("No subscriptions (add one with: memovox subscribe <url>).")
        return 0
    report = mv.sync()
    print(f"sync: {report.n_new} new, {report.n_skipped} skipped, {report.n_failed} failed")
    for e in report.entries:
        line = f"  [{e['status']}] {e.get('video_id') or e.get('url')}"
        if e.get("error"):
            line += f" — {e['error']}"
        print(line)
    return 0


def cmd_subscribe(args, mv: Memovox) -> int:
    mv.subscribe(args.url)
    print(f"Subscribed: {args.url}")
    return 0


def cmd_unsubscribe(args, mv: Memovox) -> int:
    mv.unsubscribe(args.url)
    print(f"Unsubscribed: {args.url}")
    return 0


def cmd_forget(args, mv: Memovox) -> int:
    video = _resolved_video(args)
    if not video:
        print("forget: a video_id is required (positional or --video).", file=sys.stderr)
        return 2
    if mv.delete_video(video):
        print(f"Forgot {video} (video + derived moments/claims/edges deleted).")
        return 0
    print(f"No such video: {video}", file=sys.stderr)
    return 1


def cmd_subscriptions(args, mv: Memovox) -> int:
    subs = mv.list_subscriptions()
    if not subs:
        print("No subscriptions.")
        return 0
    for url in subs:
        print(url)
    return 0


def cmd_contradictions(args, mv: Memovox) -> int:
    pairs = mv.contradictions(topic=args.topic)
    if not pairs:
        print("No contradictions found.")
        return 0
    for p in pairs:
        print(f"\n{p.relation}  (score {p.score:.2f})")
        print(f"  A [{p.claim_a.video_id}]: {truncate(p.claim_a.text, 100)}")
        if p.deep_link_a:
            print(f"     {p.deep_link_a}")
        print(f"  B [{p.claim_b.video_id}]: {truncate(p.claim_b.text, 100)}")
        if p.deep_link_b:
            print(f"     {p.deep_link_b}")
    return 0


def cmd_synthesize(args, mv: Memovox) -> int:
    syn = mv.synthesize(" ".join(args.topic))
    if args.json:
        print(json.dumps(syn.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(syn.text or "(no synthesis)")
    if syn.low_evidence:
        print("\n  (low evidence)")
    if syn.consensus_points:
        print("\nConsensus:")
        for cp in syn.consensus_points:
            print(f"  ({cp['support_count']} sources, consensus {cp['consensus']:.2f}) "
                  f"{truncate(cp['text'], 120)}")
    if syn.contradictions:
        print("\nDisagreements:")
        for c in syn.contradictions:
            print(f"  {truncate(c['a']['text'], 70)}  ⟷  {truncate(c['b']['text'], 70)}")
    if syn.citations:
        print("\nCitations:")
        for c in syn.citations:
            head = f"  [{c.index}] ({seconds_to_hms(c.t_start_s)}) {c.title or c.video_id}"
            print(head)
            if c.deep_link:
                print(f"      {c.deep_link}")
    return 0


def cmd_evolution(args, mv: Memovox) -> int:
    steps = mv.evolution(entity=args.entity, topic=args.topic)
    scope = f"entity {args.entity!r}" if args.entity else f"topic {args.topic!r}"
    if not steps:
        print(f"No claims found for {scope}.")
        return 0
    print(f"Evolution for {scope} ({len(steps)} step(s)):\n")
    for s in steps:
        when = s.get("published_at") or "undated"
        rel = f"  [{s['relation']}]" if s.get("relation") else ""
        mark = " (superseded)" if s.get("superseded") else ""
        print(f"{when}{rel}{mark}")
        print(f"  {truncate(s['text'], 140)}")
        if s.get("deep_link"):
            print(f"  {s['deep_link']}")
        print()
    return 0


def cmd_consolidate(args, mv: Memovox) -> int:
    report = mv.consolidate()
    print("Consolidation complete:")
    print(f"  topics induced     : {report['topics']}")
    print(f"  contradictions     : {report['contradictions']}")
    print(f"  agreements (supports): {report['supports']}")
    print(f"  consensus clusters : {report['consensus_clusters']}")
    print(f"  claims superseded  : {report['superseded']}")
    return 0


def cmd_export(args, mv: Memovox) -> int:
    video = _resolved_video(args)
    if not video:
        print("export: a video_id is required (positional or --video).", file=sys.stderr)
        return 2
    content = mv.export(video, fmt=args.format)
    if args.out:
        from pathlib import Path

        Path(args.out).expanduser().write_text(content, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        sys.stdout.write(content)
    return 0


def cmd_list(args, mv: Memovox) -> int:
    videos = mv.list_videos()
    if not videos:
        print("No videos ingested yet.")
        return 0
    for v in videos:
        dur = seconds_to_hms(v.duration_s) if v.duration_s else "?"
        print(f"{v.video_id:<24} {dur:>8}  {v.title}")
    print(f"\n{len(videos)} video(s).")
    return 0


def cmd_show(args, mv: Memovox) -> int:
    with LoomStore(mv.config) as store:
        video = store.get_video(args.video)
        if not video:
            print(f"error: no video {args.video!r}", file=sys.stderr)
            return 1
        moments = store.moments_for_video(args.video)
        if args.json:
            print(json.dumps(
                {"video": video.to_dict(), "moments": [m.to_dict() for m in moments]},
                indent=2, ensure_ascii=False))
            return 0
        print(f"{video.video_id}  —  {video.title}")
        if video.source_url:
            print(f"  {video.source_url}")
        print()
        for m in moments:
            print(f"[{seconds_to_hms(m.t_start_s)}–{seconds_to_hms(m.t_end_s)}] "
                  f"{m.speaker_id or ''}")
            print(f"  {truncate(m.transcript, 160)}")
    return 0


def cmd_stats(args, mv: Memovox) -> int:
    s = mv.stats()
    for key in ("videos", "moments", "claims", "claims_committed", "claims_unsupported",
                "entities", "speakers", "edges", "vectors", "visual_vectors"):
        print(f"{key:<20}: {s.get(key)}")
    print(f"{'fts5':<20}: {s.get('fts5')}")
    print(f"{'embedder':<20}: {s.get('embed_meta')}")
    print(f"{'store':<20}: {s.get('store')}")
    ledger = mv.metrics()["ledger"]
    summary = "  ".join(f"{k}={int(v)}" for k, v in sorted(ledger.items())) or "(none yet)"
    print(f"{'metrics ledger':<20}: {summary}")
    return 0


def cmd_metrics(args, mv: Memovox) -> int:
    data = mv.metrics(video_id=getattr(args, "video", None))
    ledger = data["ledger"]
    print("cumulative ledger:")
    if ledger:
        for k, v in sorted(ledger.items()):
            print(f"  {k:<18}: {int(v)}")
    else:
        print("  (no ingests recorded yet)")
    print("\nper-video stage metrics:")
    if not any(data["stage_metrics"].values()):
        print("  (none)")
    for vid, rows in data["stage_metrics"].items():
        if not rows:
            continue
        print(vid)
        for r in rows:
            counters = " ".join(f"{k}={int(v)}" for k, v in sorted(r["counters"].items()))
            caps = "".join(
                f" cap:{c['name']}(drop={c['dropped']})" for c in r["caps"] if c["dropped"]
            )
            print(f"  {r['stage']:<14} {r['wall_ms']:8.2f}ms  {counters}{caps}")
    return 0


def cmd_extract(args, mv: Memovox) -> int:
    import json

    doc = mv.extract(args.video, use_llm=getattr(args, "use_llm", False))
    print(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def cmd_backends(args, mv: Memovox) -> int:
    status = mv.backends()
    for slot, opts in status.items():
        print(f"{slot}:")
        for name, ok in opts.items():
            print(f"  {name:<22} {'available' if ok else 'not installed'}")
    return 0


def cmd_mcp(args, mv: Memovox) -> int:
    from .server.mcp import serve_stdio

    serve_stdio(mv)
    return 0


def _run_worker(mv: Memovox, *, once: bool, concurrency: int, poll_interval: float) -> int:
    from .serving.jobs import JobWorker

    if once:
        ran = JobWorker(mv, once=True).drain()
        print(f"drained {ran} job(s)", file=sys.stderr)
        return 0
    workers = [JobWorker(mv, poll_interval=poll_interval) for _ in range(max(1, concurrency))]
    for w in workers:
        w.start()
    try:
        print(f"memovox-worker: {len(workers)} thread(s), polling every {poll_interval}s "
              f"(Ctrl-C to stop)", file=sys.stderr)
        for w in workers:
            w.join()
    except KeyboardInterrupt:
        for w in workers:
            w.stop()
    return 0


def cmd_worker(args, mv: Memovox) -> int:
    return _run_worker(mv, once=args.once, concurrency=args.concurrency,
                       poll_interval=args.poll_interval)


def worker_main() -> int:
    """Console-script entry (``memovox-worker``) — runs the job worker loop."""
    import argparse

    p = argparse.ArgumentParser(prog="memovox-worker", description="Drain the memovox job queue.")
    p.add_argument("--store", default=None)
    p.add_argument("--once", action="store_true", help="drain the queue then exit.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="worker threads (default 1, deterministic; >1 is opt-in/ungated).")
    p.add_argument("--poll-interval", type=float, default=1.0)
    args = p.parse_args()
    mv = Memovox(store=args.store) if args.store else Memovox()
    return _run_worker(mv, once=args.once, concurrency=args.concurrency,
                       poll_interval=args.poll_interval)


def cmd_serve(args, mv: Memovox) -> int:
    # argparse accepts any int for --port; an out-of-range value reaches socket.bind() and
    # raises OverflowError ("port must be 0-65535"), an ArithmeticError NOT in main()'s caught
    # tuple — so it would escape as a raw traceback. Validate up front for a clean error.
    if not 0 <= args.port <= 65535:
        print("error: --port must be between 0 and 65535", file=sys.stderr)
        return 2
    if getattr(args, "fastapi", False):
        from .server import fastapi_app
        if not fastapi_app.is_available():
            print("FastAPI is not installed. Install it with: pip install 'memovox[serve]'.",
                  file=sys.stderr)
            return 1
        import uvicorn  # type: ignore
        uvicorn.run(fastapi_app.build_app(mv), host=args.host, port=args.port)
        return 0
    from .server.rest import serve

    try:
        serve(mv, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memovox",
        description="Multimodal video-to-knowledge engine (local-first, no API keys).",
    )
    p.add_argument("--version", action="version", version=f"memovox {__version__}")
    p.add_argument("--store", "--data-dir", dest="store", default=None,
                   help="knowledge store dir (default $MEMOVOX_STORE or ~/.memovox).")
    p.add_argument("--asr", help="ASR backend (auto/whisper/captions/fake).")
    p.add_argument("--embed", help="embedder backend (auto/hashing/sentence-transformers).")
    p.add_argument("--nli", help="NLI backend (auto/lexical/deberta-nli).")
    p.add_argument("--llm", help="LLM backend (auto/ollama/none).")
    p.add_argument("--entity-link", dest="entity_link",
                   help="entity linker (auto=offline slug/none/wikidata). "
                        "auto is offline by default; 'wikidata' opts into network egress.")
    p.add_argument("--allow-cpu", action="store_true",
                   help="allow a heavy ASR model to run on CPU (else it fails loud; spec §9).")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    s = sub.add_parser("ingest", help="ingest a video/audio/transcript or URL.")
    s.add_argument("source", help="local file path or URL.")
    s.add_argument("--source-url", help="canonical URL (for deep links when ingesting a local file).")
    s.add_argument("--title")
    s.add_argument("--captions", help="path to a transcript/subtitle file for a media source.")
    s.add_argument("--cookies", help="Netscape cookie file for gated URLs.")
    s.add_argument("--lang")
    s.add_argument("--glossary", nargs="*", help="domain terms to bias ASR.")
    s.add_argument("--force", action="store_true", help="re-ingest even if unchanged.")
    s.add_argument("--with-video", action="store_true",
                   help="for URLs: download video+audio (not audio-only) so the "
                        "visual track (keyframes/OCR/VLM) can be analyzed.")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("ask", help="ask a grounded, cited question.")
    s.add_argument("query", nargs="+")
    s.add_argument("--video", help="restrict to one video_id.")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("sync", help="ingest new items from subscriptions.json.")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("subscribe", help="add a channel/playlist/video URL to subscriptions.")
    s.add_argument("url")
    s.set_defaults(func=cmd_subscribe)

    s = sub.add_parser("unsubscribe", help="remove a source from subscriptions.")
    s.add_argument("url")
    s.set_defaults(func=cmd_unsubscribe)

    s = sub.add_parser("subscriptions", help="list subscribed sources.")
    s.set_defaults(func=cmd_subscriptions)

    s = sub.add_parser("contradictions", help="surface cross-corpus disagreements.")
    s.add_argument("--topic", help="restrict to a topic.")
    s.set_defaults(func=cmd_contradictions)

    s = sub.add_parser("consolidate", help="run the cross-corpus consolidation job (topics, contradictions, consensus, dedup).")
    s.set_defaults(func=cmd_consolidate)

    s = sub.add_parser("synthesize", help="corpus-level synthesis of a topic (consensus + disagreements).")
    s.add_argument("topic", nargs="+")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_synthesize)

    s = sub.add_parser("evolution", help="trace how a claim/position changed over time.")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--entity", help="entity name or id (ent:<slug>).")
    g.add_argument("--topic", help="free-text topic.")
    s.set_defaults(func=cmd_evolution)

    s = sub.add_parser("export", help="export a per-video digest.")
    s.add_argument("video", nargs="?", help="video_id (like show/extract; or use --video).")
    s.add_argument("--video", dest="video_opt", help="video_id (legacy alias for the positional).")
    s.add_argument("--format", choices=["md", "json"], default="md")
    s.add_argument("--out")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("list", help="list ingested videos.")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show a video's moments.")
    s.add_argument("video")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("stats", help="store statistics.")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("metrics", help="per-stage observability metrics + cumulative ledger.")
    s.add_argument("--video", help="restrict to one video_id.")
    s.set_defaults(func=cmd_metrics)

    s = sub.add_parser("extract", help="emit a schema-validated structured extraction document (JSON).")
    s.add_argument("video")
    s.add_argument("--use-llm", action="store_true", help="use the LLM extractor (default: free rule-based).")
    s.add_argument("--json", action="store_true", help="(default) JSON output.")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("backends", help="list backend availability.")
    s.set_defaults(func=cmd_backends)

    s = sub.add_parser("mcp", help="run the MCP server over stdio (agent-native).")
    s.set_defaults(func=cmd_mcp)

    s = sub.add_parser("serve", help="run the REST API.")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8808)
    s.add_argument("--fastapi", action="store_true",
                   help="use the FastAPI/uvicorn server ([serve] extra) instead of stdlib.")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("forget", help="delete a video + all its derived data (redaction).")
    s.add_argument("video", nargs="?", help="video_id (like show/extract; or use --video).")
    s.add_argument("--video", dest="video_opt", help="video_id (legacy alias for the positional).")
    s.set_defaults(func=cmd_forget)

    s = sub.add_parser("worker", help="run the background job worker (consolidate/sync/ingest).")
    s.add_argument("--once", action="store_true", help="drain the queue then exit.")
    s.add_argument("--concurrency", type=int, default=1,
                   help="worker threads (default 1, deterministic; >1 opt-in/ungated).")
    s.add_argument("--poll-interval", type=float, default=1.0)
    s.set_defaults(func=cmd_worker)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        mv = _make_memovox(args)
        return int(args.func(args, mv) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (MemovoxError, OSError, KeyError, ValueError) as exc:
        # OSError covers FileNotFoundError + IsADirectoryError/NotADirectoryError/
        # PermissionError from a bad ``--out`` target, so an ordinary path mistake prints a
        # clean "error: ..." (exit 1) instead of an uncaught traceback leaking internal frames.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
