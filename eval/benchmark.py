"""Real-corpus benchmark: shown-only visual lift + refusal vs confabulation.

Unlike ``eval/harness.py`` (the synthetic golden CI gate), this runs over REAL,
license-vetted videos (see ``docs/benchmark/SOURCES.md``) and produces the two
headline numbers memovox's value proposition rests on:

  1. **Shown-only visual lift** — accuracy on questions whose answer is ON SCREEN
     but never spoken, with the visual track OFF (transcript-only) vs ON
     (``--with-video``). A large positive delta is the empirical case for
     multimodal fusion over transcript-only RAG.
  2. **Refusal vs confabulation** — on adversarial out-of-corpus questions, does
     memovox refuse (``low_evidence``) rather than fabricate an answer?

This is intentionally NOT wired into ``make test`` / the CI gates: it needs a
connected machine plus ffmpeg + tesseract and real media, and its numbers are a
dated snapshot, not a determinism invariant.

Fixtures (``eval/benchmark/``):
  manifest.json : [{"video_id", "path", "license", "attribution", "source_url"}]
  qa.json       : [{"q", "expects": "present"|"absent",
                    "modality": "speech-only"|"shown-only"|"both"|"none",
                    "answer_substrings": [...]}]

Run::

  python -m eval.benchmark --manifest eval/benchmark/manifest.json \
      --qa eval/benchmark/qa.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from typing import Callable, List, Optional, Tuple

# QA item expectations.
PRESENT, ABSENT = "present", "absent"
# Response buckets.
CORRECT, WRONG, REFUSED, CONFABULATED = "correct", "wrong", "refused", "confabulated"
# Ingestion conditions (the A/B).
AUDIO_ONLY, WITH_VIDEO = "audio_only", "with_video"


def classify(low_evidence: bool, text: str, expects: str,
             answer_substrings: List[str]) -> str:
    """Bucket one response. Refusal is the hard boolean ``low_evidence``.

    - ABSENT question: refused => correct refusal; otherwise confabulation.
    - PRESENT question: refused => REFUSED (a wrong, over-cautious abstention);
      else CORRECT if any expected substring appears, else WRONG.
    """
    refused = bool(low_evidence)
    if expects == ABSENT:
        return REFUSED if refused else CONFABULATED
    if refused:
        return REFUSED
    hay = (text or "").lower()
    if any(str(s).lower() in hay for s in (answer_substrings or [])):
        return CORRECT
    return WRONG


def _present_accuracy(records: list, condition: str, modality: str) -> Optional[float]:
    rel = [r for r in records if r["condition"] == condition
           and r["expects"] == PRESENT and r["modality"] == modality]
    if not rel:
        return None
    correct = sum(1 for r in rel if r["bucket"] == CORRECT)
    return round(correct / len(rel), 4)


def summarize(records: list) -> dict:
    """Reduce per-(condition, question) records to the headline metrics."""
    modalities = sorted({r["modality"] for r in records if r["expects"] == PRESENT})
    present_accuracy = {
        cond: {m: _present_accuracy(records, cond, m) for m in modalities}
        for cond in (AUDIO_ONLY, WITH_VIDEO)
    }
    off = _present_accuracy(records, AUDIO_ONLY, "shown-only")
    on = _present_accuracy(records, WITH_VIDEO, "shown-only")
    shown_only_lift = round(on - off, 4) if off is not None and on is not None else None

    refusal = {}
    for cond in (AUDIO_ONLY, WITH_VIDEO):
        ab = [r for r in records if r["condition"] == cond and r["expects"] == ABSENT]
        if not ab:
            refusal[cond] = None
            continue
        refused = sum(1 for r in ab if r["bucket"] == REFUSED)
        confab = sum(1 for r in ab if r["bucket"] == CONFABULATED)
        refusal[cond] = {
            "n": len(ab),
            "correct_refusal_rate": round(refused / len(ab), 4),
            "confabulation_rate": round(confab / len(ab), 4),
        }

    # Of the shown-only answers the visual track recovered, how many are honestly
    # flagged as unverified on-screen content (they should all be).
    so_hits = [r for r in records if r["condition"] == WITH_VIDEO
               and r["modality"] == "shown-only" and r["bucket"] == CORRECT]
    ocr_flag_rate = (round(sum(1 for r in so_hits if r.get("ocr_unverified")) / len(so_hits), 4)
                     if so_hits else None)

    return {
        "present_accuracy": present_accuracy,
        "shown_only_lift": shown_only_lift,
        "refusal": refusal,
        "shown_only_ocr_flag_rate": ocr_flag_rate,
        "n_records": len(records),
    }


# An engine is (ingest, ask): ingest(condition, video) -> None; ask(condition, query)
# -> {"low_evidence": bool, "text": str, "citations": [{"modality", "ocr_unverified"}]}.
Engine = Tuple[Callable[[str, dict], None], Callable[[str, str], dict]]


def _default_engine(work_dir: str) -> Engine:
    """Real engine: a fresh local store per condition, ingesting with the visual
    track off vs on. Imported lazily so the scoring logic stays import-cheap."""
    from memovox import Memovox

    stores: dict = {}

    def _mv(condition: str):
        if condition not in stores:
            stores[condition] = Memovox(store=str(pathlib.Path(work_dir) / condition))
        return stores[condition]

    def ingest(condition: str, video: dict) -> None:
        _mv(condition).ingest(video["path"], with_video=(condition == WITH_VIDEO))

    def ask(condition: str, query: str) -> dict:
        ans = _mv(condition).ask(query)
        cits = [{"modality": c.modality,
                 "ocr_unverified": bool(getattr(c, "ocr_unverified", False))}
                for c in ans.citations]
        return {"low_evidence": ans.low_evidence, "text": ans.text, "citations": cits}

    return ingest, ask


def run_benchmark(manifest: list, qa: list, *, work_dir: Optional[str],
                  engine: Optional[Engine] = None) -> dict:
    """Ingest the corpus twice (visual off/on), ask the QA set under each, score."""
    ingest, ask = engine or _default_engine(work_dir or tempfile.mkdtemp(prefix="mvbench-"))
    for cond in (AUDIO_ONLY, WITH_VIDEO):
        for video in manifest:
            ingest(cond, video)
    records = []
    for cond in (AUDIO_ONLY, WITH_VIDEO):
        for item in qa:
            resp = ask(cond, item["q"])
            bucket = classify(resp["low_evidence"], resp.get("text", ""),
                              item.get("expects", PRESENT),
                              item.get("answer_substrings", []))
            records.append({
                "condition": cond,
                "q": item["q"],
                "expects": item.get("expects", PRESENT),
                "modality": item.get("modality", "none"),
                "bucket": bucket,
                "ocr_unverified": any(c.get("ocr_unverified") for c in resp.get("citations", [])),
            })
    return {"summary": summarize(records), "records": records}


def to_markdown(summary: dict) -> str:
    lines = ["# memovox benchmark", ""]
    lift = summary.get("shown_only_lift")
    lines.append(f"**Shown-only visual lift:** {lift if lift is not None else 'n/a'} "
                 "(with-video accuracy − audio-only accuracy, on shown-only questions)")
    lines.append("")
    lines.append("| modality | audio-only | with-video |")
    lines.append("|---|---|---|")
    pa = summary.get("present_accuracy", {})
    mods = sorted(set(pa.get(AUDIO_ONLY, {})) | set(pa.get(WITH_VIDEO, {})))
    for m in mods:
        lines.append(f"| {m} | {pa.get(AUDIO_ONLY, {}).get(m)} | {pa.get(WITH_VIDEO, {}).get(m)} |")
    lines.append("")
    ref = (summary.get("refusal") or {}).get(WITH_VIDEO)
    if ref:
        lines.append(f"**Refusal (out-of-corpus, with-video):** correct-refusal "
                     f"{ref['correct_refusal_rate']}, confabulation "
                     f"{ref['confabulation_rate']} (n={ref['n']})")
    flag = summary.get("shown_only_ocr_flag_rate")
    if flag is not None:
        lines.append(f"**Shown-only answers flagged `ocr_unverified`:** {flag}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="memovox real-corpus benchmark")
    ap.add_argument("--manifest", required=True, help="path to manifest.json")
    ap.add_argument("--qa", required=True, help="path to qa.json (modality-tagged)")
    ap.add_argument("--work", default=None, help="working dir for the two stores")
    ap.add_argument("--json", default=None, help="write the full report JSON here")
    args = ap.parse_args(argv)

    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    qa = json.loads(pathlib.Path(args.qa).read_text())
    report = run_benchmark(manifest, qa, work_dir=args.work)
    print(to_markdown(report["summary"]))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\n[wrote {args.json}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
