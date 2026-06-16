#!/usr/bin/env python3
"""Stress-test harness for memovox.

Fresh store each run: ingests the full local caption corpus (.stress-corpus/subs),
runs a probe battery (asks / refusals / contradictions / synthesis / timeline),
and auto-derives findings with severities. Deterministic free path
(captions + hashing embed + lexical/deberta NLI + no LLM).

Usage:
    python stress/driver.py --store /tmp/mv_stress --nli lexical \
        --out stress/reports/iterN.json --md stress/reports/iterN.md
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone

CORPUS = ".stress-corpus/subs"

# id -> (group, short label). TkPNgw2VUbU omitted: no English captions (ASR-only).
MANIFEST = [
    ("UF8uR6Z6KLc", "sweet", "Jobs Stanford 2005"),
    ("zjkBMFhNj_g", "sweet", "Karpathy Intro to LLMs"),
    ("XTeJ64KD5cg", "sweet", "Numberphile Graham's Number"),
    ("t3H5D-XxPrI", "asr", "Polyglots 21 languages"),
    ("144uOfr4SYA", "asr", "Munk Debate on AI"),
    ("nkG2SSzPUns", "asr", "NPR Tiny Desk"),
    ("AmlF6xq2SaQ", "asr", "Easy English NYC"),
    ("J7DzL2_Na80", "visual", "Strang Linear Algebra L1"),
    ("WUvTyaaNkzM", "visual", "3Blue1Brown calculus"),
    ("jbkSRLYSojo", "visual", "Rosling 200 Countries"),
    ("kCc8FmEb1nY", "visual", "Karpathy build GPT"),
    ("jGwO_UgTS7I", "visual", "Andrew Ng CS229 L1"),
    ("1M3Vdl6DRkU", "duration", "Lex Fridman #497"),
    ("aqBHXNGKvKU", "contradiction", "Sat fat CAUSES heart disease"),
    ("dzOTaNwiFmA", "contradiction", "Sat fat does NOT (Attia)"),
    ("AxIOGqHQqZM", "contradiction", "Breakfast IS important"),
    ("2su8e-nhMGw", "contradiction", "SKIP breakfast (Berg)"),
    ("n_Smy5-1cHE", "contradiction", "Carnivore risks"),
    ("SdnEbJZoNg8", "contradiction", "Vegan brain"),
    ("5KVDDfAkRgc", "timeline", "Not Ready for Superintelligence"),
    ("YeRS4TbtZWA", "timeline", "AGI timelines 2025"),
]

# Probe asks: (question, expected_video_id or None, expected keyword in answer lower)
ASKS = [
    ("what is Graham's number?", "XTeJ64KD5cg", "graham"),
    ("what are the two files that make up a large language model?", "zjkBMFhNj_g", None),
    ("what did Steve Jobs say about connecting the dots?", "UF8uR6Z6KLc", "dot"),
    ("does saturated fat cause heart disease?", None, "fat"),
    ("is breakfast the most important meal of the day?", None, "breakfast"),
    ("what do the sources say about superintelligence?", None, None),
    ("what are the risks of a carnivore diet?", "n_Smy5-1cHE", None),
    # Panel over-refusal regression cases: in-corpus but named by a df=0 proper noun
    # (speaker/event) — must stay ANSWERED, not refused.
    ("what does Peter Attia say about saturated fat?", None, "fat"),
    ("what did Hans Rosling show about countries and years?", "jbkSRLYSojo", None),
    ("what is AGI?", None, None),
]

# Refusal probes: questions with no answer in the corpus. ask() should NOT fabricate.
# Includes the panel's adversarial counterexamples (out-of-corpus questions whose
# GENERIC tokens scatter across the corpus) — these defeated the iter1 gate.
REFUSALS = [
    "what is the capital of Mongolia?",
    "what did the speaker say about underwater basket weaving championships?",
    "how do I change the oil in a 1997 Honda Civic?",
    "who won the 2014 FIFA World Cup?",
    "what is the boiling point of mercury?",
    "how do I change a car tire?",
    "what are the rules of cricket?",
    "who is the president of France?",
    "what is the tallest mountain in Africa?",
    # Single-distinctive-token cases: the only in-corpus token is an incidental hapax
    # (df<=2), not a genuine topic — the iter2 leak class.
    "what time does the bank open?",
    "how do I file my taxes?",
    "how do I train a puppy?",
    # Polysemy: the topic word recurs in the corpus in a DIFFERENT sense; the
    # question's context words are absent -> refuse.
    "how do I save energy at home?",
    "how often should I walk my dog?",
]

CONTRA_TOPICS = ["saturated fat", "breakfast", "diet", "AGI"]
SYNTH_TOPICS = ["AGI", "saturated fat", "diet"]
# Planted cross-video contradiction pairs we hope to surface.
PLANTED = [
    ("aqBHXNGKvKU", "dzOTaNwiFmA", "saturated fat"),
    ("AxIOGqHQqZM", "2su8e-nhMGw", "breakfast"),
]


def vid(cid):
    return f"yt:{cid}"


def pick_vtt(cid):
    for cand in (f"{cid}.en.vtt", f"{cid}.en-US.vtt", f"{cid}.en-orig.vtt"):
        p = os.path.join(CORPUS, cand)
        if os.path.exists(p):
            return p
    g = sorted(glob.glob(os.path.join(CORPUS, f"{cid}.en*.vtt")))
    return g[0] if g else None


def pub_of(cid):
    p = os.path.join(CORPUS, f"{cid}.info.json")
    if not os.path.exists(p):
        return None, None
    info = json.load(open(p))
    ud = info.get("upload_date")
    pub = f"{ud[:4]}-{ud[4:6]}-{ud[6:8]}T00:00:00Z" if ud else None
    return pub, info.get("title")


def is_trivial(claim: dict) -> bool:
    """Genuine residual junk among COMMITTED claims: an ultra-short fragment or a
    non-claim utterance (greeting/ad/imperative). Deliberately NARROW — the earlier
    'no subject/object' heuristic just measured S-P-O parser structure on substantive
    sentences (a non-finding per the review panel), not junk."""
    from memovox.assay.claims import is_non_claim
    text = (claim.get("text") or "").strip()
    if len(text.split()) < 3:
        return True
    return is_non_claim(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/tmp/mv_stress")
    ap.add_argument("--nli", default="lexical")
    ap.add_argument("--embed", default="hashing")
    ap.add_argument("--out", default="stress/reports/report.json")
    ap.add_argument("--md", default=None)
    ap.add_argument("--stamp", default=None, help="iteration timestamp/label")
    args = ap.parse_args()

    logging.getLogger("memovox").setLevel(logging.ERROR)
    logging.getLogger("memovox.trace").setLevel(logging.ERROR)

    from memovox import Memovox

    if os.path.exists(args.store):
        shutil.rmtree(args.store)

    report = {
        "stamp": args.stamp or "unstamped",
        "nli": args.nli, "embed": args.embed,
        "videos": [], "asks": [], "refusals": [],
        "contradictions": {}, "synthesize": {}, "evolution": {},
        "aggregate": {}, "findings": [], "errors": [],
    }

    def err(where, exc):
        report["errors"].append({"where": where, "error": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc()[-1500:]})

    mv = Memovox(store=args.store, embed_backend=args.embed, nli_backend=args.nli,
                 llm_backend="none", rerank_backend="identity", entity_backend="none")

    # ---- ingest all ----
    for cid, group, label in MANIFEST:
        v = {"id": cid, "group": group, "label": label}
        vtt = pick_vtt(cid)
        if not vtt:
            v["status"] = "no_vtt"
            report["videos"].append(v)
            continue
        pub, title = pub_of(cid)
        v["published_at"] = pub
        try:
            rep = mv.ingest(vtt, source_url=f"https://youtu.be/{cid}",
                            title=title or label, published_at=pub).to_dict()
            v.update({k: rep.get(k) for k in
                      ("video_id", "status", "n_moments", "n_claims_committed",
                       "n_claims_unsupported", "asr_backend", "duration_s")})
            # per-video claim audit
            claims = json.loads(mv.export(rep["video_id"], "json"))["claims"]
            committed = [c for c in claims if c.get("status") in (None, "committed")]
            trivial = [c for c in committed if is_trivial(c)]
            v["distinct_speakers"] = sorted({c.get("speaker_id") or c.get("speaker")
                                             for c in committed if (c.get("speaker_id") or c.get("speaker"))})
            v["n_committed_audited"] = len(committed)
            v["trivial_count"] = len(trivial)
            v["trivial_ratio"] = round(len(trivial) / max(1, len(committed)), 3)
            v["sample_trivial"] = [c.get("text") for c in trivial[:5]]
            # provenance integrity: spans well-formed + deep link present
            bad_span = [c for c in committed
                        if not isinstance(c.get("t_start_s"), (int, float))
                        or not isinstance(c.get("t_end_s"), (int, float))
                        or (c.get("t_end_s") or 0) < (c.get("t_start_s") or 0)]
            v["provenance_violations"] = len(bad_span)
            v["sample_bad_span"] = [c.get("text") for c in bad_span[:3]]
            # caps from metrics
            sm = mv.metrics(video_id=rep["video_id"])["stage_metrics"].get(rep["video_id"], [])
            caps = []
            for st in sm:
                for cap in st.get("caps", []):
                    if cap.get("dropped", 0) > 0 and cap.get("name") != "top_k":
                        caps.append({"stage": st.get("stage"), **cap})
            v["caps_dropped"] = caps
        except Exception as exc:
            v["status"] = "ERROR"
            err(f"ingest:{cid}", exc)
        report["videos"].append(v)

    try:
        report["aggregate"]["consolidate"] = mv.consolidate()
    except Exception as exc:
        err("consolidate", exc)

    # ---- asks ----
    for q, exp_vid, exp_kw in ASKS:
        a = {"q": q, "expect_video": exp_vid, "expect_kw": exp_kw}
        try:
            ans = mv.ask(q)
            cites = [c for c in ans.citations]
            cvids = [c.video_id for c in cites]
            a.update({
                "text_present": bool(ans.text and ans.text.strip()),
                "n_citations": len(cites),
                "low_evidence": bool(ans.low_evidence),
                "cite_videos": cvids[:8],
                "modalities": sorted({getattr(c, "modality", None) for c in cites if getattr(c, "modality", None)}),
                "hit": (vid(exp_vid) in cvids) if exp_vid else None,
                "kw_in_text": (exp_kw in (ans.text or "").lower()) if exp_kw else None,
                "all_cited": all(("[" + str(c.index) + "]") in (ans.text or "") for c in cites) if ans.text else None,
            })
        except Exception as exc:
            a["error"] = str(exc)
            err(f"ask:{q[:30]}", exc)
        report["asks"].append(a)

    # ---- refusals ----
    for q in REFUSALS:
        r = {"q": q}
        try:
            ans = mv.ask(q)
            r.update({
                "text_present": bool(ans.text and ans.text.strip()),
                "low_evidence": bool(ans.low_evidence),
                "n_citations": len(ans.citations),
                "text": (ans.text or "")[:200],
                # a good refusal: low_evidence True OR no text OR no citations
                "refused": bool(ans.low_evidence) or not (ans.text and ans.text.strip()) or len(ans.citations) == 0,
            })
        except Exception as exc:
            r["error"] = str(exc)
            err(f"refusal:{q[:30]}", exc)
        report["refusals"].append(r)

    # ---- contradictions ----
    for topic in CONTRA_TOPICS:
        try:
            pairs = [p.to_dict() for p in mv.contradictions(topic=topic)]
            report["contradictions"][topic] = {
                "n_pairs": len(pairs),
                "n_cross_video": sum(1 for p in pairs if p["a"]["video_id"] != p["b"]["video_id"]),
                "relations": _count([p["relation"] for p in pairs]),
                "pairs": pairs[:12],
            }
        except Exception as exc:
            err(f"contradictions:{topic}", exc)
    # did planted pairs surface (cross-video, either topic key)?
    found = []
    allpairs = [p for t in report["contradictions"].values() for p in t.get("pairs", [])]
    for a, b, topic in PLANTED:
        hit = any({p["a"]["video_id"], p["b"]["video_id"]} == {vid(a), vid(b)} for p in allpairs)
        found.append({"pair": [a, b], "topic": topic, "surfaced": hit})
    report["aggregate"]["planted_contradictions"] = found

    # ---- synthesize ----
    for topic in SYNTH_TOPICS:
        try:
            s = mv.synthesize(topic).to_dict()
            report["synthesize"][topic] = {
                "consensus_count": len(s.get("consensus_points", [])),
                "contradiction_count": len(s.get("contradictions", [])),
                "low_evidence": s.get("low_evidence"),
                "n_citations": len(s.get("citations", [])),
                "consensus_sample": [cp.get("text") if isinstance(cp, dict) else str(cp)
                                     for cp in s.get("consensus_points", [])[:5]],
            }
        except Exception as exc:
            err(f"synthesize:{topic}", exc)

    # ---- evolution / timeline ----
    for entity in ["AGI", "superintelligence"]:
        try:
            steps = mv.evolution(topic=entity)
            dates = [s.get("published_at") for s in steps]
            ordered = [d for d in dates if d]
            report["evolution"][entity] = {
                "n_steps": len(steps),
                "dates": dates,
                "ordered_ok": ordered == sorted(ordered),
                "videos": [s.get("video_id") for s in steps],
            }
        except Exception as exc:
            err(f"evolution:{entity}", exc)

    mv.close()

    # ---- aggregate + findings ----
    vids = [v for v in report["videos"] if v.get("status") not in ("no_vtt",)]
    ok = [v for v in vids if v.get("status") in ("ingested", "replaced", "unchanged")]
    report["aggregate"].update({
        "n_videos": len(vids),
        "n_ingest_ok": len(ok),
        "total_moments": sum(v.get("n_moments", 0) or 0 for v in ok),
        "total_committed": sum(v.get("n_claims_committed", 0) or 0 for v in ok),
        "total_trivial": sum(v.get("trivial_count", 0) or 0 for v in ok),
        "mean_trivial_ratio": round(sum(v.get("trivial_ratio", 0) or 0 for v in ok) / max(1, len(ok)), 3),
        "total_provenance_violations": sum(v.get("provenance_violations", 0) or 0 for v in ok),
    })

    F = report["findings"]

    def finding(sev, code, msg, **extra):
        F.append({"severity": sev, "code": code, "msg": msg, **extra})

    if report["errors"]:
        finding("CRIT", "exceptions", f"{len(report['errors'])} exceptions during run",
                where=[e["where"] for e in report["errors"]])
    for v in vids:
        if v.get("status") == "ERROR":
            finding("CRIT", "ingest_failed", f"{v['id']} ({v['label']}) failed to ingest")
        elif (v.get("n_claims_committed") or 0) == 0:
            finding("HIGH", "zero_claims", f"{v['id']} ({v['label']}) produced 0 committed claims")
        if v.get("provenance_violations"):
            finding("CRIT", "provenance", f"{v['id']} has {v['provenance_violations']} claims with bad spans",
                    sample=v.get("sample_bad_span"))
    if report["aggregate"]["mean_trivial_ratio"] > 0.25:
        worst = sorted(ok, key=lambda v: -(v.get("trivial_ratio") or 0))[:5]
        finding("HIGH", "trivial_claims",
                f"mean trivial-claim ratio {report['aggregate']['mean_trivial_ratio']} (>0.25)",
                worst=[{"id": v["id"], "ratio": v.get("trivial_ratio"), "sample": v.get("sample_trivial")} for v in worst])
    # refusals
    bad_ref = [r for r in report["refusals"] if not r.get("refused")]
    if bad_ref:
        finding("HIGH", "fabrication", f"{len(bad_ref)} out-of-corpus questions answered without low_evidence",
                examples=[{"q": r["q"], "text": r.get("text")} for r in bad_ref])
    # asks hit-rate
    hits = [a for a in report["asks"] if a.get("hit") is True]
    expected_hits = [a for a in report["asks"] if a.get("expect_video")]
    if expected_hits:
        hr = len(hits) / len(expected_hits)
        report["aggregate"]["ask_hit_rate"] = round(hr, 3)
        if hr < 0.8:
            finding("HIGH", "retrieval", f"ask hit-rate {hr:.2f} (<0.80)",
                    misses=[a["q"] for a in expected_hits if not a.get("hit")])
    # planted contradictions
    missed = [p for p in found if not p["surfaced"]]
    if missed:
        # Informational: the free LEXICAL NLI path cannot detect semantic/antonym
        # contradictions (deberta is the real path), AND the planted "saturated fat"
        # pair actually AGREES (both contrarian) — so 0 here is partly correct, not a bug.
        finding("LOW", "contradictions_missed",
                f"{len(missed)}/{len(found)} planted pairs not surfaced (lexical-NLI limit + corpus caveat)",
                missed=missed)
    # synthesize AGI synonym clustering
    agi = report["synthesize"].get("AGI", {})
    if agi and (agi.get("consensus_count", 0) == 0):
        finding("MED", "synthesis_synonyms",
                "synthesize('AGI') found 0 consensus points (token-Jaccard misses synonyms?)")
    # timeline ordering
    for entity, ev in report["evolution"].items():
        if ev.get("n_steps", 0) >= 2 and not ev.get("ordered_ok"):
            finding("HIGH", "timeline_order", f"evolution('{entity}') steps not chronologically ordered",
                    dates=ev.get("dates"))
    # caps
    capped = [v for v in ok if v.get("caps_dropped")]
    if capped:
        finding("LOW", "caps_dropped", f"{len(capped)} videos hit a cap (silent drop)",
                detail=[{"id": v["id"], "caps": v["caps_dropped"]} for v in capped])
    # speaker collapse on multi-speaker
    debate = next((v for v in ok if v["id"] == "144uOfr4SYA"), None)
    if debate and len(debate.get("distinct_speakers", [])) <= 1:
        finding("LOW", "speaker_collapse",
                "multi-speaker debate collapsed to a single speaker (no diarization backend)")

    # ---- write ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)
    if args.md:
        write_md(report, args.md)

    # ---- stdout summary ----
    a = report["aggregate"]
    print(f"\n{'='*70}\nSTRESS REPORT  stamp={report['stamp']}  nli={args.nli}")
    print(f"ingest: {a['n_ingest_ok']}/{a['n_videos']} ok | moments={a['total_moments']} "
          f"claims={a['total_committed']} trivial={a['total_trivial']} "
          f"(mean ratio {a['mean_trivial_ratio']}) prov_violations={a['total_provenance_violations']}")
    print(f"ask hit-rate={a.get('ask_hit_rate')} | "
          f"refusals_ok={sum(1 for r in report['refusals'] if r.get('refused'))}/{len(report['refusals'])}")
    sev_order = {"CRIT": 0, "HIGH": 1, "MED": 2, "LOW": 3}
    for f in sorted(F, key=lambda x: sev_order.get(x["severity"], 9)):
        print(f"  [{f['severity']:4}] {f['code']}: {f['msg']}")
    print(f"errors: {len(report['errors'])}")
    print(f"wrote {args.out}" + (f" + {args.md}" if args.md else ""))
    return 0


def _count(items):
    out = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return out


def write_md(report, path):
    a = report["aggregate"]
    L = [f"# Stress report — {report['stamp']} (nli={report['nli']})", ""]
    L.append(f"- ingest: **{a['n_ingest_ok']}/{a['n_videos']}** ok")
    L.append(f"- moments={a['total_moments']} committed_claims={a['total_committed']} "
             f"trivial={a['total_trivial']} (mean ratio **{a['mean_trivial_ratio']}**)")
    L.append(f"- provenance violations: **{a['total_provenance_violations']}**")
    L.append(f"- ask hit-rate: **{a.get('ask_hit_rate')}** | "
             f"refusals_ok: {sum(1 for r in report['refusals'] if r.get('refused'))}/{len(report['refusals'])}")
    L.append("")
    L.append("## Findings")
    if not report["findings"]:
        L.append("- (none)")
    for f in sorted(report["findings"], key=lambda x: {"CRIT": 0, "HIGH": 1, "MED": 2, "LOW": 3}.get(x["severity"], 9)):
        L.append(f"- **[{f['severity']}] {f['code']}** — {f['msg']}")
    L.append("")
    L.append("## Per-video")
    L.append("| id | group | claims | trivial% | prov! | speakers | caps |")
    L.append("|---|---|---|---|---|---|---|")
    for v in report["videos"]:
        if v.get("status") in ("no_vtt",):
            continue
        L.append(f"| {v['id']} | {v['group']} | {v.get('n_claims_committed')} | "
                 f"{v.get('trivial_ratio')} | {v.get('provenance_violations')} | "
                 f"{len(v.get('distinct_speakers', []))} | {len(v.get('caps_dropped', []))} |")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
