#!/usr/bin/env python3
"""Generate the upgraded-path (BGE-M3 + DeBERTa + cross-encoder) probe battery ONCE so a
review panel can analyze the outputs without each agent loading 2.7GB of models.

    MEMOVOX_MODELS_DIR=/tmp/mv_models_cache PYTHONPATH=src .venv/bin/python \
        stress/upgraded_probes.py --store /tmp/mv_embed3 --out stress/reports/iterG_probes.json
"""
from __future__ import annotations

import argparse
import json
import logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/tmp/mv_embed3")
    ap.add_argument("--out", default="stress/reports/iterG_probes.json")
    args = ap.parse_args()
    logging.getLogger("memovox").setLevel(logging.ERROR)

    from memovox import Memovox

    mv = Memovox(store=args.store, embed_backend="sentence-transformers",
                 nli_backend="deberta-nli", rerank_backend="cross-encoder",
                 llm_backend="none", entity_backend="none", consensus_cosine=0.7)

    # (question, expectation) — expectation documents intent; the panel judges actual.
    ASK_IN = [
        "what is Graham's number?", "what is AGI?",
        "what did Steve Jobs say about connecting the dots?",
        "does saturated fat cause heart disease?", "what are the risks of a carnivore diet?",
        "which luxury watch should I buy first?", "is the Rolex Submariner a good watch?",
        "what is the iPhone 17 Pro camera like?",
        # episodic-memory probes: guard the known-speaker-aware label-strip fix (the bare
        # "ADDIS ..." speaker label must NOT leak into the answer or citation snippets).
        "how do we reconstruct episodic memory?", "how does the brain retrieve a memory?",
        # watch/car topicality probes (round-4 WATCH_CAR fix): the bare topic noun must
        # answer even when every other token is framing.
        "what watch is best for a first purchase?", "what car should I buy?",
        # chess pieces are genuine corpus SUBJECTS (king df=40, queen df=37) — must ANSWER
        # (they are deliberately NOT in the round-6 political-role-word stoplist).
        "what is the best move for the king?",
    ]
    ASK_OOC = [
        "what is the capital of Mongolia?", "who won the 2014 FIFA World Cup?",
        "how do I bake sourdough bread?", "what is the population of Brazil?",
        "how do I knit a wool sweater?", "who discovered penicillin?",
        "who painted the Mona Lisa?", "what is the boiling point of mercury?",
        # incidental verb use of an in-corpus topic noun ('watch') — must still REFUSE
        # (absent subject 'football/game' is below min_df).
        "where can I watch the football game?",
        # generic political ROLE word (round-6 PRESIDENT_ROLE_WORD_OOC_LEAK fix): 'president'
        # recurs incidentally but the real subject 'brazil' is below min_df — must REFUSE.
        "who is the president of Brazil?", "what is the vice president?",
    ]
    ASK_HARD = [  # the previously-accepted lexical-gate limits — document upgraded behavior
        "how do I save energy at home?", "what did Steve Jobs say about death?",
        "how often should I walk my dog?", "should I buy the BMW M3?",
        "what do the sources say about superintelligence?",
    ]
    SYNTH_IN = ["saturated fat", "diet", "AGI", "breakfast"]
    SYNTH_OOC = ["capital of Mongolia", "history of the Roman Empire", "speed of light in a vacuum"]
    CONTRA = ["saturated fat", "breakfast", "diet", "AGI"]

    def ask_row(q):
        a = mv.ask(q)
        cites = list(a.citations)
        return {
            "q": q, "low_evidence": bool(a.low_evidence), "n_citations": len(cites),
            "cite_videos": [c.video_id for c in cites][:8],
            "answer": (a.text or "")[:240],
            "top_citation": (getattr(cites[0], "source_text", "") or cites[0].snippet or "")[:160] if cites else "",
        }

    def synth_row(t):
        s = mv.synthesize(t).to_dict()
        return {"topic": t, "low_evidence": s["low_evidence"], "n_citations": len(s["citations"]),
                "consensus_count": len(s["consensus_points"]), "contradiction_count": len(s["contradictions"]),
                "text": (s["text"] or "")[:240]}

    def contra_row(t):
        pairs = [p.to_dict() for p in mv.contradictions(topic=t)]
        xv = [p for p in pairs if p["a"]["video_id"] != p["b"]["video_id"]]
        return {"topic": t, "n_pairs": len(pairs), "n_cross_video": len(xv),
                "sample": [{"a": p["a"]["text"][:90], "b": p["b"]["text"][:90], "score": p["score"]}
                           for p in xv[:5]]}

    out = {
        "store": args.store,
        "ask_in_corpus": [ask_row(q) for q in ASK_IN],
        "ask_out_of_corpus": [ask_row(q) for q in ASK_OOC],
        "ask_hard_edge_cases": [ask_row(q) for q in ASK_HARD],
        "synthesize_in_corpus": [synth_row(t) for t in SYNTH_IN],
        "synthesize_out_of_corpus": [synth_row(t) for t in SYNTH_OOC],
        "contradictions": [contra_row(t) for t in CONTRA],
    }
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    # stdout summary
    ans_in = sum(1 for r in out["ask_in_corpus"] if not r["low_evidence"])
    ref_ooc = sum(1 for r in out["ask_out_of_corpus"] if r["low_evidence"])
    print(f"ask in-corpus answered: {ans_in}/{len(ASK_IN)} | OOC refused: {ref_ooc}/{len(ASK_OOC)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
