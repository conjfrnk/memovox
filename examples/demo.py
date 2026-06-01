"""End-to-end memovox demo using only stdlib fallbacks (no models, no network).

Run from the repo root:  PYTHONPATH=src python examples/demo.py
"""

import tempfile
from pathlib import Path

from memovox import Memovox

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "scaling_laws.en.vtt"


def main() -> None:
    with tempfile.TemporaryDirectory() as store:
        mv = Memovox(store=store, llm_backend="none")

        report = mv.ingest(str(SAMPLE), source_url="https://youtu.be/SCALE123")
        print(f"Ingested {report.video_id}: {report.n_moments} moments, "
              f"{report.n_claims_committed} verified claims "
              f"(asr={report.asr_backend}, embed={report.embed_backend})\n")

        for question in [
            "what chunk size is recommended for RAG?",
            "what does hybrid retrieval combine?",
        ]:
            ans = mv.ask(question)
            print(f"Q: {question}")
            print(f"A: {ans.text}")
            for c in ans.citations[:2]:
                print(f"   [{c.index}] {c.deep_link}  ({c.speaker})")
            print()

        print("Stats:", mv.stats())


if __name__ == "__main__":
    main()
