# memovox examples

A fully **free, offline** walkthrough — no models, no API keys, no network.
`scaling_laws.en.vtt` is a small sample transcript so you can try the whole
pipeline immediately.

## CLI

```bash
# from the repo root (no install needed)
PYTHONPATH=src python -m memovox --store /tmp/mvdemo \
    ingest examples/scaling_laws.en.vtt --source-url https://youtu.be/SCALE123

PYTHONPATH=src python -m memovox --store /tmp/mvdemo \
    ask "what chunk size is recommended for RAG, and who said it?"

PYTHONPATH=src python -m memovox --store /tmp/mvdemo export --video yt:SCALE123 --format md
PYTHONPATH=src python -m memovox --store /tmp/mvdemo stats
```

Expect a grounded answer with a `[1]` citation that deep-links to
`https://youtu.be/SCALE123?t=24` — the exact second the chunk size is mentioned.

## Python SDK

```bash
PYTHONPATH=src python examples/demo.py
```
