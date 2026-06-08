PY ?= python3
export PYTHONPATH := src
# Pin hash randomization so any set/dict-iteration order is reproducible run-to-run
# (defense-in-depth; the gated metric paths also carry explicit sort tiebreaks).
export PYTHONHASHSEED := 0

.PHONY: test eval benchmark benchmark-corpus lint install dev clean

test:
	$(PY) -m unittest discover -s tests -t . -p 'test_*.py'

eval:
	$(PY) -m eval.harness

# A/B-rank the available BackendConfigs (auto-shrinks to the single FREE row on a
# bare machine). Add JSON=1 for machine-readable output. Visual configs are
# unrankable on the text corpus (see eval/golden/README.md).
benchmark:
	$(PY) -m eval.harness --benchmark $(if $(JSON),--json,)

# Real-corpus benchmark (shown-only visual lift + refusal vs confabulation) over the
# license-vetted videos in eval/benchmark/. Needs ffmpeg + tesseract + media on disk
# AND ASR (the .[asr] extra); see eval/benchmark/README.md. NOT part of `make test`
# or the CI gates — its numbers are a dated snapshot, not a determinism invariant.
benchmark-corpus:
	$(PY) -m eval.benchmark --manifest eval/benchmark/manifest.json --qa eval/benchmark/qa.json $(if $(OUT),--json $(OUT),)

lint:
	ruff check src tests || true

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist ./*.egg-info .ruff_cache .pytest_cache
