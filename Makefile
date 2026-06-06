PY ?= python3
export PYTHONPATH := src
# Pin hash randomization so any set/dict-iteration order is reproducible run-to-run
# (defense-in-depth; the gated metric paths also carry explicit sort tiebreaks).
export PYTHONHASHSEED := 0

.PHONY: test eval benchmark lint install dev clean

test:
	$(PY) -m unittest discover -s tests -t . -p 'test_*.py'

eval:
	$(PY) -m eval.harness

# A/B-rank the available BackendConfigs (auto-shrinks to the single FREE row on a
# bare machine). Add JSON=1 for machine-readable output. Visual configs are
# unrankable on the text corpus (see eval/golden/README.md).
benchmark:
	$(PY) -m eval.harness --benchmark $(if $(JSON),--json,)

lint:
	ruff check src tests || true

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist ./*.egg-info .ruff_cache .pytest_cache
