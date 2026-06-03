PY ?= python3
export PYTHONPATH := src

.PHONY: test eval lint install dev clean

test:
	$(PY) -m unittest discover -s tests -t . -p 'test_*.py'

eval:
	$(PY) -m eval.harness

lint:
	ruff check src tests || true

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist ./*.egg-info .ruff_cache .pytest_cache
