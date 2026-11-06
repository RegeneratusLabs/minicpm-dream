SHELL := /bin/bash
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
VENV ?= .venv
PY := $(VENV)/bin/python
DR := $(VENV)/bin/minicpm-dream
DEMO_DATA ?= ./demo-data
EVAL_DATA ?= ./eval-data

.PHONY: demo demo-real test lint typecheck check eval build release-check clean

demo:
	rm -rf $(DEMO_DATA)
	$(DR) run \
		--source jsonl \
		--input examples/toy-harness/episodes.jsonl \
		--backend mock \
		--mock-script examples/toy-harness/mock_script.json \
		--data-dir $(DEMO_DATA) \
		--since 2026-11-04T14:00:00+00:00 \
		--until 2026-11-05T14:00:00+00:00
	$(DR) search "dentist" --data-dir $(DEMO_DATA)
	@echo
	@echo "diary:   $(DEMO_DATA)/DREAMS.md"
	@echo "review:  $(DEMO_DATA)/exports/review.json"

demo-real:
	rm -rf $(DEMO_DATA)-real
	$(DR) run \
		--source jsonl \
		--input examples/toy-harness/episodes.jsonl \
		--backend llama_cpp \
		--data-dir $(DEMO_DATA)-real \
		--since 2026-11-04T14:00:00+00:00 \
		--until 2026-11-05T14:00:00+00:00

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check .

typecheck:
	$(VENV)/bin/mypy src

check: lint typecheck test

eval:
	rm -rf $(EVAL_DATA)
	$(DR) eval \
		--suite evals/synthetic-days/basic \
		--backend mock \
		--mock-script evals/synthetic-days/basic/mock_script.json \
		--data-dir $(EVAL_DATA)

build:
	uv build

release-check: check demo eval build
	@echo "release artifacts:"
	@ls -lh dist

clean:
	rm -rf $(DEMO_DATA) $(DEMO_DATA)-real $(EVAL_DATA)
	rm -f compare-*.json compare-*.err
