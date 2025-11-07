# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := default

.PHONY: default install lint test upgrade clean agent-rules

default: agent-rules install lint test

install:
	uv sync --all-extras

lint:
	uv run python devtools/lint.py

test:
	uv run pytest

upgrade:
	uv sync --upgrade --all-extras --dev

claude:
	cat .cursor/rules/general.mdc .cursor/rules/python.mdc > CLAUDE.md

codex:
	cat .cursor/rules/general.mdc .cursor/rules/python.mdc > AGENTS.md

clean:
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf .venv/
	-rm -rf CLAUDE.md AGENTS.md
	-rm -rf .ruff_cache
