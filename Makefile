.PHONY: install lint test smoke

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

test:
	uv run pytest -q

# One traced call per provider; aliases without keys are skipped. Override: make smoke ALIASES="claude-haiku gpt-luna"
smoke:
	uv run secrag smoke $(ALIASES)
