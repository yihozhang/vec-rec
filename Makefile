check:
	uv run mypy src tests benchmarks --check-untyped-defs
	uvx ruff check src tests benchmarks

test:
	uv run pytest -n auto -v tests
