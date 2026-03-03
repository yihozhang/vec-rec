check:
	uv run mypy src tests benchmarks --check-untyped-defs

test:
	uv run pytest -v tests
