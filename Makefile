check:
	uv run mypy src tests benchmarks --check-untyped-defs

test:
	uv run pytest -n auto -v tests
