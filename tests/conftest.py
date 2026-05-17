import os

# Tests are themselves run in parallel (pytest -n auto), so compiling each
# benchmark's batches in parallel as well would oversubscribe the CPU and
# cause contention. Force sequential per-benchmark compilation here; the
# parallel mode remains the default for direct benchmark scripts.
os.environ.setdefault("VECREC_PARALLEL_COMPILE", "0")
