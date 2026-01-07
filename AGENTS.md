# VecRec

VecRec is a library for representing and transforming recurrences into efficient vectorized code.
It generates C++ code that can be compiled and run to perform high-throughput signal processing tasks.
The key idea is that we can "delay" and "dilate" the feedback coefficients of a recurrence so that
the computation of the current pixel depends on pixels that are far apart, allowing for vectorization.

## Running VecRec

VecRec is a Python library. You can run the example in `src/vecrec/__init__.py` via `uv run vecrec`. There are other examples under benchmarks that can be run via `uv run python benchmarks/...`, although some have not been implemented.


## Internals

To generate efficient C++ code, VecRec uses the C++ template library under `src/vecrec/templates/common.h`.
Code generation is handled by `src/vecrec/codegen.py`, which translates the internal expression representation into C++ code.

`src/vecrec/transform.py` contains various transformations that can be applied to signal expressions.
These transformations are delicate and changes to it require careful consideration in order to maintain correctness.

## Testing and Typechecking

* `uv run mypy src`
* `uv run mypy tests`
* `uv run mypy benchmarks`
* `uv run pytest`
