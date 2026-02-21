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

ASTs are declared under `src/vecrec/expr/`: `base.py` declares the base classes, `kernel.py` and `kernel_ops.py` declare the kernel classes and the kernel operator classes, and `signal.py` and `signal_ops.py` declare the signal classes and the signal operator classes.

The core concept in this project is signal, which is like a stream in stream processing.
A `Convolve` declares a convolution between a kernel and a signal, so it defines a finite impulse response filter (FIR).
Similarly, a `Recurse` takes a kernel and a signal and declares an infinite impulse response filter (IIR), where each output value depends on the previous outputs.

To extend this concept to 2D, we define Signal2D and 2D operators like Convolve2D, Recurse2D, Repeater, and Ith.
Repeater is like the delay operator in digital signal processing, but delays one row. They have the following type signature:
```
Convolve2D : Kernel2D -> Signal2D -> Signal1D
Recurse2D :  Kernel2D -> Signal2D -> Signal2D
Repeater : int -> (Signal2D -> Signal1D) -> Signal2D # the closure takes the previous n-1 rows.
```

## Testing

When writing test, make sure to run the generated code using e.g., `generate_and_run_benchmark`, and assert the code compiles and runs by checking the returned value is a dictionary whose 'return_code' is 0.

## Testing and Typechecking

* `uv run mypy src`
* `uv run mypy tests`
* `uv run mypy benchmarks`
* `uv run pytest`
