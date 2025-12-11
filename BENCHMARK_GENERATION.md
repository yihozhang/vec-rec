# Automatic Benchmark Generation

This document describes the automatic benchmark generation feature added to `codegen.py`.

## Overview

The `generate_benchmark()` function automatically generates a complete C++ benchmarking program for generated kernels. This eliminates the need to manually write benchmarking code and provides consistent, reliable performance measurements.

## Features

- **Automatic benchmark generation**: Creates a complete C++ program with timing infrastructure
- **Correctness checking**: Optionally compares outputs between different kernel implementations
- **Configurable parameters**: Control input size, number of iterations, and tolerance levels
- **Multiple kernel support**: Benchmark multiple kernel variants in a single program

## Usage

### Basic Example

```python
from vecrec import CodeGen, instantiate_kernels, generate_benchmark
from vecrec.expr import Recurse, TIKernel, Var

# Define your signal processing expression
kernel = TIKernel([0, 1.8, -0.9])
signal = Var("x")
expr = Recurse(kernel, signal)

# Create code generator
codegen = CodeGen(256)  # 256-bit SIMD

# Generate kernel code
code = codegen.generate(expr, "my_kernel")

# Generate kernel header
instantiate_kernels("output.h", [code])

# Generate benchmark program
generate_benchmark(
    codegen=codegen,
    exprs=[expr],
    kernel_names=["my_kernel"],
    output_path="benchmark.cpp"
)
```

### Advanced Example with Correctness Checking

```python
from vecrec import CodeGen, instantiate_kernels, generate_benchmark
from vecrec.transform import Dilate
from vecrec.expr import Recurse, TIKernel, Var

# Original kernel
kernel = TIKernel([0, 1.8, -0.9])
signal = Var("x")
expr = Recurse(kernel, signal)

# Optimized kernel using transformations
optimized = Dilate().apply_signal(expr)[0]

# Generate code
codegen = CodeGen(256)
original_code = codegen.generate(expr, "original")
optimized_code = codegen.generate(optimized, "optimized")

# Generate kernel header
instantiate_kernels("output.h", [original_code, optimized_code])

# Generate benchmark with correctness checking
generate_benchmark(
    codegen=codegen,
    exprs=[expr, optimized],
    kernel_names=["original", "optimized"],
    output_path="benchmark.cpp",
    include_correctness_check=True,      # Enable correctness checking
    correctness_tolerance=1e-3,          # Floating point tolerance
    input_size=100000,                   # Input data size
    warmup_iterations=10,                # Warmup iterations
    benchmark_iterations=100             # Benchmark iterations
)
```

### Compiling and Running

After generating the benchmark, compile and run it:

```bash
clang++ -std=c++20 -O2 -march=native benchmark.cpp -o benchmark
./benchmark
```

Example output:
```
original: 27.22 us per iteration
optimized: 4.42 us per iteration

Correctness checking:
optimized matches original: PASS
```

## API Reference

### `generate_benchmark()`

Generates a C++ benchmarking program for the given kernels.

**Parameters:**
- `codegen` (CodeGen): CodeGen instance used to generate the kernels
- `exprs` (List): List of signal expressions corresponding to each kernel
- `kernel_names` (List[str]): List of names corresponding to each kernel
- `output_path` (str): Path where the benchmark C++ file will be written
- `include_correctness_check` (bool, optional): If True, adds correctness checking code. Default: False
- `correctness_tolerance` (float, optional): Tolerance for floating point comparisons. Default: 1e-3
- `input_size` (int, optional): Size of input data for benchmarking. Default: 1000000
- `warmup_iterations` (int, optional): Number of warmup iterations before timing. Default: 10
- `benchmark_iterations` (int, optional): Number of iterations for timing. Default: 100

**Returns:** None

**Generated Benchmark Structure:**

The generated benchmark program includes:
1. Random input data generation
2. Warmup runs to stabilize cache and branch predictor
3. Timed execution with multiple iterations
4. Output collection for correctness checking
5. Performance reporting in microseconds per iteration
6. Optional correctness checking comparing all kernels against the first one

## Implementation Details

### Timing Methodology

The benchmark uses `std::chrono::high_resolution_clock` for precise timing measurements. For each kernel:

1. **Warmup phase**: Runs the kernel a configurable number of times to warm up caches and branch predictors
2. **Reset**: Recreates the kernel to reset internal state
3. **Timed phase**: Runs the kernel multiple times, recreating it for each iteration to ensure consistent input data processing
4. **Results**: Reports average time per iteration in microseconds

### Correctness Checking

When enabled, correctness checking:
- Stores outputs from all kernel runs
- Compares each kernel's output against the first kernel (reference)
- Reports PASS/FAIL with mismatch details
- Uses configurable floating point tolerance to account for numerical differences

### Input Data

The benchmark generates random input data using:
- Mersenne Twister random number generator (`std::mt19937`)
- Uniform distribution in range [-1.0, 1.0]
- Deterministic seeding from `std::random_device` for reproducibility

## Notes

- The generated benchmark requires C++20 and a compiler supporting vector extensions (e.g., Clang, GCC)
- For best results, compile with optimization flags (`-O2` or `-O3`) and `-march=native`
- Correctness checking may show small differences due to floating point arithmetic and different computation orders
- Adjust `correctness_tolerance` based on your specific requirements
