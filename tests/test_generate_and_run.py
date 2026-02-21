
import numpy as np

from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.transform import AnnotateLanes, Any, ConstantFold, Delay, Dilate, Optional, PushDownConvertLanes, Seq, Preorder, Try
from vecrec.expr import Convolve, Impulse, Num, PointwiseDiv, Recurse, TIKernel, TVKernel, Type, Var
from vecrec.util import ElementType
from vecrec.codegen import generate_kernel_executable


def test_generate_and_run() -> None:
    # Define a recursive filter kernel
    kernel = TIKernel([0, 1.8, -0.9], Type.Arith, ElementType.Float)
    signal = Var("x", Type.Arith, ElementType.Float)
    expr = Recurse(kernel, signal)
    
    # Apply transformations to optimize the kernel
    schedule = Seq(
        Optional(
            Seq(
                Dilate(),
                Dilate(),
                Any(Dilate(), Delay()),
                Preorder(Try(ConstantFold)),
            )
        ),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = schedule.apply_signal(expr)
    
    # Create code generator
    codegen = CodeGen()
    
    # Generate, compile, and run benchmark
    print("Generating, compiling, and running benchmark...")
    result = generate_and_run_benchmark(
        codegen=codegen,
        exprs=results[:3],
        include_correctness_check=True,
        input_size=1 << 16,  # Smaller size for quick testing
        warmup_iterations=5,
        benchmark_iterations=10,
    )
    
    if result['return_code'] == 0:
        print("\n✓ Benchmark completed successfully!\n")
        print("Output:")
        print(result['output'])
    else:
        print("\n✗ Benchmark failed!\n")
        print("Error:", result['error'])
        if result['output']:
            print("Output:", result['output'])
        assert False, "Benchmark failed"


def test_generate_kernel_executable_roundtrip() -> None:
    kernel = TIKernel([1], Type.Arith, ElementType.Float)
    signal = Var("x", Type.Arith, ElementType.Float)
    expr = Convolve(kernel, signal)

    schedule = Seq(
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = schedule.apply_signal(expr)
    assert len(results) > 0

    codegen = CodeGen()
    runner = generate_kernel_executable(codegen, results[0])

    x = np.linspace(-1.0, 1.0, num=513, dtype=np.float32)
    output, elapsed_us = runner.run(x)

    assert output.shape == x.shape
    assert output.dtype == np.float32
    assert elapsed_us >= 0
    assert np.allclose(output, x, atol=1e-6)


def test_generate_kernel_executable_fibonacci_sum() -> None:
    fib = Recurse(
        TIKernel([0, 1, 1], Type.Arith, ElementType.I64),
        Var("x", Type.Arith, ElementType.I64),
    )
    fib_sum = Recurse(
        TIKernel([0, 1], Type.Arith, ElementType.I64),
        fib,
    )

    schedule = Seq(
        Preorder(Try(ConstantFold)),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = schedule.apply_signal(fib_sum)
    assert len(results) > 0

    codegen = CodeGen()
    runner = generate_kernel_executable(codegen, results[0])

    n = 64
    x = np.zeros(n, dtype=np.int64)
    x[0] = 1

    output, elapsed_us = runner.run(x)

    fib_expected = np.zeros(n, dtype=np.int64)
    fib_expected[0] = 1
    for i in range(1, n):
        fib_expected[i] = fib_expected[i - 1] + (fib_expected[i - 2] if i >= 2 else 0)
    fib_sum_expected = np.cumsum(fib_expected, dtype=np.int64)

    assert output.shape == (n,)
    assert output.dtype == np.int64
    assert elapsed_us >= 0
    assert np.array_equal(output, fib_sum_expected)


def test_generate_kernel_executable_continued_fraction() -> None:
    a = Var("a", Type.Arith, ElementType.Float)
    b = Var("b", Type.Arith, ElementType.Float)
    h = Recurse(
        TVKernel(
            [Num(0.0, Type.Arith, ElementType.Float), a, b],
            Type.Arith,
            ElementType.Float,
        ),
        Impulse(1.0, Type.Arith, ElementType.Float),
    )
    delayed_h = Convolve(TIKernel([0, 1.0], Type.Arith, ElementType.Float), h)
    program = PointwiseDiv(h, delayed_h)

    schedule = Seq(
        Preorder(Try(ConstantFold)),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = schedule.apply_signal(program)
    assert len(results) > 0

    codegen = CodeGen(True)
    runner = generate_kernel_executable(codegen, results[0])

    n = 64
    a_in = np.full(n, 1.0, dtype=np.float32)
    b_in = np.full(n, 1.0, dtype=np.float32)
    output, elapsed_us = runner.run({"a": a_in, "b": b_in})

    f_expected = np.zeros(n, dtype=np.float32)
    f_expected[0] = 1.0
    for i in range(1, n):
        f_expected[i] = a_in[i] + b_in[i]/f_expected[i-1]

    assert np.allclose(output[1:], f_expected[:-1], atol=1e-3, rtol=1e-3)
