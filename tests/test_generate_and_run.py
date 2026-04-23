import numpy as np

from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.expr.signal_ops import PointwiseMul, SAdd
from vecrec.optimize import get_candidate_schedules
from vecrec.transform import (
    AnnotateLanes,
    Any,
    ConstantFold,
    Delay,
    Dilate,
    Optional,
    PushDownConvertLanes,
    SaturatePreorder,
    Seq,
)
from vecrec.expr import Convolve, Impulse, Num, PointwiseDiv, Recurse, TIKernel, TVKernel, Type, Var, with_types
from vecrec.util import ElementType
from vecrec.codegen import generate_kernel_executable


def test_generate_and_run() -> None:
    with with_types(Type.Arith, ElementType.Float):
        kernel = TIKernel([0, 1.8, -0.9])
        signal = Var("x")
        expr = Recurse(kernel, signal)

    # Apply transformations to optimize the kernel
    schedule = Seq(
        Optional(
            Seq(
                Dilate(),
                Dilate(),
                Any(Dilate(), Delay()),
                SaturatePreorder(ConstantFold),
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

    if result["return_code"] == 0:
        print("\n✓ Benchmark completed successfully!\n")
        print("Output:")
        print(result["output"])
    else:
        print("\n✗ Benchmark failed!\n")
        print("Error:", result["error"])
        if result["output"]:
            print("Output:", result["output"])
        assert False, "Benchmark failed"


def test_generate_kernel_executable_roundtrip() -> None:
    with with_types(Type.Arith, ElementType.Float):
        kernel = TIKernel([1])
        signal = Var("x")
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
    with with_types(Type.Arith, ElementType.I64):
        fib = Recurse(
            TIKernel([0, 1, 1]),
            Var("x"),
        )
        fib_sum = Recurse(
            TIKernel([0, 1]),
            fib,
        )

    schedule = Seq(
        SaturatePreorder(ConstantFold),
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
    with with_types(Type.Arith, ElementType.Float):
        a = Var("a")
        b = Var("b")
        h = Recurse(
            TVKernel([Num(0.0), Convolve(TIKernel.z(n=-2), a), (Convolve(TIKernel.z(n=-2), b))]),
            SAdd(Impulse(1.0), Convolve(TIKernel.z(), Impulse(1.0))),
        )
        delayed_h = Convolve(TIKernel.z(), h)
        program = PointwiseDiv(h, delayed_h)

    results = get_candidate_schedules(program, 512)
    assert len(results) > 0

    codegen = CodeGen()
    runner = generate_kernel_executable(codegen, results[0])

    n = 64
    # a_in = (-1) ** np.arange(n)
    # b_in = (-1) ** np.arange(n)
    a_in = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)
    b_in = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)
    a_in[0] = 0.0
    b_in[0] = 0.0
    output, elapsed_us = runner.run({"a": a_in[1:], "b": b_in[1:]})

    f_expected = np.zeros(n, dtype=np.float32)
    f_expected[0] = 1.0
    for i in range(1, n):
        f_expected[i] = a_in[i] + b_in[i] / f_expected[i - 1]
    assert np.allclose(output[2:], f_expected[1:-2], atol=1e-3, rtol=1e-3)


def test_common_subexpression_convolve_div() -> None:
    """Convolve(k1, x) / Convolve(k2, Convolve(k1, x)) produces correct output.

    k1 delays by 1:  c[t] = x[t-1]
    k2 is FIR:       d[t] = c[t] + 0.5*c[t-1] = x[t-1] + 0.5*x[t-2]
    result[t] = x[t-1] / (x[t-1] + 0.5*x[t-2])

    The same object `cse = Convolve(k1, x)` appears in both the numerator
    and the denominator, exercising common subexpression handling.
    """
    with with_types(Type.Arith, ElementType.Float):
        k1 = TIKernel([0.0, 1.0])  # delay by 1
        k2 = TIKernel([1.0, 0.5])  # FIR: y[t] = x[t] + 0.5*x[t-1]
        x = Var("x")
        cse = Convolve(k1, x)  # shared node
        program = PointwiseMul(cse, Convolve(k2, cse))

    results = get_candidate_schedules(program, 512)
    assert len(results) > 0

    runner = generate_kernel_executable(CodeGen(), results[0])

    n = 64
    rng = np.random.default_rng(42)
    x_in = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)

    output, _ = runner.run(x_in)

    # Reference: pad with zeros on the left for the delay
    x_padded = np.concatenate([[0.0], x_in]).astype(np.float32)
    c = x_padded[:-1]  # c[t] = x[t-1]
    c_padded = np.concatenate([[0.0], c])
    d = c + 0.5 * c_padded[:-1]  # d[t] = c[t] + 0.5*c[t-1]
    expected = (c * d).astype(np.float32)

    assert output.shape == (n,)
    assert output.dtype == np.float32
    # t=0: c[0]=0, d[0]=0 → skip; t=1: c[1]=x[0], d[1]=x[0] → ratio=1
    assert np.allclose(
        output[1:], expected[1:], atol=1e-5
    ), f"CSE mismatch:\nGot:      {output[:8]}\nExpected: {expected[:8]}"
