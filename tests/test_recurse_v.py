"""Tests for RecurseV, ConvolveV, and VectorAt operators."""

from __future__ import annotations

import numpy as np

from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.codegen import generate_kernel_executable
from vecrec.expr import (
    KernelMatrix,
    SignalVector,
    RecurseV,
    ConvolveV,
    VectorAt,
    TIKernel,
    Var,
    Type,
    with_types,
)
from vecrec.transform import AnnotateLanes, PushDownConvertLanes, Seq
from vecrec.util import ElementType


def _build_schedule(max_bits: int = 32) -> Seq:
    return Seq(AnnotateLanes(max_bits), PushDownConvertLanes())


def _make_mutual_recurse():
    """Build the mutual recurrence:
        f(x) = 0.6*f(x-1) + 0.2*g(x-1) + h1(x)
        g(x) = 0.3*f(x-1) + 0.4*g(x-2) + h2(x)

    KernelMatrix layout:
        A[0][0] = 0.6 * z^{-1}  (f feeds back into f)
        A[0][1] = 0.2 * z^{-1}  (g feeds back into f)
        A[1][0] = 0.3 * z^{-1}  (f feeds back into g)
        A[1][1] = 0.4 * z^{-2}  (g feeds back into g with lag 2)
    """
    with with_types(Type.Arith, ElementType.Float):
        A = KernelMatrix(
            [
                [TIKernel([0.0, 0.6]), TIKernel([0.0, 0.2])],
                [TIKernel([0.0, 0.3]), TIKernel([0.0, 0.0, 0.4])],
            ]
        )
        G = SignalVector([Var("h1"), Var("h2")])
        F = RecurseV(A, G)
        f0 = VectorAt(F, 0)
        f1 = VectorAt(F, 1)
    return f0, f1


def _reference_mutual(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Python reference for the mutual recurrence."""
    n = len(h1)
    f_ref = np.zeros(n, dtype=np.float32)
    g_ref = np.zeros(n, dtype=np.float32)
    f_prev = 0.0
    g_prev1 = 0.0  # g(x-1)
    g_prev2 = 0.0  # g(x-2)
    for x in range(n):
        new_f = 0.6 * f_prev + 0.2 * g_prev1 + h1[x]
        new_g = 0.3 * f_prev + 0.4 * g_prev2 + h2[x]
        g_prev2 = g_prev1
        g_prev1 = new_g
        f_prev = new_f
        f_ref[x] = new_f
        g_ref[x] = new_g
    return f_ref, g_ref


def test_recurse_v_mutual_compile_and_run() -> None:
    """Mutual RecurseV compiles and runs without error."""
    f0, _ = _make_mutual_recurse()
    results = _build_schedule(32).apply_signal(f0)
    assert len(results) > 0

    result = generate_and_run_benchmark(
        codegen=CodeGen(),
        exprs=results[:1],
        include_correctness_check=False,
        input_size=256,
        warmup_iterations=1,
        benchmark_iterations=2,
    )
    assert (
        result["return_code"] == 0
    ), f"Benchmark failed:\nstderr: {result.get('error', '')}\nstdout: {result.get('output', '')}"


def test_recurse_v_mutual_f_correctness() -> None:
    """VectorAt(F, 0) produces the correct f component of the mutual recurrence."""
    f0, _ = _make_mutual_recurse()
    results = _build_schedule(32).apply_signal(f0)
    assert len(results) > 0

    runner = generate_kernel_executable(CodeGen(), results[0])

    n = 64
    h1 = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    h2 = np.random.uniform(-1.0, 1.0, n).astype(np.float32)

    output, _ = runner.run({"h1": h1, "h2": h2})
    f_ref, _ = _reference_mutual(h1, h2)

    assert output.shape == (n,)
    assert output.dtype == np.float32
    assert np.allclose(
        output, f_ref, atol=1e-5
    ), f"f component mismatch:\nGot:      {output[:8]}\nExpected: {f_ref[:8]}"


def test_recurse_v_mutual_g_correctness() -> None:
    """VectorAt(F, 1) produces the correct g component of the mutual recurrence."""
    _, f1 = _make_mutual_recurse()
    results = _build_schedule(32).apply_signal(f1)
    assert len(results) > 0

    runner = generate_kernel_executable(CodeGen(), results[0])

    n = 64
    h1 = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    h2 = np.random.uniform(-1.0, 1.0, n).astype(np.float32)

    output, _ = runner.run({"h1": h1, "h2": h2})
    _, g_ref = _reference_mutual(h1, h2)

    assert output.shape == (n,)
    assert output.dtype == np.float32
    assert np.allclose(
        output, g_ref, atol=1e-5
    ), f"g component mismatch:\nGot:      {output[:8]}\nExpected: {g_ref[:8]}"


def test_convolve_v_compile_and_run() -> None:
    """2×2 ConvolveV FIR compiles and runs.

    out[0][t] = A[0][0] ⊛ g[0] + A[0][1] ⊛ g[1]
    out[1][t] = A[1][0] ⊛ g[0] + A[1][1] ⊛ g[1]

    Using identity-like kernels for simplicity.
    """
    with with_types(Type.Arith, ElementType.Float):
        k_id = TIKernel([1.0])  # identity (no delay, no feedback)
        k_zero = TIKernel([])
        A = KernelMatrix([[k_id, k_zero], [k_zero, k_id]])
        G = SignalVector([Var("x"), Var("y")])
        out = ConvolveV(A, G)
        out0 = VectorAt(out, 0)

    results = _build_schedule(32).apply_signal(out0)
    assert len(results) > 0

    result = generate_and_run_benchmark(
        codegen=CodeGen(),
        exprs=results[:1],
        include_correctness_check=False,
        input_size=256,
        warmup_iterations=1,
        benchmark_iterations=2,
    )
    assert (
        result["return_code"] == 0
    ), f"Benchmark failed:\nstderr: {result.get('error', '')}\nstdout: {result.get('output', '')}"


def test_convolve_v_correctness_identity() -> None:
    """Identity ConvolveV passes x through unchanged."""
    with with_types(Type.Arith, ElementType.Float):
        k_id = TIKernel([1.0])
        k_zero = TIKernel([])
        A = KernelMatrix([[k_id, k_zero], [k_zero, k_id]])
        G = SignalVector([Var("x"), Var("y")])
        out = ConvolveV(A, G)
        out0 = VectorAt(out, 0)

    results = _build_schedule(32).apply_signal(out0)
    assert len(results) > 0

    runner = generate_kernel_executable(CodeGen(), results[0])

    n = 64
    x_in = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    y_in = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    output, elapsed_us = runner.run({"x": x_in, "y": y_in})

    # out[0] = 1.0 * x + 0 * y = x
    assert output.shape == (n,)
    assert np.allclose(output, x_in, atol=1e-6), f"Identity mismatch:\nGot:      {output[:8]}\nExpected: {x_in[:8]}"
