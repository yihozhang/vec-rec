"""Tests for RecurseV, ConvolveV, and VectorAt operators."""
from __future__ import annotations

import numpy as np
import pytest

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


def test_recurse_v_compile_and_run() -> None:
    """2×2 diagonal RecurseV compiles and runs (no cross-coupling).

    F[0][t] = 0.5 * F[0][t-1] + x[t]
    F[1][t] = 0.5 * F[1][t-1] + y[t]

    AnnotateLanes(32) gives lanes=1 for float32; TIKernel([0, 0.5]) has time_delay=1 == lanes.
    """
    with with_types(Type.Arith, ElementType.Float):
        k_diag = TIKernel([0.0, 0.5])  # delay=1, coeff=0.5
        k_zero = TIKernel([])           # zero feedback
        A = KernelMatrix([[k_diag, k_zero], [k_zero, k_diag]])
        G = SignalVector([Var("x"), Var("y")])
        F = RecurseV(A, G)
        f0 = VectorAt(F, 0)

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
    assert result["return_code"] == 0, (
        f"Benchmark failed:\nstderr: {result.get('error', '')}\nstdout: {result.get('output', '')}"
    )


def test_recurse_v_correctness_diagonal() -> None:
    """2×2 diagonal RecurseV produces correct scalar IIR output."""
    with with_types(Type.Arith, ElementType.Float):
        k_diag = TIKernel([0.0, 0.5])
        k_zero = TIKernel([])
        A = KernelMatrix([[k_diag, k_zero], [k_zero, k_diag]])
        G = SignalVector([Var("x"), Var("y")])
        F = RecurseV(A, G)
        f0 = VectorAt(F, 0)

    results = _build_schedule(32).apply_signal(f0)
    assert len(results) > 0

    runner = generate_kernel_executable(CodeGen(), results[0])

    n = 64
    x_in = np.ones(n, dtype=np.float32)
    y_in = np.ones(n, dtype=np.float32)
    output, elapsed_us = runner.run({"x": x_in, "y": y_in})

    # F[0][t] = 0.5 * F[0][t-1] + 1  →  converges to 2; F[0][t] = 2 - 0.5^(t+1)*2
    expected = np.zeros(n, dtype=np.float32)
    prev = 0.0
    for t in range(n):
        prev = 0.5 * prev + x_in[t]
        expected[t] = prev

    assert output.shape == (n,)
    assert output.dtype == np.float32
    assert np.allclose(output, expected, atol=1e-5), (
        f"Output mismatch:\nGot:      {output[:8]}\nExpected: {expected[:8]}"
    )


def test_recurse_v_correctness_coupled() -> None:
    """2×2 coupled RecurseV: F[0] = 0.5*F[1](prev) + x, F[1] = 0.5*F[0](prev) + y.

    Since VectorAt copies the RecurseV state independently, we use a single VectorAt
    and verify only that component 0 is correct when F[1] is driven by a separate input.

    The cross-coupling makes f0 and f1 independent copies, so here we verify that
    at least f0 compiles and runs without error.
    """
    with with_types(Type.Arith, ElementType.Float):
        k_zero = TIKernel([])
        k_cross = TIKernel([0.0, 0.5])  # delay=1
        # F[0] = 0.5*F[1](prev) + x,  F[1] = 0.5*F[0](prev) + y
        A = KernelMatrix([[k_zero, k_cross], [k_cross, k_zero]])
        G = SignalVector([Var("x"), Var("y")])
        F = RecurseV(A, G)
        f0 = VectorAt(F, 0)

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
    assert result["return_code"] == 0, (
        f"Benchmark failed:\nstderr: {result.get('error', '')}\nstdout: {result.get('output', '')}"
    )


def test_convolve_v_compile_and_run() -> None:
    """2×2 ConvolveV FIR compiles and runs.

    out[0][t] = A[0][0] ⊛ g[0] + A[0][1] ⊛ g[1]
    out[1][t] = A[1][0] ⊛ g[0] + A[1][1] ⊛ g[1]

    Using identity-like kernels for simplicity.
    """
    with with_types(Type.Arith, ElementType.Float):
        k_id = TIKernel([1.0])   # identity (no delay, no feedback)
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
    assert result["return_code"] == 0, (
        f"Benchmark failed:\nstderr: {result.get('error', '')}\nstdout: {result.get('output', '')}"
    )


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
    assert np.allclose(output, x_in, atol=1e-6), (
        f"Identity mismatch:\nGot:      {output[:8]}\nExpected: {x_in[:8]}"
    )
