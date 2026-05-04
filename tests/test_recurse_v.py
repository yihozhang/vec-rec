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
    TVKernel,
    Num,
    Var,
    Type,
    with_types,
)
from vecrec.expr.kernel import TVKernel as _TVKernel
from vecrec.expr.pretty import pps
from vecrec.expr.signal_ops import SAdd
from vecrec.expr.signal_vec import RecurseV as _RecurseV
from vecrec.optimize import get_candidate_schedules, get_optimizing_schedule
from vecrec.transform import Transform
from vecrec.util import ElementType


def _build_schedule(max_bits: int = 256) -> Transform:
    return get_optimizing_schedule(max_bits, debug=False)


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


def _make_ltv_mutual_recurse():
    """2×2 first-order LTV mutual recurrence:
        f(x) = a11(x)*f(x-1) + a12(x)*g(x-1) + h1(x)
        g(x) = a21(x)*f(x-1) + a22(x)*g(x-1) + h2(x)
    """
    with with_types(Type.Arith, ElementType.Float):
        zero = Num(0.0)
        a11, a12 = Var("a11"), Var("a12")
        a21, a22 = Var("a21"), Var("a22")
        h1, h2 = Var("h1"), Var("h2")
        A = KernelMatrix([
            [TVKernel([zero, a11, zero]), TVKernel([zero, a12, zero])],
            [TVKernel([zero, a21, zero]), TVKernel([zero, a22, zero])],
        ])
        G = SignalVector([h1, h2])
        F = RecurseV(A, G)
        f0 = VectorAt(F, 0)
        f1 = VectorAt(F, 1)
    return f0, f1


def _reference_ltv_mutual(
    a11: np.ndarray,
    a12: np.ndarray,
    a21: np.ndarray,
    a22: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(h1)
    f = np.zeros(n, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    for x in range(n):
        fm1 = f[x - 1] if x >= 1 else 0.0
        gm1 = g[x - 1] if x >= 1 else 0.0
        f[x] = a11[x] * fm1 + a12[x] * gm1 + h1[x]
        g[x] = a21[x] * fm1 + a22[x] * gm1 + h2[x]
    return f, g

def _max_recurse_v_tap_len(expr: object, _visited: set[int] | None = None) -> int:
    """Return the max TVKernel length found in any RecurseV node in the expression tree."""
    from vecrec.expr.base import RecLang
    if _visited is None:
        _visited = set()
    expr_id = id(expr)
    if expr_id in _visited:
        return 0
    _visited.add(expr_id)

    max_len = 0
    if isinstance(expr, _RecurseV):
        assert isinstance(expr.a, KernelMatrix)
        for row in expr.a.data:
            for entry in row:
                if isinstance(entry, _TVKernel):
                    max_len = max(max_len, len(entry.data))

    if isinstance(expr, RecLang):
        for child in expr.children():
            max_len = max(max_len, _max_recurse_v_tap_len(child, _visited))

    return max_len


def test_recurse_v_mutual_compile_and_run() -> None:
    """Mutual RecurseV compiles and runs without error."""
    f0, _ = _make_mutual_recurse()
    results = _build_schedule(256).apply_signal(f0)
    print(pps(results))
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
    """Every codegen-supported schedule for VectorAt(F, 0) produces the correct f component."""
    f0, _ = _make_mutual_recurse()
    results = _build_schedule(256).apply_signal(f0)
    assert len(results) > 0
    runnable = results
    assert len(runnable) > 0

    n = 64
    h1 = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    h2 = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    f_ref, _ = _reference_mutual(h1, h2)

    for idx, sched in enumerate(runnable):
        runner = generate_kernel_executable(CodeGen(), sched)
        output, _ = runner.run({"h1": h1, "h2": h2})
        assert output.shape == (n,)
        assert output.dtype == np.float32
        assert np.allclose(output, f_ref, atol=1e-5), (
            f"Schedule {idx}: f component mismatch\nGot:      {output[:8]}\nExpected: {f_ref[:8]}"
        )


def test_recurse_v_mutual_g_correctness() -> None:
    """Every codegen-supported schedule for VectorAt(F, 1) produces the correct g component."""
    _, f1 = _make_mutual_recurse()
    results = _build_schedule(256).apply_signal(f1)
    assert len(results) > 0
    runnable = results
    assert len(runnable) > 0

    n = 64
    h1 = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    h2 = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    _, g_ref = _reference_mutual(h1, h2)

    for idx, sched in enumerate(runnable):
        runner = generate_kernel_executable(CodeGen(), sched)
        output, _ = runner.run({"h1": h1, "h2": h2})
        assert output.shape == (n,)
        assert output.dtype == np.float32
        assert np.allclose(output, g_ref, atol=1e-5), (
            f"Schedule {idx}: g component mismatch\nGot:      {output[:8]}\nExpected: {g_ref[:8]}"
        )


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

    results = _build_schedule(256).apply_signal(out0)
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
    """Every codegen-supported schedule for identity ConvolveV passes x through unchanged."""
    with with_types(Type.Arith, ElementType.Float):
        k_id = TIKernel([1.0])
        k_zero = TIKernel([])
        A = KernelMatrix([[k_id, k_zero], [k_zero, k_id]])
        G = SignalVector([Var("x"), Var("y")])
        out = ConvolveV(A, G)
        out0 = VectorAt(out, 0)

    results = _build_schedule(256).apply_signal(out0)
    assert len(results) > 0
    runnable = results
    assert len(runnable) > 0

    n = 64
    x_in = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    y_in = np.random.uniform(-1.0, 1.0, n).astype(np.float32)

    for idx, sched in enumerate(runnable):
        runner = generate_kernel_executable(CodeGen(), sched)
        output, _ = runner.run({"x": x_in, "y": y_in})
        assert output.shape == (n,)
        assert np.allclose(output, x_in, atol=1e-6), (
            f"Schedule {idx}: identity mismatch\nGot:      {output[:8]}\nExpected: {x_in[:8]}"
        )


def test_recurse_convolve_v() -> None:
    with with_types(Type.Arith, ElementType.Float):
        A = KernelMatrix([[TIKernel([0.5, 1.]), TIKernel([0.25, -0.5])], [TIKernel([0.25, 0.75]), TIKernel([0.5, -0.5])]])
        G = SignalVector([Var("h1"), Var("h2")])
        H = ConvolveV(A, G)
        B = KernelMatrix([[TIKernel([0., 0.75]), TIKernel([0., -0.75])], [TIKernel([0., 0.75]), TIKernel([0., -0.75])]])
        F = RecurseV(B, H)
        f0 = VectorAt(F, 0)
        f1 = VectorAt(F, 1)
        expr = SAdd(f0, f1)

    results = _build_schedule(256).apply_signal(expr)
    assert len(results) > 0
    runnable = results
    assert len(runnable) > 0

    def reference(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
        n = len(h1)
        f_ref = np.zeros(n, dtype=np.float32)
        g_ref = np.zeros(n, dtype=np.float32)
        for t in range(n):
            h1m0 = h1[t]
            h1m1 = h1[t-1] if t-1 >= 0 else 0.0
            h2m0 = h2[t]
            h2m1 = h2[t-1] if t-1 >= 0 else 0.0
            h1_conv = 0.5 * h1m0 + 1.0 * h1m1 + 0.25 * h2m0 - 0.5 * h2m1
            h2_conv = 0.25 * h1m0 + 0.75 * h1m1 + 0.5 * h2m0 - 0.5 * h2m1
            f_prev = f_ref[t-1] if t-1 >= 0 else 0.0
            g_prev = g_ref[t-1] if t-1 >= 0 else 0.0
            f_ref[t] = 0.75 * f_prev - 0.75 * g_prev + h1_conv
            g_ref[t] = 0.75 * f_prev - 0.75 * g_prev + h2_conv
        return f_ref + g_ref

    n = 64
    h1_in = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    h2_in = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    solution = reference(h1_in, h2_in)

    for idx, sched in enumerate(runnable):
        runner = generate_kernel_executable(CodeGen(), sched)
        output, _ = runner.run({"h1": h1_in, "h2": h2_in})
        assert output.shape == (n,)
        assert np.allclose(output, solution, atol=1e-6), (
            f"Schedule {idx}: mismatch\nGot:      {output[:8]}\nExpected: {solution[:8]}"
        )


def test_ltv_mutual_all_schedules_correct() -> None:
    """Every schedule for a 2×2 first-order LTV mutual recurrence produces correct output."""
    f0, f1 = _make_ltv_mutual_recurse()
    schedules_f0 = get_candidate_schedules(f0, max_bits=64)
    schedules_f1 = get_candidate_schedules(f1, max_bits=64)
    assert len(schedules_f0) > 0
    assert len(schedules_f1) > 0

    rng = np.random.default_rng(7)
    n = 32
    a11 = rng.uniform(-0.3, 0.3, n).astype(np.float32)
    a12 = rng.uniform(-0.3, 0.3, n).astype(np.float32)
    a21 = rng.uniform(-0.3, 0.3, n).astype(np.float32)
    a22 = rng.uniform(-0.3, 0.3, n).astype(np.float32)
    h1 = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    h2 = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    f_ref, g_ref = _reference_ltv_mutual(a11, a12, a21, a22, h1, h2)
    inputs = {"a11": a11, "a12": a12, "a21": a21, "a22": a22, "h1": h1, "h2": h2}

    runnable_f0 = schedules_f0
    runnable_f1 = schedules_f1
    assert len(runnable_f0) > 0, "No runnable schedules for f0"
    assert len(runnable_f1) > 0, "No runnable schedules for f1"

    for idx, sched in enumerate(runnable_f0):
        runner = generate_kernel_executable(CodeGen(), sched)
        output, _ = runner.run(inputs)
        assert output.shape == (n,), f"Schedule {idx} (f0): wrong shape {output.shape}"
        assert np.allclose(output, f_ref, atol=1e-4), (
            f"Schedule {idx} (f0): mismatch\nGot:      {output[:4]}\nExpected: {f_ref[:4]}"
        )

    for idx, sched in enumerate(runnable_f1):
        runner = generate_kernel_executable(CodeGen(), sched)
        output, _ = runner.run(inputs)
        assert output.shape == (n,), f"Schedule {idx} (f1): wrong shape {output.shape}"
        assert np.allclose(output, g_ref, atol=1e-4), (
            f"Schedule {idx} (f1): mismatch\nGot:      {output[:4]}\nExpected: {g_ref[:4]}"
        )


def test_ltv_mutual_has_dilated_schedule() -> None:
    """At least one schedule for the LTV mutual recurrence has a RecurseV with farther-away taps.

    DilateLTVArithVec with K=2 produces TVKernel entries of length 4 (taps at lags 2
    and 3), compared to the original length-3 kernels (tap at lag 1).
    """
    f0, _ = _make_ltv_mutual_recurse()
    schedules = get_candidate_schedules(f0, max_bits=256)
    assert any(_max_recurse_v_tap_len(s) >= 4 for s in schedules), (
        "No dilated schedule found: expected at least one RecurseV "
        "with TVKernel entries of length >= 4 (stride-2 or higher dilation)"
    )
