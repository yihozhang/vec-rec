"""Tests for tuple (multi-output) kernel compilation.

A SignalVectorExpr root (SignalVector, RecurseV, ...) compiles into a single
kernel with n output streams: kernel.run(out) writes component c of block-row
br into out[br * n_outputs + c].
"""

from __future__ import annotations

import numpy as np

from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.codegen import generate_kernel_executable
from vecrec.expr import (
    KernelMatrix,
    Num,
    Recurse,
    RecurseV,
    SignalVector,
    TIKernel,
    TVKernel,
    Type,
    Var,
    with_types,
)
from vecrec.optimize import get_candidate_schedules
from vecrec.util import ElementType


def _make_ltv_mutual_recurse_v() -> RecurseV:
    """2x2 first-order LTV mutual recurrence as a tuple-output root:
        f(x) = a11(x)*f(x-1) + a12(x)*g(x-1) + h1(x)
        g(x) = a21(x)*f(x-1) + a22(x)*g(x-1) + h2(x)
    """
    with with_types(Type.Arith, ElementType.Float):
        zero = Num(0.0)
        A = KernelMatrix([
            [TVKernel([zero, Var("a11"), zero]), TVKernel([zero, Var("a12"), zero])],
            [TVKernel([zero, Var("a21"), zero]), TVKernel([zero, Var("a22"), zero])],
        ])
        return RecurseV(A, SignalVector([Var("h1"), Var("h2")]))


def _reference_ltv_mutual(
    a11: np.ndarray, a12: np.ndarray, a21: np.ndarray, a22: np.ndarray,
    h1: np.ndarray, h2: np.ndarray, row_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(h1)
    f = np.zeros(n, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    for x in range(n):
        fm1 = f[x - 1] if x % row_size >= 1 else 0.0
        gm1 = g[x - 1] if x % row_size >= 1 else 0.0
        f[x] = a11[x] * fm1 + a12[x] * gm1 + h1[x]
        g[x] = a21[x] * fm1 + a22[x] * gm1 + h2[x]
    return f, g


def test_recurse_v_root_all_schedules_correct() -> None:
    """Every schedule of a RecurseV root produces both components correctly."""
    F = _make_ltv_mutual_recurse_v()
    schedules = get_candidate_schedules(F, max_bits=128)
    assert len(schedules) > 0

    rng = np.random.default_rng(7)
    n, row_size = 2048, 1024
    a11, a12, a21, a22 = (rng.uniform(-0.3, 0.3, n).astype(np.float32) for _ in range(4))
    h1, h2 = (rng.uniform(-1.0, 1.0, n).astype(np.float32) for _ in range(2))
    f_ref, g_ref = _reference_ltv_mutual(a11, a12, a21, a22, h1, h2, row_size)
    inputs = {"a11": a11, "a12": a12, "a21": a21, "a22": a22, "h1": h1, "h2": h2}

    for idx, sched in enumerate(schedules):
        runner = generate_kernel_executable(CodeGen(), sched, row_size=row_size)
        assert runner.n_outputs == 2
        output, _, _ = runner.run(inputs, n=n)
        assert output.shape == (2, n), f"Schedule {idx}: wrong shape {output.shape}"
        assert np.allclose(output[0], f_ref, atol=1e-4), f"Schedule {idx}: f component mismatch"
        assert np.allclose(output[1], g_ref, atol=1e-4), f"Schedule {idx}: g component mismatch"


def test_signal_vector_tuple_of_independent_recurrences() -> None:
    """SignalVector tuples arbitrary independent expressions into one kernel."""
    with with_types(Type.Arith, ElementType.Float):
        f0 = Recurse(TIKernel([0.0, 0.8]), Var("x"))
        f1 = Recurse(TIKernel([0.0, -0.5, 0.3]), Var("y"))
        tup = SignalVector([f0, f1])

    schedules = get_candidate_schedules(tup, max_bits=128)
    assert len(schedules) > 0

    rng = np.random.default_rng(3)
    n, row_size = 2048, 1024
    x = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    y = rng.uniform(-1.0, 1.0, n).astype(np.float32)

    f0_ref = np.zeros(n, dtype=np.float32)
    f1_ref = np.zeros(n, dtype=np.float32)
    for t in range(n):
        f0m1 = f0_ref[t - 1] if t % row_size >= 1 else 0.0
        f1m1 = f1_ref[t - 1] if t % row_size >= 1 else 0.0
        f1m2 = f1_ref[t - 2] if t % row_size >= 2 else 0.0
        f0_ref[t] = 0.8 * f0m1 + x[t]
        f1_ref[t] = -0.5 * f1m1 + 0.3 * f1m2 + y[t]

    # The cartesian product of per-component schedules can be large; checking a
    # sample that includes the reference and the most-transformed schedule is enough.
    picked = {0, len(schedules) // 2, len(schedules) - 1}
    for idx in sorted(picked):
        runner = generate_kernel_executable(CodeGen(), schedules[idx], row_size=row_size)
        assert runner.n_outputs == 2
        output, _, _ = runner.run({"x": x, "y": y}, n=n)
        assert output.shape == (2, n), f"Schedule {idx}: wrong shape {output.shape}"
        assert np.allclose(output[0], f0_ref, atol=1e-4), f"Schedule {idx}: component 0 mismatch"
        assert np.allclose(output[1], f1_ref, atol=1e-4), f"Schedule {idx}: component 1 mismatch"


def test_tuple_output_benchmark_compiles_and_validates() -> None:
    """Tuple-output kernels run through the benchmark harness, and the
    cross-schedule correctness check covers all components."""
    with with_types(Type.Arith, ElementType.Float):
        # TI mutual recurrence with stable coefficients (validation-friendly).
        A = KernelMatrix(
            [
                [TIKernel([0.0, 0.6]), TIKernel([0.0, 0.2])],
                [TIKernel([0.0, 0.3]), TIKernel([0.0, 0.0, 0.4])],
            ]
        )
        F = RecurseV(A, SignalVector([Var("h1"), Var("h2")]))

    schedules = get_candidate_schedules(F, max_bits=128)
    assert len(schedules) > 0

    result = generate_and_run_benchmark(
        codegen=CodeGen(),
        exprs=schedules[:4],
        include_correctness_check=True,
        input_size=4096,
        warmup_iterations=1,
        benchmark_iterations=2,
    )
    assert result["return_code"] == 0, f"Benchmark failed: {result.get('error', '')}"
    validation = result["output"].get("validation", {})
    for name, ok in validation.items():
        assert ok is True, f"Schedule {name} diverged from k0 at index {ok}"
