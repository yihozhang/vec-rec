"""Tests for DilateLTVHalfPoints and the WindowMap operator."""

from __future__ import annotations

import numpy as np

from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.codegen import generate_kernel_executable
from vecrec.expr import (
    Recurse,
    TVKernel,
    Var,
    Type,
    Num,
    with_types,
)
from vecrec.expr.signal_ops import WindowMap
from vecrec.transform import DilateLTVHalfPoints, AnnotateLanes, Postorder
from vecrec.util import ElementType


def _reference_ltv2(
    a: np.ndarray, b: np.ndarray, g: np.ndarray
) -> np.ndarray:
    """Python reference: f(x) = a(x)*f(x-1) + b(x)*f(x-2) + g(x)."""
    n = len(g)
    f = np.zeros(n, dtype=np.float32)
    for x in range(n):
        fm1 = f[x - 1] if x >= 1 else 0.0
        fm2 = f[x - 2] if x >= 2 else 0.0
        f[x] = a[x] * fm1 + b[x] * fm2 + g[x]
    return f


def _make_ltv2_expr():
    """Build Recurse(TVKernel([0, a, b]), g) in arith/float mode."""
    with with_types(Type.Arith, ElementType.Float):
        a = Var("a")
        b = Var("b")
        g = Var("g")
        kernel = TVKernel([Num(0.0), a, b])
        expr = Recurse(kernel, g)
    return expr


def test_dilate_ltv_half_points_produces_results() -> None:
    """DilateLTVHalfPoints must produce at least one WindowMap result for a 2nd-order LTV."""
    expr = _make_ltv2_expr()
    results = DilateLTVHalfPoints(max_bits=256).apply_signal(expr)
    wm_results = [r for r in results if isinstance(r, WindowMap)]
    assert len(wm_results) > 0, "DilateLTVHalfPoints produced no WindowMap results"


def test_dilate_ltv_half_points_annotate_lanes() -> None:
    """DilateLTVHalfPoints result survives AnnotateLanes without error."""
    expr = _make_ltv2_expr()
    results = DilateLTVHalfPoints(max_bits=256).apply_signal(expr)
    assert len(results) > 0
    wm = results[0]
    annotated = AnnotateLanes(256).apply_signal(wm)
    assert len(annotated) == 1
    assert annotated[0].lanes is not None


def test_dilate_ltv_half_points_postorder() -> None:
    """DilateLTVHalfPoints result survives Postorder traversal."""
    expr = _make_ltv2_expr()
    results = DilateLTVHalfPoints(max_bits=256).apply_signal(expr)
    assert len(results) > 0
    wm = results[0]
    annotated = AnnotateLanes(256).apply_signal(wm)[0]
    # Postorder with Noop should return the same node
    from vecrec.transform import Noop
    traversed = Postorder(Noop()).apply_signal(annotated)
    assert len(traversed) >= 1


def test_dilate_ltv_half_points_compile() -> None:
    """DilateLTVHalfPoints result compiles to valid C++ without error."""
    expr = _make_ltv2_expr()
    results = DilateLTVHalfPoints(max_bits=256).apply_signal(expr)
    assert len(results) > 0

    annotated = AnnotateLanes(256).apply_signal(results[0])
    assert len(annotated) > 0
    compiled = annotated[0]

    result = generate_and_run_benchmark(
        codegen=CodeGen(),
        exprs=[compiled],
        include_correctness_check=False,
        input_size=256,
        warmup_iterations=1,
        benchmark_iterations=2,
    )
    assert result["return_code"] == 0, (
        f"Compilation/run failed:\nstderr: {result.get('error', '')}\n"
        f"stdout: {result.get('output', '')}"
    )


def test_dilate_ltv_half_points_correctness() -> None:
    """DilateLTVHalfPoints output matches the scalar reference implementation."""
    expr = _make_ltv2_expr()
    results = DilateLTVHalfPoints(max_bits=256).apply_signal(expr)
    assert len(results) > 0

    annotated = AnnotateLanes(256).apply_signal(results[0])
    assert len(annotated) > 0
    compiled = annotated[0]

    runner = generate_kernel_executable(CodeGen(), compiled)

    rng = np.random.default_rng(42)
    n = 128
    # Use small values to avoid catastrophic cancellation in floating point
    a_in = rng.uniform(-0.4, 0.4, n).astype(np.float32)
    b_in = rng.uniform(-0.3, 0.3, n).astype(np.float32)
    g_in = rng.uniform(-1.0, 1.0, n).astype(np.float32)

    output, _, _ = runner.run({"a": a_in, "b": b_in, "g": g_in})
    reference = _reference_ltv2(a_in, b_in, g_in)

    assert output.shape == (n,), f"Shape mismatch: {output.shape} vs {(n,)}"
    assert output.dtype == np.float32
    assert np.allclose(output, reference, atol=1e-4), (
        f"Correctness mismatch:\n"
        f"Got:      {output[:8]}\n"
        f"Expected: {reference[:8]}"
    )
