"""Tests for vectorizing a *1st-order* LTV recurrence.

A 1st-order linear time-varying recurrence

    M(x) = a(x) * M(x-1) + g(x)

is expressed as ``Recurse(TVKernel([0, a]), g)``.  The LTV dilations
(``DilateLTVArith``'s dk/fk tower and ``DilateLTVHalfPoints``'s parallel scan)
are written for the 2nd-order kernel ``[0, a, b]``; a 1st-order kernel is the
special case ``b(x) = 0``.  ``_pad_ltv_first_order`` pads ``[0, a] -> [0, a, 0]``
so these transforms fire, which is what lets the coupled mean recurrence in the
Welford-variance benchmark vectorize instead of running scalar.

These tests confirm the padded path (a) fires, (b) actually vectorizes
(lanes > 1), (c) is numerically correct against a scalar reference, and
(d) is meaningfully faster than the scalar schedule.
"""

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
from vecrec.transform import DilateLTVArith, DilateLTVHalfPoints, AnnotateLanes
from vecrec.util import ElementType, get_native_bits


def _reference_ltv1(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Python reference: f(x) = a(x)*f(x-1) + g(x) (single row, no reset)."""
    n = len(g)
    f = np.zeros(n, dtype=np.float32)
    for x in range(n):
        fm1 = f[x - 1] if x >= 1 else 0.0
        f[x] = a[x] * fm1 + g[x]
    return f


def _make_ltv1_expr() -> Recurse:
    """Build Recurse(TVKernel([0, a]), g): a 1st-order LTV recurrence."""
    with with_types(Type.Arith, ElementType.Float):
        a = Var("a")
        g = Var("g")
        return Recurse(TVKernel([Num(0.0), a]), g)


def test_dilate_ltv_arith_fires_on_first_order() -> None:
    """The dk/fk tower must produce a dilated schedule for a 1st-order kernel
    (i.e. the [0, a] -> [0, a, 0] padding lets it past the len==3 guard)."""
    expr = _make_ltv1_expr()
    results = DilateLTVArith(max_bits=256).apply_signal(expr)
    # results[0] is the identity; anything beyond it is a real dilation.
    assert len(results) > 1, "DilateLTVArith did not fire on a 1st-order LTV recurrence"


def test_dilate_ltv_half_points_fires_on_first_order() -> None:
    """The half-points scan must produce a WindowMap for a 1st-order kernel."""
    expr = _make_ltv1_expr()
    results = DilateLTVHalfPoints(max_bits=256).apply_signal(expr)
    wm_results = [r for r in results if isinstance(r, WindowMap)]
    assert len(wm_results) > 0, "DilateLTVHalfPoints produced no WindowMap for a 1st-order LTV"


def test_first_order_ltv_is_vectorized() -> None:
    """The dilated schedule must be genuinely vectorized (lanes > 1), whereas the
    plain scalar schedule stays at 1 lane."""
    mb = get_native_bits()
    expr = _make_ltv1_expr()

    scalar = AnnotateLanes(mb).apply_signal(expr)[0]
    assert scalar.lanes == 1

    wms = [r for r in DilateLTVHalfPoints(mb).apply_signal(expr) if isinstance(r, WindowMap)]
    assert wms, "no WindowMap produced"
    vec = AnnotateLanes(mb).apply_signal(wms[-1])[0]  # widest K
    assert vec.lanes is not None and vec.lanes > 1, (
        f"dilated schedule was not vectorized (lanes={vec.lanes})"
    )


def test_first_order_ltv_correctness() -> None:
    """The vectorized 1st-order LTV output matches the scalar reference."""
    mb = get_native_bits()
    expr = _make_ltv1_expr()
    wms = [r for r in DilateLTVHalfPoints(mb).apply_signal(expr) if isinstance(r, WindowMap)]
    assert wms, "no WindowMap produced"
    compiled = AnnotateLanes(mb).apply_signal(wms[-1])[0]

    runner = generate_kernel_executable(CodeGen(), compiled)
    assert compiled.lanes is not None

    rng = np.random.default_rng(0)
    n = 8 * (2 * compiled.lanes)  # multiple of the 2K outputs/iteration
    # |a| < 1 keeps the recurrence contractive (bounded, no float blow-up).
    a_in = rng.uniform(-0.9, 0.9, n).astype(np.float32)
    g_in = rng.uniform(-1.0, 1.0, n).astype(np.float32)

    output, _, _ = runner.run({"a": a_in, "g": g_in})
    reference = _reference_ltv1(a_in, g_in)

    assert output.shape == (n,)
    assert output.dtype == np.float32
    assert np.allclose(output, reference, atol=1e-4), (
        f"Correctness mismatch:\nGot:      {output[:8]}\nExpected: {reference[:8]}"
    )


def test_first_order_ltv_speedup() -> None:
    """The vectorized schedule must compile, match the scalar reference, and be
    meaningfully faster than the scalar schedule on a large input."""
    mb = get_native_bits()
    expr = _make_ltv1_expr()

    scalar = AnnotateLanes(mb).apply_signal(expr)[0]
    wms = [r for r in DilateLTVHalfPoints(mb).apply_signal(expr) if isinstance(r, WindowMap)]
    assert wms, "no WindowMap produced"
    vec = AnnotateLanes(mb).apply_signal(wms[-1])[0]

    # k0 = scalar (reference), k1 = vectorized. The harness cross-checks k1 vs k0.
    result = generate_and_run_benchmark(
        codegen=CodeGen(),
        exprs=[scalar, vec],
        include_correctness_check=True,
        input_size=1 << 20,
        warmup_iterations=5,
        benchmark_iterations=50,
    )
    assert result["return_code"] == 0, (
        f"benchmark failed:\nstderr: {result.get('error', '')}\n"
        f"stdout: {result.get('output', '')}"
    )

    out = result["output"]
    assert out.get("validation", {}).get("k1") is True, (
        f"vectorized output disagrees with scalar reference: {out.get('validation')}"
    )

    scalar_mbs = out["k0"]
    vec_mbs = out["k1"]
    speedup = vec_mbs / scalar_mbs
    # Conservative bar to stay robust to timing noise / narrower SIMD; the
    # vectorized schedule should clearly beat the latency-bound scalar chain.
    assert speedup > 1.2, (
        f"expected a meaningful speedup, got {speedup:.2f}x "
        f"(scalar={scalar_mbs:.0f} MB/s, vec={vec_mbs:.0f} MB/s)"
    )
