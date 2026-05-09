"""Tests for 2D operator blocking (k rows per kernel.run call)."""

from __future__ import annotations

from typing import Callable, Tuple, Optional

import numpy as np

from vecrec.expr.base import SignalExpr, Type, with_types
from vecrec.expr.kernel import TIKernel2D
from vecrec.expr.signal import Var2D
from vecrec.expr.signal_ops import (
    Convolve2D,
    Recurse2D,
)
from vecrec.util import ElementType, get_native_bits
from vecrec.codegen import CodeGen, generate_kernel_executable
from vecrec.transform import (
    AnnotateLanes,
    InferNRows,
    Preorder,
    Eliminate2DKernels,
    Try,
)


def _build_recurse2d_2x2() -> SignalExpr:
    """A 2D recurrence with a 2×2 TIKernel2D, lowered for codegen.

    After Eliminate2DKernels this becomes Repeater(...) with n_rows=2.
    """
    with with_types(Type.Arith, ElementType.Float):
        kernel_data = [
            [0.0, 0.5],
            [0.4, 0.1],
        ]
        sat_kernel = TIKernel2D(kernel_data)
        var2d = Var2D("img")
        # Recurse2D is a 2D signal; .at(0) selects the current row.
        expr: SignalExpr = Recurse2D(sat_kernel, Convolve2D(TIKernel2D([[1.0]]), var2d)).at(0)

    expr = Preorder(Try(Eliminate2DKernels())).apply_generic(expr)[0]
    expr = InferNRows().apply_generic(expr)[0]
    expr = AnnotateLanes(get_native_bits()).apply_generic(expr)[0]
    return expr


def _build_recurse2d_3x3() -> SignalExpr:
    """A 2D recurrence with a 3×3 TIKernel2D — n_rows=3 after lowering."""
    with with_types(Type.Arith, ElementType.Float):
        kernel_data = [
            [0.0, 0.3, 0.1],
            [0.2, 0.05, 0.0],
            [0.1, 0.0, 0.0],
        ]
        sat_kernel = TIKernel2D(kernel_data)
        var2d = Var2D("img")
        expr: SignalExpr = Recurse2D(sat_kernel, Convolve2D(TIKernel2D([[1.0]]), var2d)).at(0)

    expr = Preorder(Try(Eliminate2DKernels())).apply_generic(expr)[0]
    expr = InferNRows().apply_generic(expr)[0]
    expr = AnnotateLanes(get_native_bits()).apply_generic(expr)[0]
    return expr


def _make_input(n_rows: int, row_size: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, row_size)).astype(np.float32)


def _run_blocked_vs_baseline(
    builder: Callable[[], SignalExpr],
    rows_per_iter: Optional[int],
    *,
    n_input_rows: int = 16,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build two copies of the same expression, compile one without blocking
    and one with the requested block size, and return both outputs on the same
    input."""
    expr_baseline = builder()
    expr_blocked = builder()

    runner_baseline = generate_kernel_executable(CodeGen(rows_per_iter=0), expr_baseline)
    runner_blocked = generate_kernel_executable(CodeGen(rows_per_iter=rows_per_iter), expr_blocked)

    inp = _make_input(n_input_rows, seed=seed)
    out_baseline, _ = runner_baseline.run(inp)
    out_blocked, _ = runner_blocked.run(inp)
    return out_baseline, out_blocked


def test_blocking_basic_k2_n2() -> None:
    """k=2, n_rows=2: standard case where every block-row writes to the cache.

    Both block-rows pass `block_row >= block_size - n_rows` (0 >= 0, 1 >= 0),
    so the skip-write optimisation is inactive — but Mov overrides still
    forward block-row 0's body output into block-row 1's reads.
    """
    out_baseline, out_blocked = _run_blocked_vs_baseline(
        _build_recurse2d_2x2, rows_per_iter=2
    )
    assert np.allclose(out_baseline, out_blocked, atol=1e-3, rtol=1e-3), (
        f"Mismatch: max diff={np.max(np.abs(out_baseline - out_blocked))}"
    )


def test_blocking_skip_writes_k4_n2() -> None:
    """k=4, n_rows=2: max(k-n, 0)=2 block-rows skip the cache write.

    Block-rows 0 and 1 do not store their outputs in RepeaterContext (their
    values are only consumed within the block via Mov-forwarded registers).
    Block-rows 2 and 3 do write — block-row 3's value becomes the most recent
    cached row that the next block reads back.
    """
    out_baseline, out_blocked = _run_blocked_vs_baseline(
        _build_recurse2d_2x2, rows_per_iter=4, n_input_rows=16
    )
    assert np.allclose(out_baseline, out_blocked, atol=1e-3, rtol=1e-3), (
        f"Mismatch: max diff={np.max(np.abs(out_baseline - out_blocked))}"
    )


def test_blocking_auto_detect_k2() -> None:
    """Default CodeGen() should auto-enable k=2 blocking when the expression
    contains a Repeater (Recurse2D lowers to Repeater)."""
    expr = _build_recurse2d_2x2()
    cg = CodeGen()
    program = cg.generate(expr, "k0")
    assert program.rows_per_iter == 2, f"Expected auto k=2, got {program.rows_per_iter}"


def test_blocking_disabled_for_non_repeater() -> None:
    """Pure Convolve2D over Var2D contains no Repeater after lowering, so
    rows_per_iter should remain 0 (legacy path)."""
    with with_types(Type.Arith, ElementType.Float):
        var2d = Var2D("img")
        expr: SignalExpr = Convolve2D(TIKernel2D([[1.0, 2.0], [3.0, 4.0]]), var2d)
    expr = Preorder(Try(Eliminate2DKernels())).apply_generic(expr)[0]
    expr = InferNRows().apply_generic(expr)[0]
    expr = AnnotateLanes(get_native_bits()).apply_generic(expr)[0]

    cg = CodeGen()
    program = cg.generate(expr, "k0")
    assert program.rows_per_iter == 0


def test_blocking_3x3_k2() -> None:
    """k=2 with n_rows=3 (3×3 recurrence). Block-row 1 reads offsets 1 and 2;
    offset 1 (= block-row 0's output) is forwarded via Mov, offset 2 falls
    through to the cache (previous block's contents)."""
    out_baseline, out_blocked = _run_blocked_vs_baseline(
        _build_recurse2d_3x3, rows_per_iter=2, n_input_rows=16
    )
    assert np.allclose(out_baseline, out_blocked, atol=1e-3, rtol=1e-3), (
        f"Mismatch: max diff={np.max(np.abs(out_baseline - out_blocked))}"
    )


if __name__ == "__main__":
    test_blocking_auto_detect_k2()
    test_blocking_disabled_for_non_repeater()
    test_blocking_basic_k2_n2()
    test_blocking_skip_writes_k4_n2()
    test_blocking_3x3_k2()
    print("All blocking tests passed.")
