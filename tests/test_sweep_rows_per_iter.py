"""Tests for sweep_codegen_setup's rows_per_iter restriction.

Row blocking (rows_per_iter > 1) is only valid when every input signal is 2D.
A 1D Var lowers to Signal1D, which has no block-row notion: replicating it
across block-rows would cause each replica to re-read the same offset and
produce duplicate output (and a misleadingly fast timing because the C++
compiler fuses the duplicate work).
"""

from __future__ import annotations

import pytest

from vecrec.codegen import CodeGen, sweep_codegen_setup
from vecrec.expr.base import SignalExpr, Type, with_types
from vecrec.expr.kernel import TIKernel, TIKernel2D
from vecrec.expr.signal import Var, Var2D
from vecrec.expr.signal_ops import Convolve2D, Recurse, Recurse2D
from vecrec.transform import AnnotateLanes, Eliminate2DKernels, InferNRows, Preorder, Try
from vecrec.util import ElementType, get_native_bits


def _swept_rpis(result: dict) -> set:
    """Return the set of rpi values that produced a successful timing."""
    return {rpi for (_, rpi), t in result["throughputs_mbs"].items() if t is not None}


def test_sweep_rpi_only_1_for_1d_recurrence() -> None:
    """f(x) = f(x-1) + g(x) — a 1D recurrence whose input g is a 1D Var.

    The sweep must restrict rpi to {1}, because rpi > 1 would build k
    independent Signal1D readers that all start at the same offset and
    produce duplicate output.
    """
    with with_types(Type.Arith, ElementType.Float):
        expr: SignalExpr = Recurse(TIKernel([0.0, 1.0]), Var("g"))
    expr = AnnotateLanes(get_native_bits()).apply_generic(expr)[0]

    result = sweep_codegen_setup(
        [expr],
        input_size=1 << 12,
        warmup_iterations=1,
        benchmark_iterations=2,
    )[0]
    rpis = _swept_rpis(result)
    assert rpis == {1}, f"Expected only rpi=1 for 1D recurrence, got {rpis}"
    assert result["best_rows_per_iter"] == 1


def test_sweep_rpi_gt1_for_2d_recurrence() -> None:
    """A 2D recurrence on f(x, y-1) whose only input is a Var2D.

    The sweep should try rpi > 1 because every block-row has its own correct
    InputSignal2D offset. The kernel below carries the spirit of
    f(x, y) = f(x, y-1) + g(x, y) but also depends on f(x-1, y), since codegen
    requires at least one within-row tap to lower SRecurse cleanly.
    """
    with with_types(Type.Arith, ElementType.Float):
        # Row 0 (y=0) tap on f(x-1, y); row 1 (y=-1) tap on f(x, y-1).
        kernel = TIKernel2D([[0.0, 1.0], [1.0, 0.0]])
        g2d = Var2D("g")
        expr = Recurse2D(kernel, Convolve2D(TIKernel2D([[1.0]]), g2d)).at(0)

    expr = Preorder(Try(Eliminate2DKernels())).apply_generic(expr)[0]
    expr = InferNRows().apply_generic(expr)[0]
    expr = AnnotateLanes(get_native_bits()).apply_generic(expr)[0]

    result = sweep_codegen_setup(
        [expr],
        input_size=1 << 12,
        warmup_iterations=1,
        benchmark_iterations=2,
    )[0]
    rpis = _swept_rpis(result)
    assert rpis == {1, 2, 4}, f"Expected rpi sweep {{1, 2, 4}} for 2D recurrence, got {rpis}"


def test_codegen_rejects_rpi_gt1_with_1d_var() -> None:
    """CodeGen.generate must refuse rpi > 1 when the expression has 1D Vars.

    This is the defense-in-depth backstop: even if someone constructs a
    CodeGen by hand with the wrong rpi, they get a clear ValueError instead
    of a silently-wrong kernel.
    """
    with with_types(Type.Arith, ElementType.Float):
        expr: SignalExpr = Recurse(TIKernel([0.0, 1.0]), Var("g"))
    expr = AnnotateLanes(get_native_bits()).apply_generic(expr)[0]

    with pytest.raises(ValueError, match="rows_per_iter"):
        CodeGen(rows_per_iter=2).generate(expr, "k0")


if __name__ == "__main__":
    test_sweep_rpi_only_1_for_1d_recurrence()
    test_sweep_rpi_gt1_for_2d_recurrence()
    test_codegen_rejects_rpi_gt1_with_1d_var()
    print("All sweep rpi tests passed.")
