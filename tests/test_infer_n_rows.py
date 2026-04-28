"""Tests for InferNRows: automatic inference of n_rows for Var2D and Repeater."""

from vecrec.expr.base import SignalExpr, Type, with_types
from vecrec.expr.kernel import TIKernel2D
from vecrec.expr.signal import Var, Var2D
from vecrec.expr.signal_ops import Convolve2D, Ith, Repeater, SAdd
from vecrec.util import ElementType, get_native_bits
from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.optimize import get_candidate_schedules
from vecrec.transform import AnnotateLanes, InferNRows, Preorder, Eliminate2DKernels, Try


def test_infer_var2d_from_ith() -> None:
    """Var2D with no n_rows gets n_rows=2 from Ith(var2d, 1)."""
    with with_types(Type.Arith, ElementType.Float):
        var2d = Var2D("img")
        assert var2d.n_rows is None
        expr: SignalExpr = Ith(var2d, 1)

    InferNRows().apply_generic(expr)
    assert var2d.n_rows == 2


def test_infer_repeater_via_rvar2d() -> None:
    """Repeater.n_rows inferred from Ith(rvar, i): max(i+2) over all accesses."""
    with with_types(Type.Arith, ElementType.Float):
        # Ith(rvar, 0) → n_rows >= 2; Ith(rvar, 1) → n_rows >= 3; max = 3
        repeater = Repeater(lambda r: SAdd(Ith(r, 0), Ith(r, 1)))
        assert repeater.n_rows is None

    InferNRows().apply_generic(repeater)
    assert repeater.n_rows == 3


def test_infer_repeater_via_outer_ith() -> None:
    """Repeater.n_rows inferred from Ith(repeater, i): max(i+1)."""
    with with_types(Type.Arith, ElementType.Float):
        repeater = Repeater(lambda _: Var("x"))
        expr: SignalExpr = Ith(repeater, 2)

    InferNRows().apply_generic(expr)
    assert repeater.n_rows == 3


def test_infer_via_convolve2d_eliminate() -> None:
    """Var2D n_rows inferred after Eliminate2DKernels expands Convolve2D into Ith nodes."""
    with with_types(Type.Arith, ElementType.Float):
        kernel_data = [[1.0, 0.0], [0.0, 1.0]]
        kernel2d = TIKernel2D(kernel_data)
        var2d = Var2D("img")
        expr: SignalExpr = Convolve2D(kernel2d, var2d)
        assert var2d.n_rows is None

    # Before Eliminate2DKernels: no Ith nodes, so inference would fail
    # After: Ith(var2d, 0) and Ith(var2d, 1) are present → n_rows = 2
    expr = Preorder(Try(Eliminate2DKernels())).apply_generic(expr)[0]
    InferNRows().apply_generic(expr)
    assert var2d.n_rows == 2

    expr = AnnotateLanes(get_native_bits()).apply_generic(expr)[0]
    result = generate_and_run_benchmark(CodeGen(), [expr])
    assert result["return_code"] == 0, result.get("error")


def test_infer_end_to_end_via_get_candidate_schedules() -> None:
    """Var2D and Repeater n_rows inferred end-to-end through get_candidate_schedules."""
    with with_types(Type.Arith, ElementType.Float):
        kernel_data = [[1.0, 0.0], [0.5, 0.5]]
        kernel2d = TIKernel2D(kernel_data)
        var2d = Var2D("img")
        expr: SignalExpr = Convolve2D(kernel2d, var2d)

    results = get_candidate_schedules(expr, get_native_bits())
    assert len(results) > 0
    result = generate_and_run_benchmark(CodeGen(), results)
    assert result["return_code"] == 0, result.get("error")
