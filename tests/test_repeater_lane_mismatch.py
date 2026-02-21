"""Test that Repeater and RVar2D with mismatched lane counts are handled correctly.

When a Repeater contains a Recurse whose kernel time delay limits lanes to r < max_lanes,
PushDownConvertLanes narrows Ith.lanes to r while leaving RVar2D.lanes at max_lanes.
The codegen must wrap make_ith_row with a lane converter (ConvertOne2N) to bridge
the type mismatch, rather than generating make_ith_row<vec_type_r>(make_signal2d<vec_type_max>(...)).
"""

from vecrec.expr.base import SignalExpr, Type
from vecrec.expr.pretty import pp
from vecrec.expr.signal import Var, RVar2D
from vecrec.expr.signal_ops import Convolve, Ith, Recurse, Repeater, SAdd
from vecrec.expr.kernel import TIKernel
from vecrec.util import ElementType
from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.transform import AnnotateLanes, PushDownConvertLanes


def test_repeater_recurse_lane_mismatch() -> None:
    """Repeater containing a Recurse whose lanes < max_lanes (due to kernel time delay).

    With 512-bit SIMD and Float32: max_lanes = 16.
    The feedback kernel has time delay 4, so Recurse.lanes = 4.
    After AnnotateLanes + PushDownConvertLanes, Ith.lanes = 4 but RVar2D.lanes = 16.
    The codegen fix wraps IthRow with ConvertOne2N to bridge the type mismatch.
    """
    # Kernel with time delay = 4 → Recurse.lanes = 4 when max_lanes = 16
    feedback_kernel = TIKernel([0, 0, 0, 0, 1.8, -0.9], Type.Arith, ElementType.Float)
    row_kernel = TIKernel([1.0], Type.Arith, ElementType.Float)
    g = Var("x", Type.Arith, ElementType.Float)

    def make_inner(rvar: RVar2D) -> SignalExpr:
        prev_row = Ith(rvar, 0)
        return Recurse(feedback_kernel, SAdd(g, Convolve(row_kernel, prev_row)))

    repeater = Repeater(make_inner, n_rows=2, ty=Type.Arith, element_type=ElementType.Float)

    expr: SignalExpr
    expr = Ith(repeater, 0)
    expr = AnnotateLanes(512).apply_generic(expr)[0]
    expr = PushDownConvertLanes().apply_generic(expr)[0]
    print(pp(expr))
    result = generate_and_run_benchmark(CodeGen(), [expr])
    assert result['return_code'] == 0, f"C++ compilation/execution failed: {result}"

def test_repeater_recurse_lane_mismatch2() -> None:
    # Kernel with time delay = 4 → Recurse.lanes = 4 when max_lanes = 16
    feedback_kernel = TIKernel([0, 0, 0, 0, 1.8, -0.9], Type.Arith, ElementType.Float)

    def make_inner(rvar: RVar2D) -> SignalExpr:
        prev_row = Ith(rvar, 0)
        return Recurse(feedback_kernel, prev_row)

    repeater = Repeater(make_inner, n_rows=2, ty=Type.Arith, element_type=ElementType.Float)

    expr: SignalExpr
    expr = Ith(repeater, 0)
    expr = AnnotateLanes(512).apply_generic(expr)[0]
    expr = PushDownConvertLanes().apply_generic(expr)[0]
    print(pp(expr))
    result = generate_and_run_benchmark(CodeGen(), [expr])
    assert result['return_code'] == 0, f"C++ compilation/execution failed: {result}"
