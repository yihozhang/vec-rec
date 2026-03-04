# Repeated box filter: apply a box filter three times consecutively to approximate a Gaussian filter.
# Each pass uses a summed area table (SAT) via the inclusion-exclusion principle.

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D
from vecrec.optimize import get_candidate_schedules
from vecrec.util import *


def optimized_box_filter(input_signal: SignalExpr2D, D: int) -> SignalExpr:
    with with_types(Type.Arith, ElementType.Float):
        kernel_x = [1.0] + [0.0] * (D - 2) + [-1.0]
        kernel_y = [[1.0], [0.0] * (D - 2) + [-1.0]]
        fir_y = Convolve2D(TIKernel2D(kernel_y), input_signal)
        fir_y = Convolve2D(TIKernel2D(kernel_y), fir_y.cache_rows(D))
        fir_y = Convolve2D(TIKernel2D(kernel_y), fir_y.cache_rows(D))
        fir: SignalExpr = Convolve(TIKernel(kernel_x), fir_y)
        fir = Convolve(TIKernel(kernel_x), fir)
        fir = Convolve(TIKernel(kernel_x), fir)
        fir = PointwiseDiv(fir, Num(float(D ** 6)))

        sat_kernel = TIKernel2D([[0.0, 1.0], [1.0]])
        iir = Recurse2D(sat_kernel, fir).at(0)
        iir = Recurse2D(sat_kernel, iir).at(0)
        iir = Recurse2D(sat_kernel, iir).at(0)

        return iir


for D in [3, 5, 7, 9, 11, 13, 15]:
    print(f"D = {D}")

    with with_types(Type.Arith, ElementType.Float):
        expr = optimized_box_filter(Var2D("g", D), D)

    results = get_candidate_schedules(expr, 256)

    benchmark_result = generate_and_run_benchmark(
        CodeGen(cache_cse=True), results, True
    )
