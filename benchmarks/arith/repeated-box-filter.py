# Repeated box filter: apply a box filter three times consecutively to approximate a Gaussian filter.
# Each pass uses a summed area table (SAT) via the inclusion-exclusion principle.

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D
from vecrec.optimize import get_candidate_schedules
from vecrec.util import *


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def diff_kernel(D: int) -> TIKernel2D:
    kernel = [[0.0] * D for i in range(D)]
    kernel[0][0] = kernel[D - 1][D - 1] = 1.0
    kernel[D - 1][0] = kernel[0][D - 1] = -1.0
    return TIKernel2D(kernel)


def naive_box_filter(input_signal: SignalExpr2D, D: int) -> SignalExpr:
    kernel = diff_kernel(D)

    def box_filter_sat(input_signal: SignalExpr2D, D: int) -> SignalExpr2D:
        with with_types(Type.Arith, ElementType.Float):
            df = Convolve2D(kernel, input_signal)
            df2 = PointwiseDiv(df, Num(float(D * D)))
            sat = Recurse2D(
                TIKernel2D([[0.0, 1.0], [1.0, -1.0]]),
                df2,
            )
            return sat

    box1 = box_filter_sat(input_signal, D)
    box2 = box_filter_sat(box1, D)
    expr = box_filter_sat(box2, D)
    return expr.at(0)


# def optimized_box_filter(input_signal: SignalExpr2D, D: int) -> SignalExpr:
#     with with_types(Type.Arith, ElementType.Float):
#         kernel = diff_kernel(D)
#         fir = Convolve2D(TIKernel2D(kernel), input_signal)
#         fir = Convolve2D(TIKernel2D(kernel), fir.cache_rows())
#         fir = Convolve2D(TIKernel2D(kernel), fir.cache_rows())

#         sat_kernel = TIKernel2D([[0.0, 1.0], [1.0]])
#         iir = Recurse2D(sat_kernel, fir).at(0)
#         iir = Recurse2D(sat_kernel, fir).at(0)
#         iir = Recurse2D(sat_kernel, fir).at(0)

#         return iir


def optimized_box_filter(input_signal: SignalExpr2D, D: int) -> SignalExpr:
    with with_types(Type.Arith, ElementType.Float):
        # Full 2D difference kernel: rank-1, Factorize2D will split into row + column passes.
        kernel = diff_kernel(D)
        fir: SignalExpr = Convolve2D(kernel, input_signal)
        fir = Convolve2D(kernel, fir.cache_rows())
        fir = Convolve2D(kernel, fir.cache_rows())
        fir = PointwiseDiv(fir, Num(float(D**6)))

        sat_kernel = TIKernel2D([[0.0, 1.0], [1.0, -1.0]])
        iir = Recurse2D(sat_kernel, fir).at(0)
        iir = Recurse2D(sat_kernel, iir).at(0)
        iir = Recurse2D(sat_kernel, iir).at(0)

        return iir


# More optimal: Can do all the four-tap filters first, then do prefix sum. The four-tap filters can be decomposed into x and y and fused together.
# Optimization: push down the FIRs before the IIRs.


for D in [3, 5, 7, 9, 11, 13, 15]:
    print(f"D = {D}")

    with with_types(Type.Arith, ElementType.Float):
        expr = naive_box_filter(Var2D("g"), D)

    results = get_candidate_schedules(expr, 256)

    benchmark_result = generate_and_run_benchmark(CodeGen(cache_cse=True), results, True)
