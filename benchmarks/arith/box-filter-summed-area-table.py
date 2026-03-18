# Box filter can be implemented using summed area table (integral image).
# An 11x11 box filter is computed using the SAT with the inclusion-exclusion principle:
#   box[r][c] = (SAT[r][c] - SAT[r-11][c] - SAT[r][c-11] + SAT[r-11][c-11]) / 121

import numpy as np

from vecrec.codegen import *
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D
from vecrec.optimize import get_candidate_schedules
from vecrec.util import *


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2**(x - 1).bit_length()


# More optimal way to compute: do the four-tap filter first, and then do a prefix sum, because this only needs to cache the last row.
def box_filter_naive(input_signal: SignalExpr, D: int) -> SignalExpr:
    sat = Repeater(lambda f1: Recurse(TIKernel.z(), SAdd(f1.at(0), input_signal)), next_power_of_2(D+1))
    kernel = [[0.] * D for i in range(D)]
    kernel[0][0] = kernel[D-1][D-1] = 1.
    kernel[D-1][0] = kernel[0][D-1] = -1.
    return PointwiseDiv(Convolve2D(TIKernel2D(kernel), sat), Num(float(D * D)))


def box_filter_optimized(input_signal: SignalExpr2D, D: int) -> SignalExpr:
    with with_types(Type.Arith, ElementType.Float):
        # Full 2D difference kernel: A = [1,0,...,-1]^T ⊗ [1,0,...,-1] (rank-1).
        # Factorize2D will automatically split this into separate row and column passes.
        kernel = [[0.0] * D for _ in range(D)]
        kernel[0][0] = kernel[D - 1][D - 1] = 1.0
        kernel[D - 1][0] = kernel[0][D - 1] = -1.0
        fir = Convolve2D(TIKernel2D(kernel), input_signal)

        sat_kernel = TIKernel2D([[0.0, 1.0], [1.0]])
        iir = Recurse2D(sat_kernel, fir).at(0)

        return iir


with with_types(Type.Arith, ElementType.Float):
    for D in [3, 5, 7, 9, 11, 13, 15]:
        print(f"D = {D}")
        expr = box_filter_optimized(Var2D("g", D), D)

        results = get_candidate_schedules(expr, 256)

        benchmark_result = generate_and_run_benchmark(
            CodeGen(cache_cse=True), results, True
        )
