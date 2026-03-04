# Repeated box filter: apply a box filter three times consecutively to approximate a Gaussian filter.
# Each pass uses a summed area table (SAT) via the inclusion-exclusion principle.

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D
from vecrec.optimize import get_candidate_schedules
from vecrec.util import *

def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def diff_kernel(D: int) -> list[list[float]]:
    kernel = [[0.0] * D for i in range(D)]
    kernel[0][0] = kernel[D - 1][D - 1] = 1.0
    kernel[D - 1][0] = kernel[0][D - 1] = -1.0
    return kernel


def naive_box_filter(input_signal: SignalExpr, D: int) -> SignalExpr:
    kernel = diff_kernel(D)

    def box_filter_sat(input_signal: SignalExpr, D: int) -> SignalExpr:
        with with_types(Type.Arith, ElementType.Float):
            sat = Repeater(
                lambda f1: Recurse(TIKernel.z(), SAdd(f1.at(0), input_signal)),
                next_power_of_2(D + 1),
            )
            return PointwiseDiv(Convolve2D(TIKernel2D(kernel), sat), Num(float(D * D)))

    box1 = box_filter_sat(input_signal, D)
    box2 = box_filter_sat(box1, D)
    expr = box_filter_sat(box2, D)
    return expr


# def optimized_box_filter(input_signal: SignalExpr2D, D: int) -> SignalExpr:
#     with with_types(Type.Arith, ElementType.Float):
#         kernel = diff_kernel(D)
#         fir = Convolve2D(TIKernel2D(kernel), input_signal)
#         fir = Convolve2D(TIKernel2D(kernel), fir.cache_rows(D))
#         fir = Convolve2D(TIKernel2D(kernel), fir.cache_rows(D))

#         sat_kernel = TIKernel2D([[0.0, 1.0], [1.0]])
#         iir = Recurse2D(sat_kernel, fir).at(0)
#         iir = Recurse2D(sat_kernel, fir).at(0)
#         iir = Recurse2D(sat_kernel, fir).at(0)

#         return iir


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


# More optimal: Can do all the four-tap filters first, then do prefix sum. The four-tap filters can be decomposed into x and y and fused together.
# Optimization: push down the FIRs before the IIRs.



for D in [3, 5, 7, 9, 11, 13, 15]:
    print(f"D = {D}")

    with with_types(Type.Arith, ElementType.Float):
        expr = optimized_box_filter(Var2D("g", D), D)

    results = get_candidate_schedules(expr, 256)

    benchmark_result = generate_and_run_benchmark(
        CodeGen(cache_cse=True), results, True
    )
