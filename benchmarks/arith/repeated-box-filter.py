# Repeated box filter: apply a box filter three times consecutively to approximate a Gaussian filter.
# Each pass uses a summed area table (SAT) via the inclusion-exclusion principle.

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.transform import *
from vecrec.util import *

def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2**(x - 1).bit_length()

def diff_kernel(D: int) -> list[list[float]]:
    kernel = [[0.] * D for i in range(D)]
    kernel[0][0] = kernel[D-1][D-1] = 1.
    kernel[D-1][0] = kernel[0][D-1] = -1.
    return kernel

def naive_box_filter(input_signal: SignalExpr, D: int) -> SignalExpr:
    kernel = diff_kernel(D)
    def box_filter_sat(input_signal: SignalExpr, D: int) -> SignalExpr:
        sat = Repeater(lambda f1: Recurse(TIKernel.z(Type.Arith, ElementType.Float), SAdd(Ith(f1, 0), input_signal)), next_power_of_2(D+1), Type.Arith, ElementType.Float)
        return PointwiseDiv(Convolve2D(TIKernel2D(kernel, Type.Arith, ElementType.Float), sat), Num(float(D * D), Type.Arith, ElementType.Float))

    box1 = box_filter_sat(input_signal, D)
    box2 = box_filter_sat(box1, D)
    expr = box_filter_sat(box2, D)
    return expr

def optimized_box_filter(input_signal: SignalExpr, D: int) -> SignalExpr:
    kernel = diff_kernel(D)
    def box_filter_sat(input_signal: SignalExpr, D: int) -> SignalExpr:
        fir = Convolve2D(TIKernel2D(kernel, Type.Arith, ElementType.Float), input_signal.cache_rows(D))
        sat_kernel = TIKernel2D([[0., 1.], [1.]], Type.Arith, ElementType.Float)
        return Ith(Recurse2D(sat_kernel, fir), 0)

    box1 = box_filter_sat(input_signal, D)
    box2 = box_filter_sat(box1, D)
    expr = box_filter_sat(box2, D)
    return expr
# More optimal: Can do all the four-tap filters first, then do prefix sum. The four-tap filters can be decomposed into x and y and fused together.
# Optimization: push down the FIRs before the IIRs.


g = Var("g", Type.Arith, ElementType.Float)

for D in [3, 5, 7, 9, 11, 13, 15]:
    print(f"D = {D}")
    expr = naive_box_filter(g, D)

    for D in [3, 5, 7, 9, 11, 13, 15]:
        print(f"D = {D}")
        expr = naive_box_filter(g, D)

        # print(pp(expr))

        transforms = Seq(
            Preorder(Eliminate2DKernels()),
            Preorder(Try(Dilate())),
            Preorder(Try(Dilate())),
            Preorder(Try(Dilate())),
            Any(Preorder(Try(Dilate())), Preorder(Try(Delay()))),
            Repeat(5, Preorder(Try(ConstantFold))),
            AnnotateLanes(256),
            PushDownConvertLanes()
        )

        results = transforms.apply_generic(expr)

        benchmark_result = generate_and_run_benchmark(
            CodeGen(cache_cse=True), results, True
        )

        # benchmark_result = generate_and_run_benchmark(
        #     CodeGen(cache_cse=False), results, True
        # )
