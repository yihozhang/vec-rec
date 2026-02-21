# Box filter can be implemented using summed area table (integral image).
# An 11x11 box filter is computed using the SAT with the inclusion-exclusion principle:
#   box[r][c] = (SAT[r][c] - SAT[r-11][c] - SAT[r][c-11] + SAT[r-11][c-11]) / 121

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.transform import *
from vecrec.util import *

g = Var("g", Type.Arith, ElementType.Float)

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

for D in [3, 5, 7, 9, 11, 13, 15]:
    print(f"D = {D}")
    sat = Repeater(lambda f1: Recurse(TIKernel([0., 1.], Type.Arith, ElementType.Float), SAdd(Ith(f1, 0), g)), next_power_of_2(D+1), Type.Arith, ElementType.Float)
    kernel = [[0.] * D for i in range(D)]
    kernel[0][0] = kernel[D-1][D-1] = 1.
    kernel[D-1][0] = kernel[0][D-1] = -1.
    expr = PointwiseDiv(Convolve2D(TIKernel2D(kernel, Type.Arith, ElementType.Float), sat), Num(121., Type.Arith, ElementType.Float))


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

    benchmark_result = generate_and_run_benchmark(
        CodeGen(cache_cse=False), results, True
    )
