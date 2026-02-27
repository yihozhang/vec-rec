# f(x, y) = min(f(x-1, y) + 1, f(x, y-1) + 1, f(x-1, y-1) + sqrt(2), f(x-1, y-2) + sqrt(5), f(x-2, y-1) + sqrt(5), g(x, y))

import math

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.transform import *
from vecrec.util import *

INF = float("inf")

g = Var("g", Type.TropMin, ElementType.Float)

# Kernel for Recurse2D(A, g) in the TropMin semiring (mul = +, add = min):
#   A[0] = same-row feedback:    f(x-1, y) has weight 1.0
#   A[1] = 1 row back:           f(x, y-1)+1, f(x-1, y-1)+sqrt(2), f(x-2, y-1)+sqrt(5)
#   A[2] = 2 rows back:          f(x-1, y-2) has weight sqrt(5)
# INF is the zero element of TropMin, so sparse repr skips those taps.
kernel_data = [
    [INF, 1.0],                           # row 0 (same row)
    [1.0, math.sqrt(2), math.sqrt(5)],    # row 1 (1 row back)
    [INF, math.sqrt(5)],                  # row 2 (2 rows back)
]

expr = Recurse2D(TIKernel2D(kernel_data, Type.TropMin, ElementType.Float), g)

transforms = Seq(
    Preorder(Eliminate2DKernels()),
    Repeat(5, Preorder(Try(ConstantFold))),
    AnnotateLanes(256),
    PushDownConvertLanes(),
)

results = transforms.apply_generic(expr)

benchmark_result = generate_and_run_benchmark(CodeGen(cache_cse=True), results, True)
