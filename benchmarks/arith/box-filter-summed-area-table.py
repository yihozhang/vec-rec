# Box filter can be implemented using summed area table (integral image).

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.transform import AnnotateLanes, ConstantFold, Dilate, Eliminate2DKernels, Preorder, PushDownConvertLanes, Seq, Try
from vecrec.util import *


kernel = TIKernel2D([[0., 1.], [1., 0.]], Type.Arith, ElementType.Float)
g = Var("g", Type.Arith, ElementType.Float)

expr = Ith(Recurse2D(kernel, g), 0)

# transforms = Seq(
#     Preorder(Eliminate2DKernels()),
#     Preorder(Try(Dilate())),
# )

transforms = Seq(
    Preorder(Eliminate2DKernels()),
    Preorder(Try(Dilate())),
    Preorder(Try(Dilate())),
    Preorder(Try(Dilate())),
    Preorder(Try(ConstantFold)),
    AnnotateLanes(256),
    PushDownConvertLanes()
)

results = transforms.apply_generic(expr)
print(pp(expr))
print(pps(results))

# benchmark_result = generate_and_run_benchmark(
#     CodeGen(), results, ["k" + str(i) for i in range(len(results))], True
# )
