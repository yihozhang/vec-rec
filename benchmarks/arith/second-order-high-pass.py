# f(x) = f(x-1) * 1.6 + f(x-2) * -0.64 + g(x) - 2 * g(x-1) + g(x-2)

from vecrec.expr import *
from vecrec.expr import Type
from vecrec.transform import *
from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
)


def main():
    lanes = 512
    program = Recurse(
        TIKernel([0, 1.6, -0.64], Type.Arith, ElementType.Float),
        Convolve(TIKernel([1, -2, 1], Type.Arith, ElementType.Float), Var("g", Type.Arith, ElementType.Float)),
    )
    transforms = Seq(
        Optional(Seq(Preorder(Try(Factorize())), Preorder(Try(ConstantFold)))),
        Optional(
            Seq(
                RepeatUpTo(3, Dilate(), Preorder(Try(ConstantFold))),
                Any(Dilate(), Delay()),
            ),
        ),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = transforms.apply_signal(program)
    codegen = CodeGen()
    benchmark_result = generate_and_run_benchmark(
        codegen, results, True
    )
    print(benchmark_result)


if __name__ == "__main__":
    main()
