# Audio applications can require large filters

# f(x) = f(x-1) * 1.8 + f(x-2) * -0.9 + g(x)

from vecrec.expr import *
from vecrec.expr import Type
from vecrec.transform import *
from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
)


def main():
    kernel = TIKernel([0, 1.8, -0.9], Type.Arith, ElementType.Float)
    program = Recurse(kernel, Recurse(kernel, Var("g", Type.Arith, ElementType.Float)))
    # transforms = Seq(
    #     Any(
    #         Noop(),
    #         Seq(
    #             Dilate(),
    #             Dilate(),
    #             Dilate(),
    #             Any(Noop(), Dilate()),
    #         ),
    #     ),
    #     AnnotateLanes(512),
    #     PushDownConvertLanes(),
    # )
    transforms = Seq(
        Optional(Preorder(Try(FuseRecurse()))),
        # Preorder(Try(FuseRecurse())),
        Preorder(Try(ConstantFold)),
        Preorder(Try(ConstantFold)),
        Preorder(Try(ConstantFold)),
        Any(
            Noop(),
            Seq(
                Dilate(),
                Dilate(),
                Dilate(),
                Any(Noop(), Dilate()),
            ),
        ),
        Preorder(Try(ConstantFold)),
        Preorder(Try(ConstantFold)),
        Preorder(Try(ConstantFold)),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    
    results = transforms.apply_signal(program)
    # print(results)
    codegen = CodeGen()
    benchmark_result = generate_and_run_benchmark(
        codegen, results, True
    )
    print(benchmark_result)


if __name__ == "__main__":
    main()
