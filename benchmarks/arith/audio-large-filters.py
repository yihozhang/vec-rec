# Audio applications can require large filters

# f(x) = f(x-1) * 1.8 + f(x-2) * -0.9 + g(x)

from vecrec.expr import *
from vecrec.transform import *
from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
)
from vecrec.transform import SaturatePreorder
from vecrec.util import ElementType


def main():
    with with_types(Type.Arith, ElementType.Float):
        kernel = TIKernel([0, 1.8, -0.9])
        program = Recurse(kernel, Recurse(kernel, Var("g")))
    transforms = Seq(
        Optional(Preorder(Try(FuseRecurse()))),
        # Preorder(Try(FuseRecurse())),
        SaturatePreorder(ConstantFold),
        Any(
            Noop(),
            Seq(
                Dilate(),
                Dilate(),
                Dilate(),
                Any(Noop(), Dilate()),
            ),
        ),
        SaturatePreorder(ConstantFold),
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
