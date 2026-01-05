# f(x) = f(x-1) * 1.8 + f(x-2) * -0.9 + g(x)

from vecrec.expr import *
from vecrec.expr import Type
from vecrec.transform import *
from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
    generate_benchmark,
    instantiate_kernels,
)


def main():
    program = Recurse(TIKernel([0, 1.8, -0.9], Type.Arith), Var("g", Type.Arith))
    transforms = Seq(
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
        codegen, results, ["k" + str(i) for i in range(len(results))], True
    )
    print(benchmark_result)


if __name__ == "__main__":
    main()
