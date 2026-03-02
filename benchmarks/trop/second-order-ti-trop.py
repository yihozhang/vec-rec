# f(x) = f(x-1) * 1.8 + f(x-2) * -0.9 + g(x)

from vecrec.expr import *
from vecrec.transform import *
from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
)
from vecrec.transform import DilateTrop
from vecrec.util import ElementType


def main() -> None:
    with with_types(Type.TropMax, ElementType.I64):
        program = Recurse(TIKernel([Type.TropMax.zero(), 2, 5]), Var("g"))
    transforms = Seq(
        Optional(
            Seq(
                RepeatUpTo(4, DilateTrop(), Preorder(Try(ConstantFold))),
            ),
        ),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )

    results = transforms.apply_signal(program)
    print(pps(results))
    codegen = CodeGen()
    benchmark_result = generate_and_run_benchmark(
        codegen, results, True
    )
    print(benchmark_result)


if __name__ == "__main__":
    main()
