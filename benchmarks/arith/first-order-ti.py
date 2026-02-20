# Fibonacci: f(x) = f(x-1) + f(x-2)

from vecrec.expr import *
from vecrec.expr import Type
from vecrec.transform import *
from vecrec.codegen import (
	CodeGen,
	generate_and_run_benchmark,
)


def main():
	for program in [
        Recurse(TIKernel([0, 1.0, 1.0], Type.Arith, ElementType.Float), Num(1.0, Type.Arith, ElementType.Float)),
        Recurse(TIKernel([0, 1.0, 1.0], Type.Arith, ElementType.Float), Var("x", Type.Arith, ElementType.Float)),
    ]:
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
                codegen, results, True
            )
            print(benchmark_result)


if __name__ == "__main__":
	main()

