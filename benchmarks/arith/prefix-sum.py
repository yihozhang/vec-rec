# f(x) = f(x-1) + g(x)

from vecrec.expr import *
from vecrec.expr import Type
from vecrec.transform import *
from vecrec.codegen import CodeGen, generate_and_run_benchmark


def main():
    lanes = 512
    program = Recurse(TIKernel([0, 1], Type.Arith, ElementType.Float), Var("g", Type.Arith, ElementType.Float))
    transforms = Seq(
        Any(
            Noop(),
            Seq(
                Dilate(),
                Dilate(),
                Dilate(),
                Any(Noop(), Dilate()),
            ),
        ),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = transforms.apply_signal(program)
    codegen = CodeGen()
    benchmark_result = generate_and_run_benchmark(codegen, results, True)
    print(benchmark_result)

if __name__ == "__main__":
    main()
