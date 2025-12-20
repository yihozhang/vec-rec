# f(x) = f(x-1) + g(x)

from vecrec.expr import *
from vecrec.expr import Type
from vecrec.transform import *
from vecrec.codegen import CodeGen, generate_and_run_benchmark, generate_benchmark, instantiate_kernels


def main():
    program = Recurse(TIKernel([0, 1], Type.Arith), Var("g", Type.Arith))
    transforms = Any(
        Seq(
            Dilate(),
            Dilate(),
            Dilate(),
        ),
    )
    results = transforms.apply_signal(program)
    codegen = CodeGen(256)
    benchmark_result = generate_and_run_benchmark(codegen, [program, *results], ["original", "prefix_sum"], True)
    print(benchmark_result)

if __name__ == "__main__":
    main()