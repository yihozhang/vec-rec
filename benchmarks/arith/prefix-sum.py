# f(x) = f(x-1) + g(x)

from vecrec.expr import *
from vecrec.expr import Type
from vecrec.transform import *
from vecrec.codegen import CodeGen, generate_and_run_benchmark, generate_benchmark, instantiate_kernels


def main():
    program = Recurse(TIKernel([0, 1], Type.Arith), Var("g", Type.Arith))
    transforms = [
        Delay(),
        Preorder(Try(ConstantFold)),
    ]
    results = ApplySequence(transforms).apply_signal(program)
    assert len(results) == 1
    codegen = CodeGen(256)
    code = codegen.generate(results[0], "prefix_sum")
    instantiate_kernels("prefix_sum.h", [code])
    benchmark_result = generate_and_run_benchmark(codegen, results, ["prefix_sum"], True)
    print(benchmark_result)

if __name__ == "__main__":
    main()