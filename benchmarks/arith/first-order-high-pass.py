"""First-order high-pass filter benchmark.

f(x) = a * (f(x-1) + g(x) - g(x-1))
"""

from vecrec.expr import *
from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
)
from vecrec.optimize import get_candidate_schedules
from vecrec.util import ElementType


def main():
    lanes = 512
    a = 0.8

    with with_types(Type.Arith, ElementType.Float):
        program = Recurse(
            TIKernel([0, a]),
            Convolve(TIKernel([a, -a]), Var("g")),
        )

    results = get_candidate_schedules(program, lanes)
    codegen = CodeGen()
    benchmark_result = generate_and_run_benchmark(codegen, results, True)
    print(benchmark_result)


if __name__ == "__main__":
    main()
