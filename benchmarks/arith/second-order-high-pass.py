# f(x) = f(x-1) * 1.6 + f(x-2) * -0.64 + g(x) - 2 * g(x-1) + g(x-2)

from vecrec.expr import *
from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
)
from vecrec.optimize import get_candidate_schedules
from vecrec.util import ElementType


def main() -> None:
    lanes = 512
    with with_types(Type.Arith, ElementType.Float):
        program = Recurse(
            TIKernel([0, 1.6, -0.64]),
            Convolve(TIKernel([1, -2, 1]), Var("g")),
        )
    results = get_candidate_schedules(program, lanes)
    codegen = CodeGen()
    benchmark_result = generate_and_run_benchmark(
        codegen, results, True
    )
    print(benchmark_result)


if __name__ == "__main__":
    main()
