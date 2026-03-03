"""First-order high-pass filter benchmark.

f(x) = a * (f(x-1) + g(x) - g(x-1))
"""

from vecrec.expr import *
from vecrec.transform import *
from vecrec.codegen import (
	CodeGen,
	generate_and_run_benchmark,
)
from vecrec.util import ElementType


def main():
	lanes = 512
	a = 0.8

	with with_types(Type.Arith, ElementType.Float):
		program = Recurse(
			TIKernel([0, a]),
			Convolve(TIKernel([a, -a]), Var("g")),
		)

	transforms = Seq(
		Optional(Seq(Preorder(Try(Factorize())), SaturatePreorder(ConstantFold))),
		Optional(
			Seq(
				RepeatUpTo(3, Dilate(), SaturatePreorder(ConstantFold)),
				Any(Dilate(), Delay()),
			),
		),
		AnnotateLanes(lanes),
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
