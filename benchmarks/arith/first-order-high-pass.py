"""First-order high-pass filter benchmark.

f(x) = a * (f(x-1) + g(x) - g(x-1))
"""

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
	lanes = 512
	a = 0.8

	program = Recurse(
		TIKernel([0, a], Type.Arith),
		Convolve(TIKernel([a, -a], Type.Arith), Var("g", Type.Arith)),
	)

	transforms = Seq(
		Optional(Seq(Preorder(Try(Factorize())), Preorder(Try(ConstantFold)))),
		Optional(
			Seq(
				RepeatUpTo(3, Dilate(), Preorder(Try(ConstantFold))),
				Any(Dilate(), Delay()),
			),
		),
		AnnotateLanes(lanes),
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
