# The continued fraction thing: One big use of continued fractions is getting a very good 
# rational approximation to a real number. This can be done by just skipping the final division 
# in that second-order IIR conversion trick. Nifty. Actually I guess treating the number as a 
# rational and working through the arithmetic is precisely what gives you the transformation.

# the second-order IIR transformation is known as the wallis-euler recurrence. Lentz's algorithm is 
# more popular, which turns it into a top-down evaluation using a different similar recurrence


# Consider
#   f(x)=a(x) + b(x)/f(x-1)
# Instead of computing this, we can compute
#   h(x) = a(x)*h(x-1)+b(x)*h(x-2)
# and consider
#   h(x)/h(x-1) = { a(x)*h(x-1)+b(x)*h(x-2) } / h(x-1) = a(x) + b(x) * h(x-2) / h(x-1)
# We we take f(x) = h(x)/h(x-1), this recovers the original definition of f

from vecrec.expr import *
from vecrec.expr import Type
from vecrec.transform import *
from vecrec.codegen import (
	CodeGen,
	generate_and_run_benchmark,
)
from vecrec.util import ElementType


def main() -> None:
	lanes = 512

	# Time-varying continued fraction:
	#   f(x) = a(x) + b(x) / f(x-1)
	# Convert to second-order recurrence:
	#   h(x) = a(x) * h(x-1) + b(x) * h(x-2)
	# and recover f from f(x) = h(x) / h(x-1).
	a = Var("a", Type.Arith, ElementType.Float)
	b = Var("b", Type.Arith, ElementType.Float)
	h = Recurse(
		TVKernel([Num(0.0, Type.Arith, ElementType.Float), a, b], Type.Arith, ElementType.Float),
		Impulse(1., Type.Arith, ElementType.Float),
	)
	delayed_h = Convolve(TIKernel([0, 1.0], Type.Arith, ElementType.Float), h)
	program = PointwiseDiv(h, delayed_h)

	transforms = Seq(
		Preorder(Try(ConstantFold)),
		AnnotateLanes(lanes),
		PushDownConvertLanes(),
	)

	results = transforms.apply_signal(program)
	print(pps(results))
	# exit(1)
	codegen = CodeGen(True)
	benchmark_result = generate_and_run_benchmark(codegen, results, True)
	print(benchmark_result)


if __name__ == "__main__":
	main()
