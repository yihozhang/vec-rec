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

import numpy as np

from vecrec.expr import *
from vecrec.transform import *
from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
    generate_kernel_executable,
)
from vecrec.util import ElementType


def main() -> None:
    lanes = 512

    # Time-varying continued fraction:
    #   f(x) = a(x) + b(x) / f(x-1)
    # Convert to second-order recurrence:
    #   h(x) = a(x) * h(x-1) + b(x) * h(x-2)
    # and recover f from f(x) = h(x) / h(x-1).
    with with_types(Type.Arith, ElementType.Float):
        a = Var("a")
        b = Var("b")
        h = Recurse(
            TVKernel(
                [
                    Num(0.0),
                    Convolve(TIKernel.z(n=-2), a),
                    (Convolve(TIKernel.z(n=-2), b)),
                ]
            ),
            SAdd(Impulse(1.0), Convolve(TIKernel.z(), Impulse(1.0))),
        )
        delayed_h = Convolve(TIKernel.z(), h)
        program = PointwiseDiv(h, delayed_h)

    transforms = Seq(
        Preorder(Try(ConstantFold)),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = transforms.apply_signal(program)
    codegen = CodeGen(True)
    generate_and_run_benchmark(codegen, results[1:], True)

    for result in results:
        runner = generate_kernel_executable(codegen, result)

        n = 64
        a_in = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)
        b_in = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)
        output, elapsed_us = runner.run({"a": a_in[1:], "b": b_in[1:]})

        f_expected = np.zeros(n, dtype=np.float32)
        f_expected[0] = 1.0
        for i in range(1, n):
            f_expected[i] = a_in[i] + b_in[i] / f_expected[i - 1]

        assert np.allclose(output[2:], f_expected[1:-2], atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    main()
