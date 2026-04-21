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

from __future__ import annotations

import numpy as np

from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
    generate_kernel_executable,
)
from vecrec.expr import *
from vecrec.expr.base import SignalExpr, SignalExpr2D
from vecrec.optimize import get_candidate_schedules
from vecrec.util import ElementType

from benchmark import Benchmark


class ContinuedFractionBenchmark(Benchmark):
    DEBUG = True

    @property
    def name(self) -> str:
        return "continued-fraction"

    @property
    def description(self) -> str:
        return "Continued fraction via Wallis-Euler second-order recurrence: f(x) = a(x) + b(x)/f(x-1)"

    def original_expr(self) -> PointwiseDiv:
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
            return PointwiseDiv(h, delayed_h)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        # Not used directly — verify_schedule is overridden below.
        raise NotImplementedError("ContinuedFractionBenchmark uses a custom verify_schedule")

    def verify_schedule(self, expr: SignalExpr | SignalExpr2D) -> tuple[bool, str]:
        """
        Verify using the Python continued-fraction reference:
          f[i] = a[i] + b[i] / f[i-1]
        as in the original continued-fraction.py.
        """
        try:
            codegen = CodeGen()
            runner = generate_kernel_executable(codegen, expr)

            rng = np.random.default_rng(42)
            n = 64
            a_in = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
            b_in = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)

            # Runner receives a[1:] and b[1:] (first element is used as initial condition)
            output, _ = runner.run({"a": a_in[1:], "b": b_in[1:]})

            f_expected = np.zeros(n, dtype=np.float32)
            f_expected[0] = 1.0
            for i in range(1, n):
                f_expected[i] = a_in[i] + b_in[i] / f_expected[i - 1]

            # Compare with the same slice as the original script
            out_slice = output[2:]
            ref_slice = f_expected[1:-2]
            min_len = min(len(out_slice), len(ref_slice))
            out_slice = out_slice[:min_len]
            ref_slice = ref_slice[:min_len]

            if not np.allclose(out_slice, ref_slice, atol=1e-3, rtol=1e-3):
                bad = np.where(~np.isclose(out_slice, ref_slice, atol=1e-3, rtol=1e-3))[0]
                first_diff = int(bad[0])
                return False, (
                    f"Mismatch at index {first_diff}: "
                    f"got {float(out_slice[first_diff]):.6f}, "
                    f"expected {float(ref_slice[first_diff]):.6f}"
                )
            return True, ""
        except Exception as exc:
            return False, str(exc)


def main() -> None:
    lanes = 512
    program = ContinuedFractionBenchmark().original_expr()
    results = get_candidate_schedules(program, lanes)
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
