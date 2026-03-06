# Sequence chaining
#
# Recurrence:
#   C(i) = max( w_i + max_{j < i}( C(j) + w_i - s(i,j) ) )
#
# Simplifying:
#   C(i) = w_i + max_{j < i}( C(j) - s(i,j) + w_i )
#        = w_i + max_{j < i}( C(j) + w_i - s(i,j) )
#
# This is a 1D recurrence where each element depends on ALL previous elements,
# weighted by a time-varying penalty s(i,j). This makes it fundamentally
# different from the fixed-tap IIRs in audio processing.
#
# --------------------------------------------------------------------------
# Scheduling notes for vectorization
# --------------------------------------------------------------------------
#
# The key challenge is the "max over all j < i" -- this is a full prefix
# operation, not a fixed-order recurrence.
#
# Approach 1 -- Truncate to fixed order:
#   In practice, the scoring function s(i,j) often grows with |i-j|, making
#   contributions from distant anchors negligible. We can truncate to a
#   fixed window W: C(i) = w_i + max_{i-W <= j < i}( C(j) + w_i - s(i,j) ).
#   This turns it into a W-tap recurrence in TropMax, which can be dilated.
#   Common window sizes: W = 4, 8, 16 for typical genomic fragment chaining.
#
# Approach 2 -- Range-tree / segment-tree augmented DP:
#   Not directly expressible as a VecRec recurrence, but the truncated version
#   is. The segment-tree approach handles the full-prefix case in O(n log n)
#   but doesn't map to the IIR framework.
#
# Approach 3 -- Decompose into prefix-max + pointwise:
#   If s(i,j) decomposes as s(i,j) = f(i) + g(j) (separable penalty), then:
#     C(i) = w_i + max_{j < i}( C(j) - g(j) ) + w_i - f(i)
#   The inner max is now a plain prefix-max scan (1st-order TropMax recurrence
#   with identity kernel), which dilates perfectly. The pointwise terms are
#   trivially parallel.
#
# Below we model the truncated version with a fixed window.

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.transform import *
from vecrec.util import ElementType

INF = float("inf")
Z = Type.TropMax.zero()  # -inf


def make_chaining_program(window: int) -> SignalExpr:
    """Model a truncated chaining recurrence with a fixed lookback window.

    C(i) = max_{k=1..W}( C(i-k) + penalty(k) ) + w_i

    penalty(k) models w_i - s(i, i-k); here we use a simple linear decay.
    In TropMax semiring, mul = +, add = max.
    """
    with with_types(Type.TropMax, ElementType.I64):
        penalties = [Z] + [-k for k in range(1, window + 1)]
        kernel = TIKernel(penalties)
        feed = Var("feed")
        return Recurse(kernel, feed)


def main() -> None:
    for window in [4, 8, 16]:
        print(f"--- window = {window} ---")
        program = make_chaining_program(window)

        transforms = Seq(
            RepeatUpTo(4, DilateTrop(), Preorder(Try(ConstantFold))),
            AnnotateLanes(512),
            PushDownConvertLanes(),
        )

        results = transforms.apply_signal(program)
        print(pps(results))
        codegen = CodeGen()
        benchmark_result = generate_and_run_benchmark(codegen, results, True)
        assert benchmark_result["return_code"] == 0, benchmark_result
        print(benchmark_result)


if __name__ == "__main__":
    main()
