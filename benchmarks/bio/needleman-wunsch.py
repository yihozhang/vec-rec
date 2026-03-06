# Needleman-Wunsch global alignment
#
# Recurrence:
#   M(i,j) = max( M(i-1,j-1) + s(x_i, y_j),
#                 M(i-1,j)   + g,
#                 M(i,j-1)   + g )
#
# where s(x_i, y_j) is a substitution score (match/mismatch) and g is a gap
# penalty (typically negative).
#
# The recurrence is 2D: each cell depends on the cell above (i-1,j),
# to the left (i,j-1), and diagonally above-left (i-1,j-1).
#
# --------------------------------------------------------------------------
# Scheduling notes for vectorization
# --------------------------------------------------------------------------
#
# The (i,j-1) dependency is the bottleneck: it is a left-to-right recurrence
# along each row (the "horizontal feedback"), preventing naive SIMD across
# columns within a single row.
#
#
# Approach 1 -- Dilate the horizontal recurrence:
#   After Eliminate2DKernels, the 2D recurrence becomes
#     Repeater(Recurse(K_row0, Convolve2D(K_rest, prev) + g))
#   Dilating K_row0 (the horizontal feedback kernel) doubles the dependency
#   gap each time, letting us process 2^k columns in parallel per step
#   while keeping the vertical (row-by-row) iteration serial.
#
# TODO(bala)
# Approach 2 -- Row-parallel with speculative prefix:
#   Speculatively assume gap-only extensions and correct with a TropMax
#   prefix scan. The gap-only prefix is a simple linear recurrence
#   M_gap(j) = M_gap(j-1) + g, which can be dilated aggressively.
#

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.transform import *
from vecrec.util import ElementType

Z = Type.TropMax.zero()  # -inf


def main() -> None:
    # 2D kernel for M(i,j) = max(M(i,j-1)+gap, M(i-1,j)+gap, M(i-1,j-1)+score):
    #   row 0 (same row):  [Z, gap]     -- M(i,j-1) + gap  (horizontal feedback)
    #   row 1 (prev row):  [gap, score]  -- M(i-1,j) + gap, M(i-1,j-1) + score

    gap = -1
    score = 2

    with with_types(Type.TropMax, ElementType.I64):
        kernel_data: list[list[float]] = [
            [Z, gap],
            [gap, score],
        ]
        g = Var("g")
        expr = Recurse2D(TIKernel2D(kernel_data), g)

    transforms = Seq(
        Preorder(Try(Eliminate2DKernels())),
        Repeat(5, Preorder(Try(ConstantFold))),
        Optional(
            Seq(
                Preorder(Try(DilateTrop())),
                Preorder(Try(ConstantFold)),
                Optional(
                    Seq(
                        Preorder(Try(DilateTrop())),
                        Preorder(Try(ConstantFold)),
                    ),
                ),
            ),
        ),
        AnnotateLanes(256),
        PushDownConvertLanes(),
    )

    results = transforms.apply_generic(expr)
    codegen = CodeGen(cache_cse=True)
    benchmark_result = generate_and_run_benchmark(codegen, results, True)
    assert benchmark_result["return_code"] == 0, benchmark_result
    print(benchmark_result)


if __name__ == "__main__":
    main()
