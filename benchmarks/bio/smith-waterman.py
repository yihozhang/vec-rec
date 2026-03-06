# Smith-Waterman local alignment
#
# Recurrence:
#   M(i,j) = max( 0,
#                 M(i-1,j-1) + s(x_i, y_j),
#                 M(i-1,j)   + g,
#                 M(i,j-1)   + g )
#
# Identical to Needleman-Wunsch except for the extra "0" term which clips
# negative scores, making this a local (rather than global) alignment.
#
# --------------------------------------------------------------------------
# Scheduling notes for vectorization
# --------------------------------------------------------------------------
#
# The same horizontal dependency M(i,j-1) is the serial bottleneck.
# All NW strategies apply here; the only difference is the clamp to 0.
#
# Approach 1 -- Dilate the intra-row recurrence:
#   After Eliminate2DKernels, the structure is
#     Repeater(Recurse(K_row0, Convolve2D(K_rest, prev) + g))
#   Dilating K_row0 vectorizes the horizontal direction. The zero-clamp
#   does NOT break the semiring structure because max(0, ...) can be folded
#   into the feed signal (i.e., feed'(j) = max(0, feed(j))).
#
# Approach 2 -- Striped / inter-sequence parallelism:
#   Instead of parallelizing within one alignment, align many short query
#   sequences against the same database sequence simultaneously. Each lane
#   handles a different query. This avoids the intra-row dependency entirely
#   and is the approach used by SWIPE / SWIMD. In VecRec terms, this would
#   be multiple independent Recurse nodes sharing the same structure.
#
# Approach 3 -- Wavefront on anti-diagonals (same as NW):
#   Cells on the same anti-diagonal are independent. Map them to SIMD lanes.
#   Works for moderate sequence lengths but has load-balance issues for
#   very long sequences.
#
# Approach 4 -- Prefix-scan with correction:
#   The 0-clamp complicates pure prefix-scan approaches since the semiring
#   homomorphism is broken by the clamp. However, one can speculatively
#   compute without the clamp, then apply corrections. This is an active
#   research area (e.g., the "lazy-F" optimization in Farrar's striped SW).
#
# Below we express the full 2D recurrence as Recurse2D. The kernel structure
# is identical to NW; the 0-clamp is a property of the feed signal, not the
# recurrence kernel.

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.transform import *
from vecrec.util import ElementType

Z = Type.TropMax.zero()  # -inf


def main() -> None:
    # 2D kernel -- identical to NW:
    #   row 0 (same row):  [Z, gap]     -- M(i,j-1) + gap
    #   row 1 (prev row):  [gap, score] -- M(i-1,j) + gap, M(i-1,j-1) + score

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
