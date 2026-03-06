# RNA secondary structure prediction (Nussinov-style folding)
#
# Recurrence:
#   N(i,j) = max( N(i+1,j),
#                 N(i,j-1),
#                 N(i+1,j-1) + s(i,j),
#                 max_{k=i+1..j-1}( N(i,k) + N(k+1,j) ) )
#
# This is a 2D recurrence over intervals [i,j] of the RNA sequence.
# N(i,j) counts the maximum number of base pairs in the subsequence from
# position i to position j.
#
# --------------------------------------------------------------------------
# Scheduling notes for vectorization
# --------------------------------------------------------------------------
#
# This recurrence is unusual: it is defined over *intervals* rather than
# grid positions, so the dependency structure is triangular (j > i only).
# The inner max over k (the "bifurcation" term) makes each cell depend on
# an entire row/column segment, not just a fixed set of neighbors.
#
# Approach 1 -- Anti-diagonal wavefront on the (i,j) triangle:
#   Process cells with the same interval length d = j - i together. For a
#   fixed d, all (i, i+d) cells are independent of each other (they only
#   depend on shorter intervals). This is the standard O(n^3) DP schedule.
#   Within each diagonal, the n cells can be processed in parallel using
#   SIMD. No intra-diagonal recurrence exists, so this is embarrassingly
#   parallel per diagonal.
#
# Approach 2 -- Vectorize the bifurcation (inner) loop:
#   For a single cell N(i,j), the max over k is a reduction:
#     B(i,j) = max_{k}( N(i,k) + N(k+1,j) )
#   This is an inner product in the TropMax semiring between row i and
#   column j of the DP table. It can be computed with a vectorized dot
#   product (using SIMD max and add). This doesn't reduce asymptotic
#   complexity but improves constants.
#
# Approach 3 -- Four Russians speedup:
#   Precompute small blocks of the DP table and use lookup tables to
#   accelerate the bifurcation loop. This reduces complexity to O(n^3/log n).
#   Not directly expressible as a VecRec recurrence, but the block structure
#   could be modeled as a tiled Repeater pattern.
#
# Approach 4 -- Strip the bifurcation term:
#   Without the bifurcation max over k, the recurrence becomes:
#     N(i,j) = max( N(i+1,j), N(i,j-1), N(i+1,j-1) + s(i,j) )
#   This IS a 2D recurrence with fixed 2D kernel, directly expressible as
#   Recurse2D in TropMax. The bifurcation can be applied as a correction
#   pass. Below we model this simplified version.
#
# The simplified recurrence (without bifurcation) is a 2D TropMax IIR.
# Note: the original recurrence grows i from high to low (i+1 dependency),
# but we can reverse the indexing so row 0 is the outermost and the
# recurrence scans top-to-bottom, left-to-right:
#   N'(r,c) = max( N'(r-1,c), N'(r,c-1), N'(r-1,c-1) + s'(r,c) )
#
# After Eliminate2DKernels, the Recurse2D becomes
#   Repeater(Recurse(K_row0, Convolve2D(K_rest, prev) + g))
# where K_row0 = [Z, 0] is the horizontal feedback (N'(r,c-1)).
# Dilating this inner Recurse vectorizes the horizontal direction while
# the Repeater keeps the vertical direction serial.

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.transform import *
from vecrec.util import ElementType

Z = Type.TropMax.zero()  # -inf


def main() -> None:
    # 2D kernel:
    #   row 0 (same row):  [Z, 0]     -- N'(r,c-1) + 0 (horizontal feedback)
    #   row 1 (prev row):  [0, score] -- N'(r-1,c) + 0, N'(r-1,c-1) + score

    score = 1

    with with_types(Type.TropMax, ElementType.I64):
        kernel_data: list[list[float]] = [
            [Z, 0],
            [0, score],
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
