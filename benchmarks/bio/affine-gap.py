# Affine gap alignment (mutually recursive formulation)
#
# Recurrences:
#   M(i,j) = max( M(i-1,j-1), I(i-1,j-1), D(i-1,j-1) ) + s(x_i, y_j)
#   I(i,j) = max( M(i-1,j) + g_o + g_e,  I(i-1,j) + g_e )
#   D(i,j) = max( M(i,j-1) + g_o + g_e,  D(i,j-1) + g_e )
#
# Three coupled matrices. M tracks matches/mismatches, I tracks insertions
# (gaps in the database sequence), D tracks deletions (gaps in the query).
# g_o is the gap-open penalty, g_e is the gap-extend penalty.
#
# --------------------------------------------------------------------------
# Scheduling notes for vectorization
# --------------------------------------------------------------------------
#
# The three matrices have different dependency shapes:
#   - M(i,j) depends only on the diagonal (i-1,j-1) -- no serial dependency
#     within a row or column, given the previous row.
#   - I(i,j) depends on (i-1,j) -- column-wise (vertical) recurrence.
#     Within a single row, all I(i,j) for different j are independent.
#   - D(i,j) depends on (i,j-1) -- row-wise (horizontal) recurrence.
#     This is the serial bottleneck within each row.
#
# Approach 1 -- Vectorize across columns, serialize over rows:
#   For a fixed row i, M and I are fully parallel across j (they only look
#   at row i-1). Only D has intra-row dependence. We can:
#   (a) Compute M(i,:) and I(i,:) as vectorized operations over all j.
#   (b) Compute D(i,:) as a 1st-order TropMax IIR along j:
#         D(j) = max( M(i,j-1) + g_o + g_e,  D(j-1) + g_e )
#       This is a standard 1st-order recurrence that can be dilated.
#   After Eliminate2DKernels, the Recurse2D becomes
#     Repeater(Recurse(K_row0, Convolve2D(K_rest, prev) + g))
#   and dilating the inner Recurse vectorizes only the horizontal direction.
#   This is essentially the Farrar striped approach.
#
# Approach 2 -- Anti-diagonal wavefront:
#   Process all three matrices along anti-diagonals (i+j = const).
#   All cells on the same anti-diagonal are independent for M and I.
#   For D, cells on the same anti-diagonal have different j values, but
#   D(i,j) depends on D(i,j-1) which is on a DIFFERENT anti-diagonal,
#   so anti-diag parallelism works for D too. This gives full SIMD width.
#
# Approach 3 -- Decouple and correct:
#   Compute M and I first (both trivially parallel per row given prev row).
#   Then solve D as an independent 1st-order row recurrence with feed from
#   the already-computed M. Finally, if D changes any M values for the next
#   row, iterate. In practice, convergence is fast because gap extensions
#   decay quickly.
#
# Below we model the D recurrence as a 2D Recurse2D. The D matrix depends
# on D(i,j-1) horizontally and on M(i-1,j) from the previous row (via the
# gap-open + gap-extend penalty). We treat M's contribution as the feed
# signal and express D's recurrence structure in the 2D kernel.

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.transform import *
from vecrec.util import ElementType

Z = Type.TropMax.zero()  # -inf


def main() -> None:
    gap_open = -2
    gap_extend = -1

    # D(i,j) = max( D(i,j-1) + g_e,  feed(i,j) )
    # where feed(i,j) = M(i,j-1) + g_o + g_e (from previous row's M values).
    #
    # As a 2D kernel:
    #   row 0 (same row):  [Z, gap_extend]       -- D(i,j-1) + g_e
    #   row 1 (prev row):  [Z, gap_open+gap_extend] -- M(i-1,j-1) + g_o + g_e
    # The (1,0) entry is Z because D doesn't depend on D(i-1,j) directly
    # (that would be the I recurrence). The (1,1) entry captures the
    # contribution from the previous row shifted by one column (the j-1
    # in "M(i,j-1) + g_o + g_e" -- since M is from row i, which in the
    # Repeater's frame is the previous row's output).

    with with_types(Type.TropMax, ElementType.I64):
        kernel_data: list[list[float]] = [
            [Z, gap_extend],
            [Z, gap_open + gap_extend],
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
