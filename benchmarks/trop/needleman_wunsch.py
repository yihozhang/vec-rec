# M(i,j) = max( M(i-1,j-1) + s(x_i,y_j), M(i-1,j) + g, M(i,j-1) + g )

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.expr.base import KernelExpr2D
from vecrec.util import ElementType

from benchmark import Benchmark

GAP = -1
# SConvolveOp / SRecurseOp for TropMax use INT64_MIN/8 as the out-of-bounds sentinel.
_BOUND = np.int64(np.iinfo(np.int64).min // 8)

# Kernel for Recurse2D(A, zero) in TropMax semiring (mul = +, add = max):
#   A[0] = same-row feedback:  M(i, j-1) + g
#   A[1] = 1 row back:         M(i-1, j) + g  and  M(i-1, j-1) + s(i,j)
# NEGINF (TropMax zero) at A[0][0] means no self-reference.
# The diagonal term A[1][1] = s(i,j) is time-varying (TVKernel2D).
# The Repeater buffer is memset(0), so "row -1" is all zeros.


class NeedlemanWunschBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "needleman-wunsch"

    @property
    def description(self) -> str:
        return "Needleman-Wunsch global alignment in TropMax (max-plus) semiring"

    @property
    def verify_n_rows_2d(self) -> int:
        return 8

    def original_expr(self) -> SignalExpr:
        with with_types(Type.TropMax, ElementType.I64):
            s = Var2D("s")
            neginf = Num(Type.TropMax.zero())
            gap = Num(GAP)
            kernel: KernelExpr2D = TVKernel2D([
                [neginf, gap],     # row 0: M(i, j-1) + g
                [gap, s.at(0)],    # row 1: M(i-1, j) + g  and  M(i-1, j-1) + s(i,j)
            ])
            return Recurse2D(kernel, Impulse(0)).at(0)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        s = inputs["s"].astype(np.int64)
        rows, cols = s.shape
        M = np.zeros((rows, cols), dtype=np.int64)

        for i in range(rows):
            # Repeater buffer: initialized to 0, so "row -1" is all zeros.
            prev_row = M[i - 1] if i > 0 else np.full(cols, _BOUND, dtype=np.int64)

            for j in range(cols):
                if i == 0 and j == 0:
                    M[i, j] = 0  # Recurse2D zero (TropMax neginf) is overridden by TVKernel2D at (0,0).
                    continue
                # Convolve(TVKernel([gap, s[i]]), prev_row):
                #   kernel[0]=gap  multiplies prev_row[j]   (M(i-1, j))
                #   kernel[1]=s[i,j] multiplies prev_row[j-1] (M(i-1, j-1))
                # Boundary prev_row[-1] uses _BOUND (SConvolveOp TropMax init).
                up_val = GAP + int(prev_row[j])
                diag_input = int(prev_row[j - 1]) if j > 0 else int(_BOUND)
                diag_val = int(s[i, j]) + diag_input

                # Recurse(TVKernel([neginf, gap]), external):
                # sparse kernel is [gap] at index 1: M[i][j] = max(gap + M[i][j-1], external_j)
                # Boundary M[i][-1] uses _BOUND (SRecurseOp TropMax init).
                left_val = GAP + (int(M[i, j - 1]) if j > 0 else int(_BOUND))
                M[i, j] = max(left_val, diag_val, up_val)

        return M.flatten()
