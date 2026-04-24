# M(i,j) = max( 0, M(i-1,j-1) + s(x_i,y_j), M(i-1,j) + g, M(i,j-1) + g )

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.expr.base import KernelExpr2D
from vecrec.util import ElementType

from benchmark import Benchmark

GAP = -1
_BOUND = np.int64(np.iinfo(np.int64).min // 8)

# Same kernel as Needleman-Wunsch, but the external signal is Num(0) instead of
# Impulse(0).  Num(0) contributes 0 (the TropMax "one") at every column j, so
# each cell can "restart" from 0 — the defining property of local alignment.
# Kernel layout (TropMax: mul=+, add=max):
#   row 0 [neginf, gap]:      M(i, j-1) + g  (same-row left feedback)
#   row 1 [gap, s(i,j)]:     M(i-1, j) + g  and  M(i-1, j-1) + s(i,j)
# The Repeater buffer is initialised with the TropMax zero element (_BOUND),
# so "row -1" is effectively -infinity.


class SmithWatermanBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "smith-waterman"

    @property
    def description(self) -> str:
        return "Smith-Waterman local alignment in TropMax (max-plus) semiring"

    @property
    def verify_n_rows_2d(self) -> int:
        return 8

    def original_expr(self) -> SignalExpr:
        with with_types(Type.TropMax, ElementType.I64):
            s = Var2D("s")
            neginf = Num(Type.TropMax.zero())
            gap = Num(GAP)
            kernel: KernelExpr2D = TVKernel2D([
                [neginf, gap],           # row 0: M(i, j-1) + g
                [gap, s.at(0)],     # row 1: M(i-1, j) + g  and  M(i-1, j-1) + s(i,j)
            ])
            return Recurse2D(kernel, Num(0)).at(0)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        s = inputs["s"].astype(np.int64)
        rows, cols = s.shape
        M = np.zeros((rows, cols), dtype=np.int64)

        for i in range(rows):
            # Repeater buffer is initialised with TropMax zero (_BOUND), so row -1 is all _BOUND.
            prev_row = M[i - 1] if i > 0 else np.full(cols, _BOUND, dtype=np.int64)

            for j in range(cols):
                up_val = GAP + int(prev_row[j])
                diag_input = int(prev_row[j - 1]) if j > 0 else int(_BOUND)
                diag_val = int(s[i, j]) + diag_input
                left_val = GAP + (int(M[i, j - 1]) if j > 0 else int(_BOUND))
                # SW: clamp to 0 — local alignment can always restart
                M[i, j] = max(0, left_val, diag_val, up_val)

        return M.flatten()
