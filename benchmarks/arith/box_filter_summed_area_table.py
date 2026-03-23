# Box filter can be implemented using summed area table (integral image).
# An 11x11 box filter is computed using the SAT with the inclusion-exclusion principle:
#   box[r][c] = (SAT[r][c] - SAT[r-11][c] - SAT[r][c-11] + SAT[r-11][c-11]) / 121

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D
from vecrec.util import ElementType

from benchmark import Benchmark


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def box_filter_optimized(input_signal: SignalExpr2D, D: int) -> SignalExpr:
    """Compute box filter using SAT; Factorize2D will split the rank-1 kernel."""
    with with_types(Type.Arith, ElementType.Float):
        kernel = [[0.0] * D for _ in range(D)]
        kernel[0][0] = kernel[D - 1][D - 1] = 1.0
        kernel[D - 1][0] = kernel[0][D - 1] = -1.0
        fir = Convolve2D(TIKernel2D(kernel), input_signal)

        sat_kernel = TIKernel2D([[0.0, 1.0], [1.0, -1.0]])
        iir = Recurse2D(sat_kernel, fir).at(0)

        return iir


def _sat_reference(g: np.ndarray, D: int) -> np.ndarray:
    """
    Reference implementation of box_filter_optimized.

    g: (rows, cols) float32 array.
    Returns: (rows * cols,) float32 — the 2D prefix sum of the difference image,
             laid out in row-major order.
    """
    rows, cols = g.shape
    g = g.astype(np.float32)

    # Step 1: difference image (the 4-tap 2D kernel)
    fir = np.zeros_like(g)
    for y in range(rows):
        for x in range(cols):
            v = g[y, x]
            if y >= D - 1 and x >= D - 1:
                v += g[y - (D - 1), x - (D - 1)]
            if y >= D - 1:
                v -= g[y - (D - 1), x]
            if x >= D - 1:
                v -= g[y, x - (D - 1)]
            fir[y, x] = v

    # Step 2: 2D prefix sum (SAT) — s[y,x] = s[y-1,x] + s[y,x-1] - s[y-1,x-1] + fir[y,x]
    s = np.zeros_like(fir)
    for y in range(rows):
        for x in range(cols):
            s_up = s[y - 1, x] if y >= 1 else 0.0
            s_left = s[y, x - 1] if x >= 1 else 0.0
            s_upleft = s[y - 1, x - 1] if (y >= 1 and x >= 1) else 0.0
            s[y, x] = s_up + s_left - s_upleft + fir[y, x]

    return s.flatten().astype(np.float32)


class BoxFilterSATBenchmark(Benchmark):
    """Box filter via summed area table, parameterised by filter radius D."""

    def __init__(self, D: int) -> None:
        self.D = D

    @property
    def name(self) -> str:
        return f"box-filter-sat-D{self.D}"

    @property
    def description(self) -> str:
        return f"Box filter (SAT, D={self.D}) via 2D difference kernel + SAT recursion"

    def original_expr(self) -> SignalExpr:
        with with_types(Type.Arith, ElementType.Float):
            return box_filter_optimized(Var2D("g"), self.D)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"]  # shape (rows, cols)
        return _sat_reference(g, self.D)
