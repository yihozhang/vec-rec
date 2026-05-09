# f(x, y) = min(f(x-1, y) + 1, f(x, y-1) + 1, f(x-1, y-1) + sqrt(2), f(x-1, y-2) + sqrt(5), f(x-2, y-1) + sqrt(5), g(x, y))

from __future__ import annotations

import math

import numpy as np

from vecrec.codegen.codegen import CodeGen
from vecrec.expr import *
from vecrec.transform import *
from vecrec.util import *

from benchmark import Benchmark

INF = float("inf")

# Kernel for Recurse2D(A, g) in the TropMin semiring (mul = +, add = min):
#   A[0] = same-row feedback:    f(x-1, y) has weight 1.0
#   A[1] = 1 row back:           f(x, y-1)+1, f(x-1, y-1)+sqrt(2), f(x-2, y-1)+sqrt(5)
#   A[2] = 2 rows back:          f(x-1, y-2) has weight sqrt(5)
# INF is the zero element of TropMin, so sparse repr skips those taps.
_KERNEL_DATA = [
    [INF, 1.0],  # row 0 (same row)
    [1.0, math.sqrt(2), math.sqrt(5)],  # row 1 (1 row back)
    [INF, math.sqrt(5)],  # row 2 (2 rows back)
]


class DistanceTransformBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "distance-transform"

    @property
    def description(self) -> str:
        return "2D distance transform in TropMin semiring with knight-move kernel"

    # @property
    # def verify_n_rows_2d(self) -> int:
    #     return 4

    # @property
    # def verify_rtol(self) -> float:
    #     # Vectorized float32 min-plus reorders vs scalar; allow 5% relative error
    #     return 0.05

    def original_expr(self) -> SignalExpr:
        with with_types(Type.TropMin, ElementType.Float):
            g = Var2D("g")
            return Recurse2D(TIKernel2D(_KERNEL_DATA), g.at(0)).at(0)

    def build_codegen(self) -> CodeGen:
        return CodeGen(rows_per_iter=2)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        """
        Naive 2D min-plus recurrence:
          f[y,x] = min(g[y,x],
                       f[y, x-1] + 1,
                       f[y-1, x] + 1,
                       f[y-1, x-1] + sqrt(2),
                       f[y-1, x-2] + sqrt(5),
                       f[y-2, x-1] + sqrt(5))
        """
        g = inputs["g"].astype(np.float32)
        rows, cols = g.shape
        f = np.full((rows, cols), np.inf, dtype=np.float32)

        sqrt2 = math.sqrt(2)
        sqrt5 = math.sqrt(5)

        for y in range(rows):
            for x in range(cols):
                candidates = [float(g[y, x])]
                if x >= 1:
                    candidates.append(f[y, x - 1] + 1.0)
                if y >= 1:
                    candidates.append(f[y - 1, x] + 1.0)
                if y >= 1 and x >= 1:
                    candidates.append(f[y - 1, x - 1] + sqrt2)
                if y >= 1 and x >= 2:
                    candidates.append(f[y - 1, x - 2] + sqrt5)
                if y >= 2 and x >= 1:
                    candidates.append(f[y - 2, x - 1] + sqrt5)
                f[y, x] = min(candidates)

        print(f.flatten(), flush=True)
        return f.flatten()
