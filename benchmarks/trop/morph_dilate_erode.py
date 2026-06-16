# 1D causal running max/min and 2D column-wise running max using tropical semirings.
#
# MorphDilate1D:  y[i] = max(y[i-1], x[i])  (TropMax, running maximum)
# MorphErode1D:   y[i] = min(y[i-1], x[i])  (TropMin, running minimum)
# MorphDilate2D:  out[r][j] = max(out[r-1][j], x[r][j])  (TropMax, column-wise running max)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class MorphDilate1DBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "morph-dilate-1d"

    @property
    def description(self) -> str:
        return "1D causal running max (morphological dilation) in TropMax semiring"

    def original_expr(self) -> SignalExpr:
        with with_types(Type.TropMax, ElementType.Float):
            # TIKernel([zero, 0.0]): at lag-1 coeff = 0.0 (the multiplicative one of TropMax)
            # y[i] = max(0.0 + y[i-1], x[i]) = max(y[i-1], x[i])
            return Recurse(TIKernel([Type.TropMax.zero(), 0.0]), Var("x"))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        x = inputs["x"].astype(np.float32)
        return np.maximum.accumulate(x).astype(np.float32)


class MorphErode1DBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "morph-erode-1d"

    @property
    def description(self) -> str:
        return "1D causal running min (morphological erosion) in TropMin semiring"

    def original_expr(self) -> SignalExpr:
        with with_types(Type.TropMin, ElementType.Float):
            # TIKernel([zero, 0.0]): at lag-1 coeff = 0.0 (the multiplicative one of TropMin)
            # y[i] = min(0.0 + y[i-1], x[i]) = min(y[i-1], x[i])
            return Recurse(TIKernel([Type.TropMin.zero(), 0.0]), Var("x"))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        x = inputs["x"].astype(np.float32)
        return np.minimum.accumulate(x).astype(np.float32)


class MorphDilate2DBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "morph-dilate-2d"

    @property
    def description(self) -> str:
        return "2D column-wise running max (separable dilation) in TropMax semiring"

    @property
    def verify_n_rows_2d(self) -> int:
        return 8

    def original_expr(self) -> SignalExpr:
        with with_types(Type.TropMax, ElementType.Float):
            x = Var2D("x")
            # Repeater: prev_rows is the previous row's output.
            # For the current row: out[j] = max(prev_out[j], x[j])
            # SAdd in TropMax is pointwise max.
            rep = Repeater(lambda prev_rows: SAdd(Ith(prev_rows, 0), Ith(x, 0)))
            return Ith(rep, 0)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["x"].astype(np.float32)
        # Cumulative max along the row (axis=0) direction
        result = np.maximum.accumulate(g, axis=0)
        return result.flatten().astype(np.float32)
