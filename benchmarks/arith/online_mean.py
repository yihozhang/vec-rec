# Welford's online mean algorithm
#
# mean[n] = mean[n-1] + (x[n] - mean[n-1]) / n
#         = (1 - 1/n) * mean[n-1] + x[n]/n
#
# This is a time-VARYING first-order IIR:
#   mean[n] = coeff[n] * mean[n-1] + x[n] * inv_n[n]
# where coeff[n] = 1 - 1/n and inv_n[n] = 1/n.
#
# The coefficient array and scaled input are pre-computed and passed as variables.
# VecRec handles the IIR part via Recurse(TVKernel([Num(0), coeff_signal]), input_signal).

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class OnlineMeanBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "online-mean"

    @property
    def description(self) -> str:
        return "Welford online mean: mean[n] = (1-1/n)*mean[n-1] + x[n]/n"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            # inv_n[n] = 1/n, coeff[n] = 1 - 1/n
            inv_n = Var("inv_n")
            coeff = SSub(Num(1.0), inv_n)  # (1 - 1/n)
            # scaled input: x[n] * (1/n)
            x_scaled = PointwiseMul(Var("x"), inv_n)
            # mean[n] = coeff[n] * mean[n-1] + x_scaled[n]
            return Recurse(TVKernel([Num(0.0), coeff]), x_scaled)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        x = inputs["x"].astype(np.float32)
        inv_n = inputs["inv_n"].astype(np.float32)
        mean = np.zeros(len(x), dtype=np.float32)
        for i in range(len(x)):
            mean_prev = mean[i - 1] if i >= 1 else np.float32(0.0)
            coeff = np.float32(1.0) - inv_n[i]
            mean[i] = coeff * mean_prev + x[i] * inv_n[i]
        return mean
