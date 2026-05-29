# GARCH(1,1) financial volatility model
#
# h[t] = omega + alpha * r[t-1]^2 + beta * h[t-1]
#
# where r[t] is the return series (input) and h[t] is the conditional variance.
#
# Rewritten as a time-varying IIR:
#   h[t] = beta * h[t-1] + g[t]
# where g[t] = alpha * r[t-1]^2 + omega
#
# This is Recurse(TIKernel([0, beta]), g_signal)
# where g_signal = PointwiseMul(Num(alpha), PointwiseMul(prev_r, prev_r)) + Num(omega)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark

OMEGA = 0.1
ALPHA = 0.1
BETA = 0.85


class GARCHBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "garch"

    @property
    def description(self) -> str:
        return "GARCH(1,1) conditional variance: h[t] = omega + alpha*r[t-1]^2 + beta*h[t-1]"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            r = Var("r")
            # r[t-1]: one-step delayed return
            prev_r = Convolve(TIKernel.z(), r)
            # g[t] = alpha * r[t-1]^2 + omega
            r_sq = PointwiseMul(prev_r, prev_r)
            g = SAdd(PointwiseMul(Num(ALPHA), r_sq), Num(OMEGA))
            # h[t] = beta * h[t-1] + g[t]
            return Recurse(TIKernel([0.0, BETA]), g)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        r = inputs["r"].astype(np.float32)
        omega = np.float32(OMEGA)
        alpha = np.float32(ALPHA)
        beta = np.float32(BETA)
        h = np.zeros(len(r), dtype=np.float32)
        for t in range(len(r)):
            r_prev = r[t - 1] if t >= 1 else np.float32(0.0)
            h_prev = h[t - 1] if t >= 1 else np.float32(0.0)
            h[t] = omega + alpha * r_prev * r_prev + beta * h_prev
        return h
