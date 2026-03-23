"""First-order high-pass filter benchmark.

f(x) = a * (f(x-1) + g(x) - g(x-1))
"""

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class FirstOrderHighPassBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "first-order-high-pass"

    @property
    def description(self) -> str:
        return f"First-order high-pass filter: f(x) = {self.a}*(f(x-1) + g(x) - g(x-1))"

    a = 0.8

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            return Recurse(
                TIKernel([0, self.a]),
                Convolve(TIKernel([self.a, -self.a]), Var("g")),
            )

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)
        f = np.zeros(len(g), dtype=np.float32)
        for i in range(len(g)):
            f_prev = f[i - 1] if i >= 1 else 0.0
            g_prev = g[i - 1] if i >= 1 else 0.0
            f[i] = self.a * (f_prev + g[i] - g_prev)
        return f
