# f(x) = f(x-1) * 1.6 + f(x-2) * -0.64 + g(x) - 2 * g(x-1) + g(x-2)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class SecondOrderHighPassBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "second-order-high-pass"

    @property
    def description(self) -> str:
        return "Second-order high-pass: f(x) = 1.6*f(x-1) - 0.64*f(x-2) + g(x) - 2*g(x-1) + g(x-2)"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            return Recurse(
                TIKernel([0, 1.6, -0.64]),
                Convolve(TIKernel([1, -2, 1]), Var("g")),
            )

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)
        f = np.zeros(len(g), dtype=np.float32)
        for i in range(len(g)):
            f1 = f[i - 1] if i >= 1 else 0.0
            f2 = f[i - 2] if i >= 2 else 0.0
            g0 = g[i]
            g1 = g[i - 1] if i >= 1 else 0.0
            g2 = g[i - 2] if i >= 2 else 0.0
            f[i] = np.float32(1.6) * f1 + np.float32(-0.64) * f2 + g0 - np.float32(2.0) * g1 + g2
        return f
