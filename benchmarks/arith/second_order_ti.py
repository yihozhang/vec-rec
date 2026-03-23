# f(x) = f(x-1) * 1.8 + f(x-2) * -0.9 + g(x)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class SecondOrderTIBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "second-order-ti"

    @property
    def description(self) -> str:
        return "Second-order TI IIR: f(x) = 1.8*f(x-1) - 0.9*f(x-2) + g(x)"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            return Recurse(TIKernel([0, 1.8, -0.9]), Var("g"))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)
        f = np.zeros(len(g), dtype=np.float32)
        for i in range(len(g)):
            f1 = f[i - 1] if i >= 1 else 0.0
            f2 = f[i - 2] if i >= 2 else 0.0
            f[i] = np.float32(1.8) * f1 + np.float32(-0.9) * f2 + g[i]
        return f
