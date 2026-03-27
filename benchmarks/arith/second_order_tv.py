# f(x) = f(x-1) * 1.8 + f(x-2) * -0.9 + g(x)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class SecondOrderTVBenchmark(Benchmark):
    DEBUG = True
    @property
    def name(self) -> str:
        return "second-order-tv"

    @property
    def description(self) -> str:
        return "Second-order TV IIR: f(x) = a(x)*f(x-1) + b(x)*f(x-2) + g(x)"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            return Recurse(TVKernel([Num(0), Var("a"), Var("b")]), Var("g"))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)
        a = inputs["a"].astype(np.float32)
        b = inputs["b"].astype(np.float32)
        f = np.zeros(len(g), dtype=np.float32)
        for i in range(len(g)):
            f1 = f[i - 1] if i >= 1 else 0.0
            f2 = f[i - 2] if i >= 2 else 0.0
            f[i] = a[i] * f1 + b[i] * f2 + g[i]
        return f
