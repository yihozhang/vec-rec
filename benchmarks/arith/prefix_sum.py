# f(x) = f(x-1) + g(x)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class PrefixSumBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "prefix-sum"

    @property
    def description(self) -> str:
        return "Prefix sum: f(x) = f(x-1) + g(x)"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            return Recurse(TIKernel([0, 1]), Var("g"))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)
        f = np.zeros(len(g), dtype=np.float32)
        for i in range(len(g)):
            f[i] = (f[i - 1] if i >= 1 else 0.0) + g[i]
        return f
