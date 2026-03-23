# Audio applications can require large filters
# f(x) = f(x-1) * 1.8 + f(x-2) * -0.9 + g(x)  (applied twice in cascade)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class AudioLargeFiltersBenchmark(Benchmark):

    @property
    def name(self) -> str:
        return "audio-large-filters"

    @property
    def description(self) -> str:
        return "Cascaded second-order IIR: two passes of 1.8*f(x-1) - 0.9*f(x-2) + input"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            kernel = TIKernel([0, 1.8, -0.9])
            return Recurse(kernel, Recurse(kernel, Var("g")))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)
        # First pass: g1[i] = 1.8*g1[i-1] - 0.9*g1[i-2] + g[i]
        g1 = np.zeros(len(g), dtype=np.float32)
        for i in range(len(g)):
            g1_1 = g1[i - 1] if i >= 1 else 0.0
            g1_2 = g1[i - 2] if i >= 2 else 0.0
            g1[i] = np.float32(1.8) * g1_1 + np.float32(-0.9) * g1_2 + g[i]
        # Second pass: f[i] = 1.8*f[i-1] - 0.9*f[i-2] + g1[i]
        f = np.zeros(len(g), dtype=np.float32)
        for i in range(len(g)):
            f1 = f[i - 1] if i >= 1 else 0.0
            f2 = f[i - 2] if i >= 2 else 0.0
            f[i] = np.float32(1.8) * f1 + np.float32(-0.9) * f2 + g1[i]
        return f
