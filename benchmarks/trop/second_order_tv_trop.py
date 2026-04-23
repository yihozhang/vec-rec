# f(x) = max(f(x-1) + a(x), f(x-2) + b(x), g(x)) in TropMax semiring

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class SecondOrderTVTropBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "second-order-tv-trop"

    @property
    def description(self) -> str:
        return "Second-order TV recurrence in TropMax (max-plus) semiring"

    def original_expr(self) -> Recurse:
        with with_types(Type.TropMax, ElementType.I64):
            return Recurse(TVKernel([Num(Type.TropMax.zero()), Var("a"), Var("b")]), Var("g"))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.int64)
        a = inputs["a"].astype(np.int64)
        b = inputs["b"].astype(np.int64)
        NEGINF = np.iinfo(np.int64).min // 2  # TropMax zero element

        f = np.full(len(g), NEGINF, dtype=np.int64)
        for i in range(len(g)):
            terms: list[int] = [int(g[i])]
            if i >= 1 and f[i - 1] != NEGINF:
                terms.append(int(f[i - 1]) + int(a[i]))
            if i >= 2 and f[i - 2] != NEGINF:
                terms.append(int(f[i - 2]) + int(b[i]))
            f[i] = np.int64(max(terms))
        return f
