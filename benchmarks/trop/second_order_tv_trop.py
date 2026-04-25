# f(x) = max(f(x-1) + a(x), f(x-2) + b(x), g(x)) in TropMax semiring

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class _SecondOrderTVTropBenchmark(Benchmark):
    """Base class for second-order TV TropMax benchmarks parameterized by element type."""

    _element_type: ElementType

    @property
    def name(self) -> str:
        return f"second-order-tv-trop-{self._element_type.name.lower()}"

    @property
    def description(self) -> str:
        return f"Second-order TV recurrence in TropMax (max-plus) semiring ({self._element_type.name})"

    def original_expr(self) -> Recurse:
        with with_types(Type.TropMax, self._element_type):
            return Recurse(TVKernel([Num(Type.TropMax.zero()), Var("a"), Var("b")]), Var("g"))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        et = self._element_type
        np_dtype = {ElementType.I8: np.int8, ElementType.I32: np.int32, ElementType.I64: np.int64}[et]
        np_iinfo = np.iinfo(np_dtype) # type: ignore

        g = inputs["g"].astype(np_dtype)
        a = inputs["a"].astype(np_dtype)
        b = inputs["b"].astype(np_dtype)

        if et == ElementType.I8:
            NEGINF = int(np_iinfo.min) // 2
            INT_MIN = int(np_iinfo.min)
            INT_MAX = int(np_iinfo.max)

            def add(x: int, y: int) -> int:
                return max(INT_MIN, min(INT_MAX, x + y))
        else:
            NEGINF = int(np_iinfo.min) // 8

            def add(x: int, y: int) -> int:
                return x + y

        f = np.full(len(g), NEGINF, dtype=np_dtype)
        for i in range(len(g)):
            terms: list[int] = [int(g[i])]
            if i >= 1 and f[i - 1] != NEGINF:
                terms.append(add(int(f[i - 1]), int(a[i])))
            if i >= 2 and f[i - 2] != NEGINF:
                terms.append(add(int(f[i - 2]), int(b[i])))
            f[i] = np_dtype(max(terms))
        return f


class SecondOrderTVTropI8Benchmark(_SecondOrderTVTropBenchmark):
    _element_type = ElementType.I8


class SecondOrderTVTropI32Benchmark(_SecondOrderTVTropBenchmark):
    _element_type = ElementType.I32


class SecondOrderTVTropI64Benchmark(_SecondOrderTVTropBenchmark):
    _element_type = ElementType.I64


# Keep old name as alias for the I8 variant (backward compat with run_all imports)
SecondOrderTVTropBenchmark = SecondOrderTVTropI8Benchmark
