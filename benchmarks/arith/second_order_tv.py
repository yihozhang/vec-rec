# f(x) = f(x-1) * 1.8 + f(x-2) * -0.9 + g(x)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class _SecondOrderTVBenchmark(Benchmark):

    _element_type: ElementType

    @property
    def name(self) -> str:
        return f"second-order-tv-arith-{self._element_type.name.lower()}"

    @property
    def description(self) -> str:
        return f"Second-order TV IIR: f(x) = a(x)*f(x-1) + b(x)*f(x-2) + g(x) ({self._element_type.name})"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, self._element_type):
            return Recurse(TVKernel([Num(0), Var("a"), Var("b")]), Var("g"))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        et = self._element_type
        np_dtype = {
            ElementType.I8: np.int8,
            ElementType.I32: np.int32,
            ElementType.I64: np.int64,
            ElementType.Float: np.float32,
        }[et]

        g = inputs["g"].astype(np_dtype)
        a = inputs["a"].astype(np_dtype)
        b = inputs["b"].astype(np_dtype)
        f = np.zeros(len(g), dtype=np_dtype)
        for i in range(len(g)):
            f1 = f[i - 1] if i >= 1 else 0.0
            f2 = f[i - 2] if i >= 2 else 0.0
            f[i] = a[i] * f1 + b[i] * f2 + g[i]
        return f


class SecondOrderTVArithI8Benchmark(_SecondOrderTVBenchmark):
    _element_type = ElementType.I8


class SecondOrderTVArithI32Benchmark(_SecondOrderTVBenchmark):
    _element_type = ElementType.I32


class SecondOrderTVArithI64Benchmark(_SecondOrderTVBenchmark):
    _element_type = ElementType.I64


class SecondOrderTVArithFloatBenchmark(_SecondOrderTVBenchmark):
    _element_type = ElementType.Float
