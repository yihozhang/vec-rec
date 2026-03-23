# Fibonacci-like: f(x) = f(x-1) + f(x-2) [+ input]

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


class FirstOrderTIConstBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "first-order-ti-const"

    @property
    def description(self) -> str:
        return "Fibonacci-like IIR with constant input 1: f(x) = f(x-1) - f(x-2) + 1"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            return Recurse(TIKernel([0, 1.0, -1.0]), Num(1.0))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        f = np.zeros(n, dtype=np.float32)
        for i in range(n):
            f1 = f[i - 1] if i >= 1 else 0.0
            f2 = f[i - 2] if i >= 2 else 0.0
            f[i] = f1 - f2 + 1.0
        return f


class FirstOrderTIVarBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "first-order-ti-var"

    @property
    def description(self) -> str:
        return "Fibonacci-like IIR with variable input: f(x) = f(x-1) - f(x-2) + x"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            return Recurse(TIKernel([0, 1.0, -1.0]), Var("x"))

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        x = inputs["x"].astype(np.float32)
        f = np.zeros(len(x), dtype=np.float32)
        for i in range(len(x)):
            f1 = f[i - 1] if i >= 1 else 0.0
            f2 = f[i - 2] if i >= 2 else 0.0
            f[i] = f1 - f2 + x[i]
        return f
