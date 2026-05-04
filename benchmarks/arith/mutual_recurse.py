# Mutual 2×2 TI IIR recurrence:
#   f(x) = 0.6*f(x-1) + 0.2*g(x-1) + h1(x)
#   g(x) = 0.3*f(x-1) + 0.4*g(x-2) + h2(x)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.optimize import get_candidate_schedules
from vecrec.util import ElementType

from benchmark import Benchmark

def _make_mutual_recurse() -> tuple[SignalExpr, SignalExpr]:
    with with_types(Type.Arith, ElementType.Float):
        A = KernelMatrix(
            [
                [TIKernel([0.0, 0.6]), TIKernel([0.0, 0.2])],
                [TIKernel([0.0, 0.3]), TIKernel([0.0, 0.0, 0.4])],
            ]
        )
        G = SignalVector([Var("h1"), Var("h2")])
        F = RecurseV(A, G)
        f0 = VectorAt(F, 0)
        f1 = VectorAt(F, 1)
    return f0, f1


def _reference(h1: np.ndarray, h2: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    f = np.zeros(n, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    f_prev, g_prev1, g_prev2 = 0.0, 0.0, 0.0
    for x in range(n):
        new_f = 0.6 * f_prev + 0.2 * g_prev1 + h1[x]
        new_g = 0.3 * f_prev + 0.4 * g_prev2 + h2[x]
        g_prev2 = g_prev1
        g_prev1 = new_g
        f_prev = new_f
        f[x] = new_f
        g[x] = new_g
    return f, g


class MutualRecurseTIFBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "mutual-recurse-ti-f"

    @property
    def description(self) -> str:
        return "Mutual 2×2 TI IIR (f output): f(x)=0.6f(x-1)+0.2g(x-1)+h1, g(x)=0.3f(x-1)+0.4g(x-2)+h2"

    def original_expr(self) -> SignalExpr:
        f0, _ = _make_mutual_recurse()
        return f0

    def build_schedules(self) -> list:
        schedules = get_candidate_schedules(self.original_expr(), self.max_bits)
        print(f"Generated {len(schedules)} candidate schedules for {self.name}")
        return schedules

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        f, _ = _reference(inputs["h1"], inputs["h2"], n)
        return f


class MutualRecurseTIGBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "mutual-recurse-ti-g"

    @property
    def description(self) -> str:
        return "Mutual 2×2 TI IIR (g output): f(x)=0.6f(x-1)+0.2g(x-1)+h1, g(x)=0.3f(x-1)+0.4g(x-2)+h2"

    def original_expr(self) -> SignalExpr:
        _, f1 = _make_mutual_recurse()
        return f1

    def build_schedules(self) -> list:
        schedules = get_candidate_schedules(self.original_expr(), self.max_bits)
        print(f"Generated {len(schedules)} candidate schedules for {self.name}")
        return schedules

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        _, g = _reference(inputs["h1"], inputs["h2"], n)
        return g
