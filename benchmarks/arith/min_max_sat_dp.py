# DP recurrence using a custom semiring where plus(a, b) = max(min(a, b), 1).
#
# The DP is:
#   f(i, j) = max(min(f(i-1, j) - g(j), f(i, j-1) - g(j)), 1)
#
# Mapping to VecRec semiring (times = a+b, zero = +inf, one = 0):
#   signal  = f(i-1, j) - g(j)           [f_prev row minus g]
#   kernel  = [+inf, -g(j)]              [lag 0 = zero, lag 1 = -g]
#   recurse applies:  acc = add(signal, mul(-g(j), f(i,j-1)))
#                        = max(min(f(i-1,j)-g(j), f(i,j-1)-g(j)), 1)

from __future__ import annotations

import math

import numpy as np

from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D
from vecrec.semiring import CustomSemiring
from vecrec.transform import InferNRows, AnnotateLanes
from vecrec.util import ElementType

from benchmark import Benchmark


MinMaxSat = CustomSemiring(
    name="MinMaxSat",
    zero=float("inf"),
    one=0.0,
    plus=lambda a, b: max(min(a, b), 1.0),
    times=lambda a, b: a + b,
    is_zero_fn=lambda v: math.isinf(v) and v > 0,
    is_one_fn=lambda v: abs(v) < 1e-6,
    cpp_zero_init="std::numeric_limits<elt_type>::infinity()",
    cpp_mul_op="[](auto a, auto b) { return a + b; }",
    cpp_add_op="[](auto acc, auto x) { return __builtin_elementwise_max(__builtin_elementwise_min(acc, x), (decltype(acc))1); }",
    cpp_sadd_op="[](auto left, auto right) { return __builtin_elementwise_max(__builtin_elementwise_min(left, right), (decltype(left))1); }",
    cpp_pmul_op="[](auto left, auto right) { return left + right; }",
    cpp_ssub_op="[](auto left, auto right) { return left - right; }",
)


class MinMaxSatDPBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "min-max-sat-dp"

    @property
    def description(self) -> str:
        return "DP: f(i,j) = max(min(f(i-1,j)-g(j), f(i,j-1)-g(j)), 1)  [MinMaxSat semiring]"

    @property
    def max_bits(self) -> int:
        return 32  # scalar (lanes=1): TVKernel time_delay=1 forces scalar execution

    def original_expr(self) -> Repeater:
        with with_types(MinMaxSat, ElementType.Float):
            g = Var("g")

            def row_func(prev_rows: RVar2D) -> Recurse:
                f_prev = Ith(prev_rows, 0)
                signal = SSub(f_prev, g)
                kernel = TVKernel([Num(float("inf")), SNeg(g)])
                return Recurse(kernel, signal)

            return Repeater(row_func, MinMaxSat, ElementType.Float)

    def build_schedules(self) -> list[SignalExpr2D]:
        expr = self.original_expr()
        exprs = InferNRows().apply_generic(expr)
        assert len(exprs) == 1
        exprs2 = AnnotateLanes(self.max_bits).apply_signal2d(exprs[0])
        assert len(exprs2) == 1
        return [exprs2[0]]

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)
        row_size = len(g)
        n_rows = n // row_size
        f = np.full((n_rows, row_size), np.inf, dtype=np.float32)
        for i in range(n_rows):
            for j in range(row_size):
                top = f[i - 1, j] if i > 0 else np.inf
                left = f[i, j - 1] if j > 0 else np.inf
                val = max(min(float(top - g[j]), float(left - g[j])), 1.0)
                f[i, j] = val
        return f.ravel()
