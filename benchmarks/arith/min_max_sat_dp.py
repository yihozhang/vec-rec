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
from vecrec.semiring import CustomSemiring
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
    add_idempotent=True,  # assuming all values >= 1, plus(a,b) = max(min(a,b),1) = min(a,b)
)


class MinMaxSatDPBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "min-max-sat-dp"

    @property
    def description(self) -> str:
        return "DP: f(i,j) = max(min(f(i-1,j)-g(j), f(i,j-1)-g(j)), 1)  [MinMaxSat semiring]"

    def original_expr(self) -> Repeater:
        with with_types(MinMaxSat, ElementType.Float):
            g = Var("g")

            def row_func(prev_rows: RVar2D) -> Recurse:
                f_prev = Ith(prev_rows, 0)
                # MinMaxSat has no real additive inverse, so the subtraction and
                # negation must happen in the arithmetic semiring; we then view
                # the results back as MinMaxSat for the recurrence itself.
                f_prev_arith = ChangeSemiring(f_prev, Type.Arith)
                g_arith = ChangeSemiring(g, Type.Arith)
                signal = ChangeSemiring(SSub(f_prev_arith, g_arith), MinMaxSat)
                neg_g = ChangeSemiring(SNeg(g_arith), MinMaxSat)
                kernel = TVKernel([Num(float("inf")), neg_g, neg_g])
                return Recurse(kernel, signal)

            return Repeater(row_func, MinMaxSat, ElementType.Float)

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
