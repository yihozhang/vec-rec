# Recursive Gaussian filter (Young & van Vliet 1995), causal part only.
#
# Approximates Gaussian convolution with a third-order IIR. Coefficients are
# derived from sigma:
#   sigma <  2.5: q = 1.31564 * (sqrt(1 + 0.490811*sigma^2) - 1)
#   sigma >= 2.5: q = 0.98711*sigma - 0.96330
#   b0 = 1.57825 + 2.44413*q + 1.4281*q^2 + 0.422205*q^3
#   b1 = 2.44413*q + 2.85619*q^2 + 1.26661*q^3
#   b2 = -1.4281*q^2 - 1.26661*q^3
#   b3 = 0.422205*q^3
#   a_k = b_k / b0
#   B   = 1 - (a1 + a2 + a3)
#
# Causal recurrence:
#   f(x) = a1*f(x-1) + a2*f(x-2) + a3*f(x-3) + B*g(x)

from __future__ import annotations

import math

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark


SIGMA = 2.0
if SIGMA < 2.5:
    _q = 1.31564 * (math.sqrt(1.0 + 0.490811 * SIGMA * SIGMA) - 1.0)
else:
    _q = 0.98711 * SIGMA - 0.96330
_b0 = 1.57825 + 2.44413 * _q + 1.4281 * _q**2 + 0.422205 * _q**3
_b1 = 2.44413 * _q + 2.85619 * _q**2 + 1.26661 * _q**3
_b2 = -1.4281 * _q**2 - 1.26661 * _q**3
_b3 = 0.422205 * _q**3
A1 = _b1 / _b0
A2 = _b2 / _b0
A3 = _b3 / _b0
B = 1.0 - (A1 + A2 + A3)


class RecursiveGaussianBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "recursive-gaussian"

    @property
    def description(self) -> str:
        return f"Causal recursive Gaussian (Young & van Vliet, sigma={SIGMA}): third-order IIR"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            return Recurse(
                TIKernel([0, A1, A2, A3]),
                Convolve(TIKernel([B]), Var("g")),
            )

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)
        a1 = np.float32(A1)
        a2 = np.float32(A2)
        a3 = np.float32(A3)
        b = np.float32(B)
        f = np.zeros(len(g), dtype=np.float32)
        for i in range(len(g)):
            f1 = f[i - 1] if i >= 1 else np.float32(0.0)
            f2 = f[i - 2] if i >= 2 else np.float32(0.0)
            f3 = f[i - 3] if i >= 3 else np.float32(0.0)
            f[i] = a1 * f1 + a2 * f2 + a3 * f3 + b * g[i]
        return f
