# Domain transform filter (Gastal & Oliveira 2011)
#
# Transforms bilateral filtering into 1D filtering via a cumulative integral:
#   t[i] = t[i-1] + sqrt(1 + c^2 * (x[i] - x[i-1])^2)
#
# where c = sigma_s / sigma_r.
#
# The recurrence is a first-order IIR with coefficient 1:
#   t[i] = 1.0 * t[i-1] + g[i]
# where g[i] = sqrt(1 + c^2 * (x[i] - x[i-1])^2) is pre-computed.
#
# VecRec handles the IIR recurrence part; the g computation is done in Python
# and passed as a precomputed input (Var("g")).

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark

# Typical bilateral filter parameters
SIGMA_S = 16.0   # spatial standard deviation
SIGMA_R = 0.1    # range standard deviation
C = SIGMA_S / SIGMA_R


class DomainTransformBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "domain-transform"

    @property
    def description(self) -> str:
        return "Domain transform for bilateral filtering: t[i] = t[i-1] + sqrt(1 + c^2*(dx)^2)"

    def original_expr(self) -> Recurse:
        with with_types(Type.Arith, ElementType.Float):
            # g[i] = sqrt(1 + c^2 * (x[i] - x[i-1])^2) is pre-computed externally
            g = Var("g")
            # t[i] = 1.0 * t[i-1] + g[i]
            return Recurse(TIKernel([0.0, 1.0]), g)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)
        # t[i] = t[i-1] + g[i]  (cumsum of g)
        return np.cumsum(g).astype(np.float32)
