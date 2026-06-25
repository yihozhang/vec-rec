# Repeated box filter: apply a box filter three times consecutively to approximate a Gaussian filter.
# Each pass uses a summed area table (SAT) via the inclusion-exclusion principle.

from __future__ import annotations

import numpy as np

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D
from vecrec.optimize import get_candidate_schedules
from vecrec.util import ElementType

from benchmark import Benchmark


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def diff_kernel(D: int) -> TIKernel2D:
    kernel = [[0.0] * D for i in range(D)]
    kernel[0][0] = kernel[D - 1][D - 1] = 1.0
    kernel[D - 1][0] = kernel[0][D - 1] = -1.0
    return TIKernel2D(kernel)


def naive_box_filter(input_signal: SignalExpr2D, D: int) -> SignalExpr:
    kernel = diff_kernel(D)

    def box_filter_sat(input_signal: SignalExpr2D, D: int) -> SignalExpr2D:
        with with_types(Type.Arith, ElementType.Float):
            df = Convolve2D(kernel, input_signal)
            df2 = PointwiseDiv(df, Num(float(D * D)))
            sat = Recurse2D(
                TIKernel2D([[0.0, 1.0], [1.0, -1.0]]),
                df2,
            )
            return sat

    box1 = box_filter_sat(input_signal, D)
    box2 = box_filter_sat(box1, D)
    expr = box_filter_sat(box2, D)
    return expr.at(0)


def _single_sat_pass_reference(g: np.ndarray, D: int) -> np.ndarray:
    """
    Reference for one pass of box_filter_sat on a 2D input.
    Returns a 2D array of the same shape as g.
    """
    rows, cols = g.shape
    g = g.astype(np.float32)
    scale = float(D * D)

    # Difference image
    fir = np.zeros_like(g)
    for y in range(rows):
        for x in range(cols):
            v = g[y, x] / scale
            if y >= D - 1 and x >= D - 1:
                v += g[y - (D - 1), x - (D - 1)] / scale
            if y >= D - 1:
                v -= g[y - (D - 1), x] / scale
            if x >= D - 1:
                v -= g[y, x - (D - 1)] / scale
            fir[y, x] = v

    # 2D prefix sum with extra -s[y-1,x-1] term:
    # s[y,x] = s[y,x-1] + s[y-1,x] - s[y-1,x-1] + fir[y,x]
    # (matches TIKernel2D([[0.0, 1.0], [1.0, -1.0]]))
    s = np.zeros_like(fir)
    for y in range(rows):
        for x in range(cols):
            s_left = s[y, x - 1] if x >= 1 else 0.0
            s_up = s[y - 1, x] if y >= 1 else 0.0
            s_upleft = s[y - 1, x - 1] if (y >= 1 and x >= 1) else 0.0
            s[y, x] = s_left + s_up - s_upleft + fir[y, x]

    return s.astype(np.float32)


class RepeatedBoxFilterBenchmark(Benchmark):
    """Three-pass box filter (Gaussian approximation), parameterised by D."""

    def __init__(self, D: int) -> None:
        self.D = D

    @property
    def name(self) -> str:
        return f"repeated-box-filter-D{self.D}"

    @property
    def description(self) -> str:
        return f"3-pass box filter (D={self.D}) approximating a Gaussian"

    @property
    def verify_n_rows_2d(self) -> int:
        # SAT kernel grows like Pascal's triangle; 4 rows keeps values in safe float32 range
        return 4
    
    def build_codegen(self) -> CodeGen:
        return CodeGen(ele_per_iter_multiple=2, rows_per_iter=2)

    @property
    def verify_rtol(self) -> float:
        # Vectorized float32 SAT reorders additions vs scalar reference; allow up to 5% relative error
        return 0.05

    def original_expr(self) -> SignalExpr:
        with with_types(Type.Arith, ElementType.Float):
            return naive_box_filter(Var2D("g"), self.D)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        g = inputs["g"].astype(np.float32)  # shape (rows, cols)
        # Three cascaded SAT box filter passes
        g1 = _single_sat_pass_reference(g, self.D)
        g2 = _single_sat_pass_reference(g1, self.D)
        g3 = _single_sat_pass_reference(g2, self.D)
        return g3.flatten()


def main() -> None:
    for D in [3, 5, 7, 9, 11, 13, 15]:
        print(f"D = {D}")

        with with_types(Type.Arith, ElementType.Float):
            expr = naive_box_filter(Var2D("g"), D)

        results = get_candidate_schedules(expr, 256)

        benchmark_result = generate_and_run_benchmark(CodeGen(), results, True)
        print(benchmark_result)


if __name__ == "__main__":
    main()
