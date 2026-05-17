"""Minimal blocking benchmark.

Pipeline (single Recurse2D, kernel [[0, 1], [1, 0]], with a 3-tap moving
average on the input):
  smooth(x, y) = (g(x, y) + g(x-1, y) + g(x-2, y)) / 3
  result(x, y) = result(x-1, y) + result(x, y-1) + smooth(x, y)

The same-row tap result(x-1, y) is the prefix-sum-like component; the cross-row
tap result(x, y-1) is what blocking can amortize via in-block register reuse.

Schedule: only Eliminate2DKernels + InferNRows + AnnotateLanes + PushDownConvertLanes.
No dilation, no factorization.

Sweeps rows_per_iter to verify whether blocking provides speedup on this minimal
2D kernel.
"""

from __future__ import annotations

from vecrec.expr import *
from vecrec.expr.pretty import pp
from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.optimize import get_candidate_schedules
from vecrec.transform import (
    Eliminate2DKernels,
    InferNRows,
    AnnotateLanes,
    PushDownConvertLanes,
    Preorder,
    Try,
    ConstantFold,
    SaturatePreorder,
    Seq,
)
from vecrec.util import get_native_bits, ElementType


def build_expr() -> SignalExpr:
    with with_types(Type.Arith, ElementType.Float):
        g = Var2D("g")
        # 3-tap same-row moving average on g
        smooth = Convolve(TIKernel([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), g.at(0))
        # f(x, y) = f(x-1, y) + f(x, y-1) + smooth(x, y)
        result = Recurse2D(TIKernel2D([[0.0, 1.0], [1.0, 0.0]]), smooth)
    return result.at(0)


def minimal_schedule(expr: SignalExpr) -> SignalExpr:
    """Apply only the lowering needed for codegen, no dilation/factorization."""
    pipeline = Seq(
        Preorder(Try(Eliminate2DKernels())),
        SaturatePreorder(ConstantFold),
        InferNRows(),
        AnnotateLanes(get_native_bits()),
        PushDownConvertLanes(),
    )
    return pipeline.apply_generic(expr)[0]


def sweep_rpi(label: str, exprs: list[SignalExpr]) -> None:
    """Benchmark every expr in `exprs` across rpi values and print the best per rpi.

    Passing multiple exprs lets the full-schedule path compare its candidate
    schedules; the minimal path passes a single-element list."""
    print(f"=== {label} ===")
    if len(exprs) == 1:
        print(f"Schedule: {pp(exprs[0])}")
    else:
        print(f"{len(exprs)} candidate schedules; reporting best-of per rpi.")
    print()
    print(f"{'rpi':>5}  {'best MB/s':>10}  {'best schedule':>14}")
    print(f"{'-' * 5}  {'-' * 10}  {'-' * 14}")
    # for rpi in [1, 2, 3, 4, 6, 8]:
    for rpi in [1, 2, 3, 4]:
        cg = CodeGen(ele_per_iter_multiple=2, rows_per_iter=rpi)
        raw = generate_and_run_benchmark(cg, exprs, include_correctness_check=True, compiler_flags=["-ffast-math"])
        print(raw)
        timings = {k: v for k, v in raw["output"].items() if k != "validation"}
        best_key = max(timings, key=lambda k: timings[k])
        print(f"{rpi:>5}  {timings[best_key]:>10.0f}  {best_key:>14}")
    print()


def main() -> None:
    raw_expr = build_expr()

    # Path 1: minimal schedule (lower 2D kernels + lanes only).
    # minimal = minimal_schedule(raw_expr)
    # sweep_rpi("Minimal schedule", [minimal])

    # Path 2: full optimizer (dilation, factorization, etc.).
    full = list(get_candidate_schedules(raw_expr, 512))
    sweep_rpi("Full optimizer", full)


if __name__ == "__main__":
    main()
