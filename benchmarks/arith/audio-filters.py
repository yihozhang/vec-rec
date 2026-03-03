# Compare dilation/delay scheduling on two recurrence topologies:
#   1. Long-tap: single Recurse with many feedback taps (high-order kernel)
#   2. Cascade:  nested Recurse nodes, each with small (1st/2nd order) kernels

import random
from typing import List, Sequence, Tuple

from vecrec.expr import *
from vecrec.expr import Type
from vecrec.transform import *
from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.util import ElementType

SEED = 42
LANES = 512
MAX_KERNEL_ORDER = 128


def make_long_tap(order: int, rng: random.Random) -> SignalExpr:
    """Single Recurse whose kernel has `order` feedback taps."""
    taps: List[float] = [0.0] + [round(rng.uniform(-0.3, 0.3), 4) for _ in range(order)]
    return Recurse(
        TIKernel(taps, Type.Arith, ElementType.Float),
        Var("g", Type.Arith, ElementType.Float),
    )


def make_cascade(n_stages: int, stage_order: int, rng: random.Random) -> SignalExpr:
    """Chain of `n_stages` nested Recurse nodes, each with `stage_order` taps."""
    g: SignalExpr = Var("g", Type.Arith, ElementType.Float)
    for _ in range(n_stages):
        taps: List[float] = [0.0] + [
            round(rng.uniform(-0.3, 0.3), 4) for _ in range(stage_order)
        ]
        g = Recurse(TIKernel(taps, Type.Arith, ElementType.Float), g)
    return g


def make_long_tap_schedule(n_dilations: int, n_delays: int) -> Transform:
    """Schedule for a single top-level Recurse (long-tap case)."""
    steps: list[Transform] = []
    for _ in range(n_dilations):
        steps += [Dilate(), SaturatePreorder(ConstantFold)]
    for _ in range(n_delays):
        steps += [Delay(), SaturatePreorder(ConstantFold)]
    if not steps:
        return Seq(Noop(), AnnotateLanes(LANES), PushDownConvertLanes())
    return Seq(*steps, AnnotateLanes(LANES), PushDownConvertLanes())


# TODO, simplify this to just use preorder command instead


class ApplyToAllRecurse(Transform):
    """Apply a transform to every Recurse node in the tree, producing exactly 1 result.
    Bottom-up traversal that takes only the first result at each node."""

    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def _walk_signal(self, expr: SignalExpr) -> SignalExpr:
        if expr.is_leaf():
            return expr
        new_children = [self._walk_generic(c) for c in expr.children()] # type: ignore
        rebuilt = type(expr)(*new_children)  # type: ignore
        rebuilt.lanes = expr.lanes
        results = self.transform.apply_signal(rebuilt)
        return results[0] if results else rebuilt

    def _walk_kernel(self, expr: KernelExpr) -> KernelExpr:
        if isinstance(expr, (TIKernel, TVKernel)):
            return expr
        new_children = [self._walk_generic(c) for c in expr.children()] # type: ignore
        rebuilt = type(expr)(*new_children)  # type: ignore
        rebuilt.lanes = expr.lanes
        return rebuilt

    def _walk_generic(self, expr: SignalExpr | KernelExpr) -> SignalExpr | KernelExpr:
        if isinstance(expr, KernelExpr):
            return self._walk_kernel(expr)
        return self._walk_signal(expr)

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        return [self._walk_signal(expr)]


def make_cascade_schedule(n_dilations: int, n_delays: int) -> Transform:
    """Schedule for cascaded Recurse nodes -- applies uniformly to every stage."""
    steps: list[Transform] = []
    cf = ApplyToAllRecurse(Try(ConstantFold))
    for _ in range(n_dilations):
        steps += [ApplyToAllRecurse(Try(Dilate())), cf]
    for _ in range(n_delays):
        steps += [ApplyToAllRecurse(Try(Delay())), cf]
    if not steps:
        return Seq(Noop(), AnnotateLanes(LANES), PushDownConvertLanes())
    return Seq(*steps, AnnotateLanes(LANES), PushDownConvertLanes())


def schedule_label(n_dilations: int, n_delays: int) -> str:
    return f"{n_dilations}D+{n_delays}d"


def run_config(
    topology: str,
    params: str,
    program: SignalExpr,
    transforms: Transform,
    sched_label: str,
) -> List[Tuple[str, str, str, str, str]]:
    import sys

    print(
        f"  [{topology} | {params} | {sched_label}] transforming...",
        end="",
        flush=True,
        file=sys.stderr,
    )
    results = transforms.apply_signal(program)
    print(
        f" {len(results)} variant(s), compiling...", end="", flush=True, file=sys.stderr
    )
    codegen = CodeGen()
    benchmark_result = generate_and_run_benchmark(
        codegen, results, include_correctness_check=True
    )
    assert benchmark_result["return_code"] == 0, (
        f"Benchmark failed for {topology} {params} {sched_label}: "
        f"{benchmark_result.get('error', '')}"
    )
    rows: List[Tuple[str, str, str, str, str]] = []
    output = benchmark_result["output"]
    for variant_name, time_ns in output.items():
        if variant_name == "validation":
            continue
        rows.append((topology, params, sched_label, variant_name, str(time_ns)))
    print(f" done", file=sys.stderr)
    return rows


def main() -> None:
    long_tap_orders = [4, 8, 16, 32]
    # PushDownConvertLanes produces 2^K variants for K cascade stages,
    # so we keep stages small to avoid exponential compile times.
    cascade_stages = [2, 3, 4]
    cascade_order = 2

    # (n_dilations, n_delays) pairs to sweep
    schedules = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1), (3, 1)]

    header = f"{'topology':<12} | {'params':<20} | {'schedule':<8} | {'variant':<10} | {'time_ns'}"
    sep = "-" * len(header)

    all_rows: List[Tuple[str, str, str, str, str]] = []

    # --- Long-tap sweep ---
    for order in long_tap_orders:
        rng = random.Random(SEED)
        program = make_long_tap(order, rng)
        for n_dil, n_del in schedules:
            estimated_order = order * (2**n_dil)
            if estimated_order > MAX_KERNEL_ORDER:
                continue
            sched = make_long_tap_schedule(n_dil, n_del)
            label = schedule_label(n_dil, n_del)
            rows = run_config("long_tap", f"order={order}", program, sched, label)
            all_rows.extend(rows)

    # --- Cascade sweep ---
    for n_stages in cascade_stages:
        rng = random.Random(SEED)
        program = make_cascade(n_stages, cascade_order, rng)
        for n_dil, n_del in schedules:
            sched = make_cascade_schedule(n_dil, n_del)
            label = schedule_label(n_dil, n_del)
            rows = run_config(
                "cascade",
                f"stages={n_stages},order={cascade_order}",
                program,
                sched,
                label,
            )
            all_rows.extend(rows)

    # --- Print results ---
    print(sep)
    print(header)
    print(sep)
    for topo, params, sched_l, variant, time in all_rows:
        print(f"{topo:<12} | {params:<20} | {sched_l:<8} | {variant:<10} | {time}")
    print(sep)


if __name__ == "__main__":
    main()
