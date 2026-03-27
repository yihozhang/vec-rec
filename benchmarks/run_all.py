#!/usr/bin/env python3
"""Run all registered VecRec benchmarks.

Usage:
    uv run python benchmarks/run_all.py [FILTER ...] [--verbose]

Arguments:
    FILTER      Optional substrings. Only benchmarks whose name contains
                ALL provided substrings are run. If omitted, all benchmarks run.

Examples:
    uv run python benchmarks/run_all.py
    uv run python benchmarks/run_all.py prefix-sum
    uv run python benchmarks/run_all.py trop
    uv run python benchmarks/run_all.py box-filter D5
    uv run python benchmarks/run_all.py --verbose
"""

from __future__ import annotations

import sys
import traceback
from benchmark import Benchmark, BenchmarkResult
from arith.prefix_sum import PrefixSumBenchmark

# from arith.first_order_ti import FirstOrderTIConstBenchmark, FirstOrderTIVarBenchmark
from arith.first_order_high_pass import FirstOrderHighPassBenchmark
from arith.second_order_ti import SecondOrderTIBenchmark
from arith.second_order_tv import SecondOrderTVBenchmark
from arith.second_order_high_pass import SecondOrderHighPassBenchmark
from arith.audio_large_filters import AudioLargeFiltersBenchmark

from arith.continued_fraction import ContinuedFractionBenchmark
from arith.box_filter_summed_area_table import BoxFilterSATBenchmark
from arith.repeated_box_filter import RepeatedBoxFilterBenchmark
from trop.second_order_ti_trop import SecondOrderTITropBenchmark
from trop.second_order_tv_trop import SecondOrderTVTropBenchmark
from trop.distance_transform import DistanceTransformBenchmark


REGISTRY: list[Benchmark] = []


REGISTRY.extend(
    [
        PrefixSumBenchmark(),
        # FirstOrderTIConstBenchmark(),
        # FirstOrderTIVarBenchmark(),
        FirstOrderHighPassBenchmark(),
        SecondOrderTIBenchmark(),
        SecondOrderHighPassBenchmark(),
        SecondOrderTVBenchmark(),
        AudioLargeFiltersBenchmark(),
        ContinuedFractionBenchmark(),
        *[BoxFilterSATBenchmark(D) for D in [3, 5, 7, 9, 11, 13, 15]],
        *[RepeatedBoxFilterBenchmark(D) for D in [3, 5, 7, 9, 11, 13, 15]],
        SecondOrderTITropBenchmark(),
        SecondOrderTVTropBenchmark(),
        DistanceTransformBenchmark(),
    ]
)


def main() -> None:
    # Simple argument parsing: --verbose flag + optional filter substrings
    args = sys.argv[1:]
    verbose = "--verbose" in args
    filters = [a for a in args if not a.startswith("--")]
    if "--list" in args:
        print("Registered benchmarks:")
        for bench in REGISTRY:
            print(f"  {bench.name}: {bench.description}")
        sys.exit(0)

    benchmarks = list(REGISTRY)
    if filters:
        benchmarks = [b for b in benchmarks if all(f in b.name for f in filters)]

    if not benchmarks:
        print("No benchmarks matched.", file=sys.stderr)
        sys.exit(1)

    passed = 0
    failed = 0

    for bench in benchmarks:
        print(f"\n=== {bench.name} ===")
        print(f"  {bench.description}")
        try:
            result = bench.run()
            _print_result(result, verbose)
            ok = result.validation_ok and result.ref_verified and result.best_verified
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")
            if verbose:
                traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} total")
    sys.exit(1 if failed > 0 else 0)


def _print_result(result: BenchmarkResult, verbose: bool) -> None:
    # Original expression
    print(f"  Expr:    {result.original_expr_pp}")

    # Best schedule and throughput
    best_mb = result.timing.get(result.best_schedule_pp, 0.0)
    print(f"  Best:    {result.best_schedule_pp}  ({best_mb:.2f} MB/s)")

    if verbose:
        print("  Timing (all schedules):")
        for sched_pp, mb in result.timing.items():
            marker = " *" if sched_pp == result.best_schedule_pp else ""
            print(f"    {mb:>10.2f} MB/s  {sched_pp}{marker}")

    # Cross-schedule validation
    val_status = "PASS" if result.validation_ok else "FAIL"
    print(f"  Validation:    {val_status}")
    if verbose or not result.validation_ok:
        for k, v in result.validation_detail.items():
            print(f"    {k}: {v}")

    # Schedule verification (ref and best vs Python reference)
    ref_status = "PASS" if result.ref_verified else "FAIL"
    best_status = "PASS" if result.best_verified else "FAIL"
    print(f"  Ref verified:  {ref_status}" + (f"  [{result.ref_error}]" if result.ref_error else ""))
    print(f"  Best verified: {best_status}" + (f"  [{result.best_error}]" if result.best_error else ""))


if __name__ == "__main__":
    main()
