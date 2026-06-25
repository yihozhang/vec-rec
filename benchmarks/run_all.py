#!/usr/bin/env python3
"""Run all registered VecRec benchmarks.

Usage:
    uv run python benchmarks/run_all.py [FILTER ...] [options]

Arguments:
    FILTER      Optional substrings. Only benchmarks whose name contains
                ALL provided substrings are run. If omitted, all benchmarks run.

Options:
    --verbose, -v          Print per-schedule timing, sweep timings, and validation detail.
    --list                 List registered benchmarks and exit.
    --save [DESCRIPTION]   Opt in to writing a JSON result file. The argument
                           is an optional human-readable description, embedded
                           in the file's metadata and sanitized into the
                           filename. Without --save, nothing is written.
    --output-dir DIR       Folder to write the JSON result file into.
                           Default: results/  (only used with --save)

Output filename format:
    YYYY-MM-DDTHH-MM-SS[__sanitized-description].json

Examples:
    uv run python benchmarks/run_all.py
    uv run python benchmarks/run_all.py prefix-sum
    uv run python benchmarks/run_all.py --save "experiment: tweaked PushConvolveBelowRecurse"
    uv run python benchmarks/run_all.py box-filter D5 --save "D-sweep, semiring v2"
    uv run python benchmarks/run_all.py --save --output-dir results/exp1
    uv run python benchmarks/run_all.py --list
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import traceback
from typing import Any
from benchmark import Benchmark, BenchmarkResult
from arith.prefix_sum import PrefixSumBenchmark

# from arith.first_order_ti import FirstOrderTIConstBenchmark, FirstOrderTIVarBenchmark
from arith.first_order_high_pass import FirstOrderHighPassBenchmark
from arith.second_order_ti import SecondOrderTIBenchmark
from arith.second_order_tv import (
    SecondOrderTVArithI8Benchmark,
    SecondOrderTVArithI32Benchmark,
    SecondOrderTVArithI64Benchmark,
    SecondOrderTVArithFloatBenchmark,
)
from arith.second_order_high_pass import SecondOrderHighPassBenchmark
from arith.audio_large_filters import AudioLargeFiltersBenchmark
from arith.recursive_gaussian import RecursiveGaussianBenchmark

from arith.continued_fraction import ContinuedFractionBenchmark
from arith.mutual_recurse import MutualRecurseTIFBenchmark, MutualRecurseTIGBenchmark
from arith.box_filter_summed_area_table import BoxFilterSATBenchmark
from arith.repeated_box_filter import RepeatedBoxFilterBenchmark
from trop.second_order_ti_trop import SecondOrderTITropBenchmark
from trop.second_order_tv_trop import (
    SecondOrderTVTropI8Benchmark,
    SecondOrderTVTropI32Benchmark,
    SecondOrderTVTropI64Benchmark,
)
from trop.needleman_wunsch import NeedlemanWunschBenchmark
from trop.distance_transform import DistanceTransformBenchmark
from trop.affine_gap import AffineGapBenchmark
from trop.edit_distance import EditDistanceBenchmark
from arith.min_max_sat_dp import MinMaxSatDPBenchmark

REGISTRY: list[Benchmark] = []


REGISTRY.extend(
    [
        PrefixSumBenchmark(),
        # FirstOrderTIConstBenchmark(),
        # FirstOrderTIVarBenchmark(),
        FirstOrderHighPassBenchmark(),
        SecondOrderTIBenchmark(),
        SecondOrderHighPassBenchmark(),
        SecondOrderTVArithI8Benchmark(),
        SecondOrderTVArithI32Benchmark(),
        SecondOrderTVArithI64Benchmark(),
        SecondOrderTVArithFloatBenchmark(),
        SecondOrderTITropBenchmark(),
        SecondOrderTVTropI8Benchmark(),
        SecondOrderTVTropI32Benchmark(),
        SecondOrderTVTropI64Benchmark(),
        AudioLargeFiltersBenchmark(),
        RecursiveGaussianBenchmark(),
        ContinuedFractionBenchmark(),
        *[BoxFilterSATBenchmark(D) for D in [3, 5, 7, 9, 11, 13, 15]],
        *[RepeatedBoxFilterBenchmark(D) for D in [3, 5, 7, 9, 11, 13, 15]],
        DistanceTransformBenchmark(),
        NeedlemanWunschBenchmark(),
        AffineGapBenchmark(),
        EditDistanceBenchmark(),
        MutualRecurseTIFBenchmark(),
        MutualRecurseTIGBenchmark(),
        MinMaxSatDPBenchmark(),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VecRec benchmarks and (optionally) record JSON results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("filters", nargs="*", help="Substrings; benchmark names must contain ALL of them.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-schedule timing.")
    parser.add_argument("--list", action="store_true", help="List registered benchmarks and exit.")
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        default=None,
        metavar="DESCRIPTION",
        help="Opt in to saving a JSON result file with an optional description.",
    )
    parser.add_argument("--output-dir", default="results", help="Folder for the JSON result file (default: results).")
    args = parser.parse_args()

    if args.list:
        print("Registered benchmarks:")
        for bench in REGISTRY:
            print(f"  {bench.name}: {bench.description}")
        sys.exit(0)

    benchmarks = list(REGISTRY)
    if args.filters:
        benchmarks = [b for b in benchmarks if all(f in b.name for f in args.filters)]

    if not benchmarks:
        print("No benchmarks matched.", file=sys.stderr)
        sys.exit(1)

    started_at = datetime.datetime.now()
    passed = 0
    failed = 0
    entries: list[dict[str, Any]] = []

    for bench in benchmarks:
        print(f"\n=== {bench.name} ===")
        print(f"  {bench.description}")
        try:
            result = bench.run()
            _print_result(result, args.verbose)
            ok = result.validation_ok and result.ref_verified and result.best_verified
            if ok:
                passed += 1
            else:
                failed += 1
            entries.append(_result_to_json(result))
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")
            if args.verbose:
                traceback.print_exc()
            failed += 1
            entries.append(
                {
                    "name": bench.name,
                    "description": bench.description,
                    "error": str(exc),
                }
            )

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} total")

    if args.save is not None:
        out_path = _save_results(
            output_dir=args.output_dir,
            description=args.save,
            filters=args.filters,
            started_at=started_at,
            entries=entries,
            passed=passed,
            failed=failed,
        )
        print(f"Saved results to {out_path}")

    sys.exit(1 if failed > 0 else 0)


def _result_to_json(result: BenchmarkResult) -> dict[str, Any]:
    """Serialize a BenchmarkResult into a JSON-friendly dict.

    `relative_to_best` is throughput / max(throughput); 1.0 marks the winner.
    """
    best_mb = result.timing.get(result.best_schedule_pp, 0.0)
    schedules = []
    for sched_pp, mb in result.timing.items():
        schedules.append(
            {
                "schedule": sched_pp,
                "throughput_mbs": mb,
                "relative_to_best": (mb / best_mb) if best_mb > 0 else 0.0,
                "is_best": sched_pp == result.best_schedule_pp,
            }
        )
    return {
        "name": result.name,
        "description": result.description,
        "original_expr": result.original_expr_pp,
        "best_schedule": result.best_schedule_pp,
        "best_throughput_mbs": best_mb,
        "best_ele_per_iter_multiple": result.best_ele_per_iter_multiple,
        "best_rows_per_iter": result.best_rows_per_iter,
        "sweep_timings_mbs": result.sweep_timings,
        "schedules": schedules,
        "validation_ok": result.validation_ok,
        "validation_detail": {k: v for k, v in result.validation_detail.items()},
        "ref_verified": result.ref_verified,
        "best_verified": result.best_verified,
        "ref_error": result.ref_error,
        "best_error": result.best_error,
    }


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_for_filename(text: str) -> str:
    """Lowercase, collapse runs of unsafe chars to single hyphens, trim."""
    cleaned = _FILENAME_SAFE_RE.sub("-", text.strip().lower()).strip("-_.")
    return cleaned[:80]


def _save_results(
    output_dir: str,
    description: str,
    filters: list[str],
    started_at: datetime.datetime,
    entries: list[dict[str, Any]],
    passed: int,
    failed: int,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stamp = started_at.strftime("%Y-%m-%dT%H-%M-%S")
    suffix = _sanitize_for_filename(description)
    filename = f"{stamp}__{suffix}.json" if suffix else f"{stamp}.json"
    out_path = os.path.join(output_dir, filename)

    payload = {
        "metadata": {
            "timestamp": started_at.isoformat(),
            "description": description,
            "filters": filters,
            "passed": passed,
            "failed": failed,
        },
        "benchmarks": entries,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _print_result(result: BenchmarkResult, verbose: bool) -> None:
    # Original expression
    print(f"  Expr:    {result.original_expr_pp}")

    # Best schedule and throughput
    best_mb = result.timing.get(result.best_schedule_pp, 0.0)
    print(f"  Best:    {result.best_schedule_pp}  ({best_mb:.2f} MB/s)")
    print(
        f"  Setup:   ele_per_iter_multiple={result.best_ele_per_iter_multiple}, "
        f"rows_per_iter={result.best_rows_per_iter}"
    )

    if verbose:
        print("  Timing (all schedules):")
        for sched_pp, mb in result.timing.items():
            marker = " *" if sched_pp == result.best_schedule_pp else ""
            print(f"    {mb:>10.2f} MB/s  {sched_pp}{marker}")

        if result.sweep_timings:
            print("  Sweep throughputs (MB/s, higher is better):")
            best_key = f"{result.best_ele_per_iter_multiple},{result.best_rows_per_iter}"
            for entry in result.sweep_timings:
                sched_pp = entry["schedule"]
                is_winning_sched = bool(entry.get("is_best"))
                print(f"    {sched_pp}{' *' if is_winning_sched else ''}")
                for cfg, mbs in entry["throughputs_mbs"].items():
                    epi, rpi = cfg.split(",")
                    marker = " *" if is_winning_sched and cfg == best_key else ""
                    mbs_str = f"{mbs:>10.2f} MB/s" if mbs is not None else f"{'FAIL':>15}"
                    print(f"      epi={epi}, rpi={rpi}: {mbs_str}{marker}")

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
