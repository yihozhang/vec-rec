"""Base class and registry for VecRec benchmarks."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from vecrec.codegen import CodeGen, generate_and_run_benchmark, generate_kernel_executable
from vecrec.expr.base import SignalExpr, SignalExpr2D
from vecrec.expr.pretty import pp
from vecrec.optimize import get_candidate_schedules
from vecrec.util import Colors, ElementType, get_native_bits


# ── Result type ──────────────────────────────────────────────────────────────


@dataclasses.dataclass
class BenchmarkResult:
    name: str
    description: str
    original_expr_pp: str  # pp() of the unoptimized base expression
    timing: dict[str, float]  # pretty-printed schedule → throughput (MB/s)
    validation_ok: bool  # every schedule value is True or int >= 256
    validation_detail: dict[str, Any]  # raw validation dict (e.g. {"k1": True, "k2": 342})
    ref_verified: bool  # verify_schedule(schedules[0]) passed
    best_verified: bool  # verify_schedule(best_schedule) passed
    ref_error: str  # empty on success
    best_error: str  # empty on success
    best_schedule_pp: str  # pp() of the fastest schedule


# ── Base class ────────────────────────────────────────────────────────────────

native_bits = get_native_bits()
class Benchmark(ABC):
    """Abstract base class for all VecRec benchmarks."""

    DEBUG: bool = False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def original_expr(self) -> SignalExpr | SignalExpr2D:
        """Return the unoptimized base expression (used for pp() in results)."""
        ...

    def build_schedules(self) -> list[SignalExpr] | list[SignalExpr2D]:
        """Return candidate schedules. Default: get_candidate_schedules on original_expr."""
        return get_candidate_schedules(self.original_expr(), self.max_bits, debug=self.DEBUG)

    def build_codegen(self) -> CodeGen:
        """Return the CodeGen instance to use. Override to customize ele_per_iter, rows_per_iter, etc."""
        return CodeGen()

    @property
    def max_bits(self) -> int:
        """Hardware SIMD width"""
        return native_bits

    @property
    def verify_atol(self) -> float:
        """Absolute tolerance for verify_schedule. Override for 2D benchmarks."""
        return 1e-3

    @property
    def verify_rtol(self) -> float:
        """Relative tolerance for verify_schedule. Override for 2D benchmarks."""
        return 1e-3

    @property
    def verify_n_rows_2d(self) -> int:
        """Number of 2D rows to use in verify_schedule. Override to reduce for recursive 2D kernels."""
        return 16

    @abstractmethod
    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        """
        Pure-Python reference implementation.

        Args:
            inputs: dict of variable name → numpy array (may be empty for constant inputs).
            n: output length (= row_size for 1D; rows*cols for 2D).

        Returns:
            numpy array of the expected output values.
        """
        ...

    def verify_schedule(self, expr: SignalExpr | SignalExpr2D) -> tuple[bool, str]:
        """
        Verify a schedule by running generate_kernel_executable and comparing to
        _python_reference on auto-generated random inputs.

        Returns (ok, error_message).
        """
        try:
            codegen = self.build_codegen()
            runner = generate_kernel_executable(codegen, expr, row_size=256)

            n_1d = runner.row_size  # 1024 by default
            n_rows_2d = self.verify_n_rows_2d

            inputs: dict[str, np.ndarray] = {}
            for var in runner.variable_order:
                inputs[var] = _sample(runner.variable_types[var], (n_1d,))
            for var in runner.vars2d_order:
                inputs[var] = _sample(runner.vars2d_info[var], (n_rows_2d, n_1d))

            # Determine output length and run
            if runner.vars2d_order:
                n_out = n_rows_2d * n_1d
            else:
                n_out = n_1d

            kernel_out, _ = runner.run(inputs)
            expected = self._python_reference(inputs, n_out)

            # Trim to the shorter of the two (handles any alignment/padding differences)
            min_len = min(len(kernel_out), len(expected))
            kernel_out = kernel_out[:min_len]
            expected = expected[:min_len]

            if not np.allclose(kernel_out, expected, atol=self.verify_atol, rtol=self.verify_rtol, equal_nan=False):
                bad = np.where(~np.isclose(kernel_out, expected, atol=self.verify_atol, rtol=self.verify_rtol))[0]
                first_diff = int(bad[0])
                return False, (
                    f"Mismatch at index {first_diff}: "
                    f"got {float(kernel_out[first_diff]):.6f}, "
                    f"expected {float(expected[first_diff]):.6f}"
                )
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def run(self) -> BenchmarkResult:
        """Run the full benchmark: timing, validation, and schedule verification."""
        codegen = self.build_codegen()
        schedules = self.build_schedules()
        original = self.original_expr()

        # Run all schedules with cross-schedule correctness check
        raw = generate_and_run_benchmark(codegen, schedules, include_correctness_check=True)
        output = raw["output"]

        # Check validation: half of the schedules are accurate enough
        # (vacuously True when there is only one schedule and no cross-schedule comparisons)
        validation_raw: dict[str, Any] = output.get("validation", {})
        if len(validation_raw) <= 1:
            print(f"{Colors.WARNING}WARNING: no cross-schedule validation data found. Skipping validation check.{Colors.ENDC}")
        validation_ok = len(validation_raw) == 0 or (
            sum([v is True or (isinstance(v, int) and v >= 256) for v in validation_raw.values()])
        ) > 0.5 * len(validation_raw)

        # Find best (fastest) schedule: highest throughput
        timing_keys = [k for k in output if k != "validation"]
        best_key = max(timing_keys, key=lambda k: output[k])
        best_idx = int(best_key[1:])  # "k3" → 3
        best_expr = schedules[best_idx]

        # Build timing dict keyed by pretty-printed expression
        timing = {pp(schedules[int(k[1:])]): output[k] for k in timing_keys}

        # Verify reference (k0) and best schedule against Python reference
        ref_ok, ref_err = self.verify_schedule(schedules[0])
        if best_idx == 0:
            best_ok, best_err = ref_ok, ref_err
        else:
            best_ok, best_err = self.verify_schedule(best_expr)

        return BenchmarkResult(
            name=self.name,
            description=self.description,
            original_expr_pp=pp(original),
            timing=timing,
            validation_ok=validation_ok,
            validation_detail=dict(validation_raw),
            ref_verified=ref_ok,
            best_verified=best_ok,
            ref_error=ref_err,
            best_error=best_err,
            best_schedule_pp=pp(best_expr),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sample(elt: ElementType, shape: tuple[int, ...]) -> np.ndarray:
    """Generate random test data appropriate for the given element type."""
    match elt:
        case ElementType.Float:
            return np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        case ElementType.I32:
            return np.random.randint(-1000, 1000, shape, dtype=np.int32)
        case ElementType.I64:
            # Non-negative for tropical operations
            return np.random.randint(0, 1000, shape, dtype=np.int64)
        case ElementType.I8:
            return np.random.randint(-1, 1, shape, dtype=np.int8)
