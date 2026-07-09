"""Tests for CustomPointwiseOp — user-defined binary pointwise operators.

A custom pointwise op is defined entirely by a benchmark-owned C++ struct: the
struct name and its full definition (`cpp_prelude`) travel with the AST node, the
same self-describing pattern as ComputedSignal2D but for a binary op on two
signals. Because it subclasses SignalExprBinOp it flows through every
transform/schedule/lane-annotation pass like a built-in binop; only codegen
differs (custom struct name, no semiring suffix, prelude emitted once).

The example operator is PointwiseModulo (`a % b`), valid for integer element
types. The second operand must be nonzero (integer division by zero is UB and
traps on x86), so these tests keep divisors nonzero.
"""

import numpy as np
import pytest

from vecrec.expr import (
    CustomPointwiseOp,
    Num,
    PointwiseModulo,
    Recurse,
    TIKernel,
    Type,
    Var,
    pp,
    with_types,
)
from vecrec.optimize import get_candidate_schedules
from vecrec.transform import AnnotateLanes, PushDownConvertLanes, Seq
from vecrec.util import ElementType, get_native_bits
from vecrec.codegen import CodeGen, generate_and_run_benchmark, generate_kernel_executable


def _annotate(expr):
    return Seq(AnnotateLanes(get_native_bits()), PushDownConvertLanes()).apply_signal(expr)[0]


def test_pointwise_modulo_is_custom_pointwise_op() -> None:
    """PointwiseModulo is the extensible base's example subclass, carrying its C++
    struct name and definition as class attributes (not AST children)."""
    with with_types(Type.Arith, ElementType.I32):
        expr = PointwiseModulo(Var("x"), Var("y"))
    assert isinstance(expr, CustomPointwiseOp)
    assert expr.cpp_struct == "PointwiseModulo"
    assert "struct PointwiseModulo" in expr.cpp_prelude
    # Children are just the two operands; the C++ metadata is out-of-band, so the
    # generic type(expr)(*children) reconstruction used by every pass still works.
    assert expr.children() == [expr.a, expr.b]
    assert type(expr)(expr.a, expr.b) == expr


def test_pointwise_modulo_codegen_emits_struct_once() -> None:
    """The operator struct is collected as a prelude and instantiated (unsuffixed)
    in the kernel, and is deduplicated even when the op appears twice."""
    with with_types(Type.Arith, ElementType.I32):
        # Same op used twice in one expression.
        m = PointwiseModulo(Var("x"), Var("y"))
        expr = PointwiseModulo(m, Var("z"))
    expr = _annotate(expr)

    code = CodeGen().generate(expr, "k0")
    assert len(code.custom_op_preludes) == 1, "prelude should be collected exactly once"
    assert "struct PointwiseModulo" in code.custom_op_preludes[0]
    # Instantiated by the benchmark-owned name with no semiring suffix.
    assert "PointwiseModulo<int32_vec" in code.text
    assert "PointwiseModuloArith" not in code.text


def test_pointwise_modulo_correctness() -> None:
    """PointwiseModulo(x, y) computes x % y elementwise (positive operands, so C
    truncated modulo agrees with numpy)."""
    with with_types(Type.Arith, ElementType.I32):
        expr = PointwiseModulo(Var("x"), Var("y"))
    expr = _annotate(expr)

    runner = generate_kernel_executable(CodeGen(), expr)

    n = 4096
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1_000_000, size=n).astype(np.int32)
    y = rng.integers(1, 97, size=n).astype(np.int32)  # nonzero, positive divisor

    output, _, _ = runner.run({"x": x, "y": y})

    assert output.dtype == np.int32
    assert np.array_equal(output, (x % y).astype(np.int32))


def test_pointwise_modulo_over_recurrence_schedules_agree() -> None:
    """The op survives the full optimizing schedule: wrapping a dilatable integer
    prefix-sum recurrence yields several schedules, all computing cumsum(g) % 100."""
    with with_types(Type.Arith, ElementType.I32):
        expr = PointwiseModulo(Recurse(TIKernel([0, 1]), Var("g")), Num(100))

    schedules = get_candidate_schedules(expr, get_native_bits())
    assert len(schedules) > 1, "dilation of the inner recurrence should give >1 schedule"

    n = 2048
    rng = np.random.default_rng(1)
    g = rng.integers(0, 50, size=n).astype(np.int32)  # nonneg keeps cumsum nonneg
    expected = (np.cumsum(g, dtype=np.int64).astype(np.int32) % 100).astype(np.int32)

    for i, sched in enumerate(schedules):
        # row_size=n: run the whole stream as one row so the prefix-sum recurrence
        # is continuous (it resets its state at row boundaries).
        runner = generate_kernel_executable(CodeGen(), sched, row_size=n)
        output, _, _ = runner.run({"g": g})
        assert np.array_equal(output, expected), f"schedule {i} disagreed"


def test_pointwise_modulo_benchmark_path() -> None:
    """Full search/compile/cross-schedule-correctness path (exercises the header
    assembly in instantiate_kernels). Constant divisor avoids the benchmark's
    random-fill producing a zero divisor."""
    with with_types(Type.Arith, ElementType.I32):
        expr = PointwiseModulo(Recurse(TIKernel([0, 1]), Var("g")), Num(1000))

    schedules = get_candidate_schedules(expr, get_native_bits())

    result = generate_and_run_benchmark(
        codegen=CodeGen(),
        exprs=schedules[:3],
        include_correctness_check=True,
        input_size=1 << 14,
        warmup_iterations=2,
        benchmark_iterations=5,
    )
    assert result["return_code"] == 0, f"benchmark failed: {result.get('error')}\n{result.get('output')}"


def test_pretty_prints_custom_op() -> None:
    with with_types(Type.Arith, ElementType.I32):
        expr = PointwiseModulo(Var("x"), Var("y"))
    assert pp(expr) == "PointwiseModulo(x, y)"
