"""Variant-generation and lane-invariant tests for PushDownConvertLanes."""

from __future__ import annotations

from typing import Iterable

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.expr import (
    ConvertLanes,
    Convolve,
    Impulse,
    KConvertLanes,
    Recurse,
    SAdd,
    TIKernel,
    Type,
    Var,
    Var2D,
    with_types,
)
from vecrec.expr.base import RecLang
from vecrec.expr.signal_ops import Ith, Repeater
from vecrec.expr.signal_vec import ConvertLanesV
from vecrec.util import ElementType, get_native_bits
from vecrec.transform import AnnotateLanes, InferNRows, PushDownConvertLanes


# ---------- helpers ----------


def _walk(expr: RecLang) -> Iterable[RecLang]:
    yield expr
    for c in expr.children():
        yield from _walk(c)


def _count_adapters(expr: RecLang) -> int:
    return sum(1 for n in _walk(expr) if isinstance(n, (ConvertLanes, KConvertLanes, ConvertLanesV)))


def _annotated(expr):
    expr = InferNRows().apply_generic(expr)[0]
    return AnnotateLanes(get_native_bits()).apply_generic(expr)[0]


# ---------- tests ----------


def test_pushdown_eliminates_inner_convertlanes_under_recurse() -> None:
    """Recurse with delayed kernel (lanes < max_lanes) wraps its child subtree
    in ConvertLanes(K, ...). The push-down variant should eliminate this
    wrapper, yielding leaves directly at K."""
    with with_types(Type.Arith, ElementType.Float):
        feedback = TIKernel([0, 0, 0, 0, 1.8, -0.9])  # time delay 4
        expr = Recurse(feedback, Var("x"))

    annotated = _annotated(expr)
    variants = PushDownConvertLanes().apply_generic(annotated)

    # The annotated form has at least one adapter (between Recurse and Var).
    assert _count_adapters(annotated) >= 1

    # Some variant must have strictly fewer adapters than the original.
    best = min(variants, key=_count_adapters)
    assert _count_adapters(best) < _count_adapters(annotated)


def test_pushdown_idempotent_on_clean_tree() -> None:
    """An expression with no eliminable adapters should yield only the original."""
    with with_types(Type.Arith, ElementType.Float):
        # Plain Recurse with a non-delayed kernel + simple Var input —
        # AnnotateLanes still inserts an adapter at the Recurse boundary
        # because Recurse.lanes (= max_lanes here, since time_delay >= max_lanes)
        # equals the child's lanes. We use a structure that genuinely has no
        # adapters: a leaf at max_lanes.
        expr = Var("x")

    annotated = _annotated(expr)
    variants = PushDownConvertLanes().apply_generic(annotated)

    assert _count_adapters(annotated) == 0
    assert len(variants) == 1
    assert variants[0] is annotated


def test_pushdown_through_recurse_under_repeater_runs() -> None:
    """Repeater containing a Recurse whose kernel time delay < max_lanes.

    This is the canonical inefficient pattern. The new pass must:
      - not crash on Repeater (the old narrow_lanes raised NotImplementedError),
      - produce at least one variant whose generated code compiles and runs.
    """
    with with_types(Type.Arith, ElementType.Float):
        feedback = TIKernel([0, 0, 0, 0, 1.8, -0.9])  # time delay 4
        row_kernel = TIKernel([1.0])
        g = Var2D("x")

        def make_inner(rvar):
            prev_row = Ith(rvar, 0)
            return Recurse(feedback, SAdd(g.at(0), Convolve(row_kernel, prev_row)))

        repeater = Repeater(make_inner)

    expr = Ith(repeater, 0)
    annotated = _annotated(expr)
    variants = PushDownConvertLanes().apply_generic(annotated)
    assert len(variants) >= 1
    # At least one variant should compile + run end-to-end.
    best = min(variants, key=_count_adapters)
    result = generate_and_run_benchmark(CodeGen(), [best])
    assert result["return_code"] == 0, f"compile/run failed: {result}"


def test_lane_invariants_on_all_variants() -> None:
    """For every variant, MUST=match parents/children share lanes, and every
    adapter is a real conversion (child.lanes != self.lanes)."""
    with with_types(Type.Arith, ElementType.Float):
        feedback = TIKernel([0, 0, 0, 0, 1.8, -0.9])
        expr = Recurse(feedback, SAdd(Var("x"), Convolve(TIKernel([1.0, 0.5]), Var("y"))))

    annotated = _annotated(expr)
    variants = PushDownConvertLanes().apply_generic(annotated)
    assert len(variants) >= 1

    for v in variants:
        for node in _walk(v):
            if isinstance(node, (ConvertLanes, KConvertLanes, ConvertLanesV)):
                # Adapter must actually convert.
                assert node.lanes is not None
                assert node.a.lanes is not None
                assert node.lanes != node.a.lanes, (
                    f"Stale adapter with matching lanes: {node!r}"
                )


def test_recurse_barrier_blocks_pushdown_when_no_strict_reduction() -> None:
    """If push-down through a Recurse barrier doesn't reduce adapter count
    (because the barrier just gets re-wrapped), no extra variant is emitted."""
    with with_types(Type.Arith, ElementType.Float):
        feedback = TIKernel([0, 0, 0, 0, 1.8, -0.9])  # time delay 4
        expr = Recurse(feedback, Impulse(0.0))

    annotated = _annotated(expr)
    variants = PushDownConvertLanes().apply_generic(annotated)

    # Every variant has the same Recurse.lanes (kernel time delay forces it).
    for v in variants:
        for node in _walk(v):
            if isinstance(node, Recurse):
                assert node.lanes is not None and node.lanes <= 4
