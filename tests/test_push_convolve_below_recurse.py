import numpy as np

from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.expr import Convolve, Recurse, TIKernel, Type, Var, with_types
from vecrec.transform import (
    AnnotateLanes,
    PushConvolveBelowRecurse,
    PushDownConvertLanes,
    SaturatePreorder,
    ConstantFold,
    Seq,
)
from vecrec.util import ElementType
from vecrec.codegen import generate_kernel_executable


def test_push_convolve_below_recurse_structure() -> None:
    """Convolve(h, Recurse(a, g)) should rewrite to Recurse(a, Convolve(h, g))."""
    with with_types(Type.Arith, ElementType.Float):
        a = TIKernel([0, 0.5])
        h = TIKernel([1.0, 0.25])
        g = Var("x")
        expr = Convolve(h, Recurse(a, g))

    results = PushConvolveBelowRecurse().apply_signal(expr)
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, Recurse)
    assert result.a == a
    assert isinstance(result.g, Convolve)
    assert result.g.a == h
    assert result.g.b == g


def test_push_convolve_below_recurse_no_match() -> None:
    """The transformation should not fire when there is no Convolve wrapping a Recurse."""
    with with_types(Type.Arith, ElementType.Float):
        a = TIKernel([0, 0.5])
        g = Var("x")
        expr = Recurse(a, g)

    results = PushConvolveBelowRecurse().apply_signal(expr)
    assert results == []


def test_push_convolve_below_recurse_correctness() -> None:
    """Verify that Convolve(h, Recurse(a, g)) and Recurse(a, Convolve(h, g))
    produce the same numerical output."""
    with with_types(Type.Arith, ElementType.Float):
        a = TIKernel([0, 0.5])
        h = TIKernel([1.0, 0.25])
        g = Var("x")
        original = Convolve(h, Recurse(a, g))

    rewritten = PushConvolveBelowRecurse().apply_signal(original)
    assert len(rewritten) == 1

    schedule = Seq(
        SaturatePreorder(ConstantFold),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )

    original_exprs = schedule.apply_signal(original)
    rewritten_exprs = schedule.apply_signal(rewritten[0])
    assert len(original_exprs) > 0
    assert len(rewritten_exprs) > 0

    codegen = CodeGen()
    result = generate_and_run_benchmark(
        codegen=codegen,
        exprs=[original_exprs[0], rewritten_exprs[0]],
        include_correctness_check=True,
        input_size=1 << 12,
        warmup_iterations=2,
        benchmark_iterations=5,
    )
    assert result["return_code"] == 0, f"Benchmark failed: {result.get('error')}"
