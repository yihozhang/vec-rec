"""Regression tests for pretty-printer let-binding behavior."""

from vecrec.expr.base import Type, with_types
from vecrec.expr.pretty import pp
from vecrec.expr.signal import Var
from vecrec.expr.signal_ops import PointwiseMul, SAdd
from vecrec.util import ElementType


def test_pp_inlines_single_use_binding() -> None:
    """A child used once in rendered output should be inlined."""
    with with_types(Type.Arith, ElementType.Float):
        shared_child = PointwiseMul(Var("a"), Var("x"))
        shared_parent = SAdd(shared_child, Var("b"))
        expr = SAdd(shared_parent, shared_parent)

    rendered = pp(expr)

    assert "let t0 = a · x + b" in rendered
    assert "in t0 + t0" in rendered
    assert "let t0 = a · x\n" not in rendered


def test_pp_keeps_binding_for_multi_use_subexpr() -> None:
    """A subexpression used more than once should still be let-bound."""
    with with_types(Type.Arith, ElementType.Float):
        expr = SAdd(
            PointwiseMul(Var("a"), Var("x")),
            PointwiseMul(Var("a"), Var("x")),
        )

    rendered = pp(expr)

    assert "let t0 = a · x" in rendered
    assert "in t0 + t0" in rendered
