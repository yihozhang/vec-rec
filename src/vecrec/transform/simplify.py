from __future__ import annotations
from typing import List, Sequence, overload
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr
from vecrec.expr.base import SignalExpr2D
from vecrec.expr.signal_ops import Convolve2D, Ith, Repeater, Recurse2D
from vecrec.expr.signal import Var2D
from vecrec.expr.kernel import TIKernel2D, TVKernel2D
from . import Transform, ArithTransform
from .combinators import Any

__all__ = [
    "ConstantFoldAdd",
    "ConstantFoldSub",
    "ConstantFoldConvolve",
    "ConstantFoldNegate",
    "InlinePointwiseOps",
    "ConstantFold",
    "PushDownAndFuseConvolve",
    "EliminateDegenerate2Dkernels",
]


# Constant folding
class ConstantFoldAdd(Transform):
    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        match expr:
            case KAdd(a, b) if isinstance(a, KernelConstant) and isinstance(b, KernelConstant):
                return [a + b]
            case _:
                return []

    def apply_signal(self, expr):
        match expr:
            case SAdd(Num(a), b) | SAdd(b, Num(a)) if expr.ty.is_zero(a):
                return [b]
            case SSub(b, Num(a)) if expr.ty.is_zero(a):
                return [b]
            case _:
                return []


class ConstantFoldSub(Transform):
    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        match expr:
            case KSub(a, b) if isinstance(a, KernelConstant) and isinstance(b, KernelConstant):
                return [a - b]
            case _:
                return []


class ConstantFoldConvolve(Transform):
    """Constant fold convolution of kernels."""

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        match expr:
            case KConvolve(a, b) if isinstance(a, KernelConstant) and isinstance(b, KernelConstant):
                return [a * b]
            case _:
                return []

    def apply_signal(self, expr):
        match expr:
            case Convolve(TIKernel(a), b) if all(expr.ty.is_zero(v) for v in a):
                return [Num(0, expr.ty, expr.element_type)]
            case Convolve(TIKernel(a), b) if expr.ty.is_one(a[0]) and all(expr.ty.is_zero(v) for v in a[1:]):
                return [b]
            case Convolve(TIKernel(a), b) if all(expr.ty.is_zero(v) for v in a):
                return [Num(0, expr.ty, expr.element_type)]
            case _:
                return []


@ArithTransform
class ConstantFoldNegate(Transform):
    """Constant fold negation of time invariant kernels."""

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        match expr:
            case KNeg(a) if isinstance(a, KernelConstant):
                return [-a]
            case _:
                return []


class InlinePointwiseOps(Transform):
    """Inline and transform Convolve(kernel, signal) `PointwiseMul/PointwiseDiv` constants into `Convolve(KMul/KDiv(kernel, constant), signal)`"""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case PointwiseMul(Convolve(k, s), Num(c)):
                assert isinstance(k, TIKernel | TVKernel)
                return [Convolve(k * c, s)]
            case PointwiseDiv(Convolve(k, s), Num(c)):
                assert isinstance(k, TIKernel | TVKernel)
                return [Convolve(k / c, s)]
            case PointwiseMul(Recurse(k, s), Num(c)):
                assert isinstance(k, TIKernel | TVKernel)
                return [Recurse(k * c, s)]
            case PointwiseDiv(Recurse(k, s), Num(c)):
                assert isinstance(k, TIKernel | TVKernel)
                return [Recurse(k / c, s)]
            case PointwiseMul(Convolve2D(k, s), Num(c)):
                assert isinstance(k, TIKernel2D | TVKernel2D)
                return [Convolve2D(k * c, s)]
            case PointwiseDiv(Convolve2D(k, s), Num(c)):
                assert isinstance(k, TIKernel2D | TVKernel2D)
                return [Convolve2D(k / c, s)]
            # case PointwiseMul(Recurse2D(k, s), Num(c)):
            #     assert isinstance(k, TIKernel2D | TVKernel2D)
            #     return [Recurse2D(k * c, s)]
            # case PointwiseDiv(Recurse2D(k, s), Num(c)):
            #     assert isinstance(k, TIKernel2D | TVKernel2D)
            #     return [Recurse2D(k / c, s)]
            case _:
                return []


ConstantFold = Any(
    ConstantFoldAdd(),
    ConstantFoldSub(),
    ConstantFoldConvolve(),
    ConstantFoldNegate(),
)


class PushDownAndFuseConvolve(Transform):
    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Convolve2D(k1, f):

                @overload
                def push_down(expr: SignalExpr) -> List[SignalExpr]: ...
                @overload
                def push_down(expr: SignalExpr2D) -> List[SignalExpr2D]: ...

                def push_down(expr):
                    match expr:
                        case Recurse(a, g):
                            return [Recurse(a, g_) for g_ in push_down(g)]
                        case Convolve(h, g):
                            return [Convolve(h, g_) for g_ in push_down(g)]
                        case SAdd(a, b) | SSub(a, b) | PointwiseMul(a, b) | PointwiseDiv(a, b):
                            return [expr.__class__(a_, b_) for a_ in push_down(a) for b_ in push_down(b)]
                        case Recurse2D(b, g):
                            return [Recurse2D(b, g_) for g_ in push_down(g)]
                        case Convolve2D(k2, Var2D(_) as g):
                            assert isinstance(k1, TIKernel2D | TVKernel2D)
                            assert isinstance(k2, TIKernel2D | TVKernel2D)
                            return [Convolve2D(k1 * k2, g)]
                        case Repeater(f, n_rows, prev_rows_var):
                            return [Repeater.make(f_, n_rows, prev_rows_var) for f_ in push_down(f)]
                        case Ith(f, i):
                            return [Ith(f_, i) for f_ in push_down(f)]
                        case _:  # RVar2D, Var2D, Repeater, Ith, ...
                            return []

                return [Ith(f_, 0) for f_ in push_down(f)]
            case _:
                return []


def _contains_node(expr: RecLang, target: RecLang) -> bool:
    """Return True if `target` appears anywhere in the expression tree rooted at `expr`."""
    if expr == target:
        return True
    return any(_contains_node(child, target) for child in expr.children())


class EliminateDegenerate2Dkernels(Transform):
    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Ith(Repeater(f, _n_rows, prev_rows_var), 0) if not _contains_node(f, prev_rows_var):
                # If the Repeater doesn't actually depend on previous rows, we can eliminate the Repeater and just use the function directly.
                return [f]
            case _:
                return []
