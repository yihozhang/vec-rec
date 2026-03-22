from __future__ import annotations
from typing import List, Sequence, overload
import typing
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr, Type
from vecrec.expr.base import KernelExpr2D, SignalExpr2D, KernelMatrixExpr, SignalVectorExpr
from vecrec.expr import pp


class Transform:
    @overload
    def apply_generic(self, expr: KernelExpr) -> Sequence[KernelExpr]: ...
    @overload
    def apply_generic(self, expr: SignalExpr) -> Sequence[SignalExpr]: ...
    @overload
    def apply_generic(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]: ...
    @overload
    def apply_generic(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]: ...
    @overload
    def apply_generic(self, expr: KernelMatrixExpr) -> Sequence[KernelMatrixExpr]: ...
    @overload
    def apply_generic(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]: ...

    def apply_generic(self, expr):
        if isinstance(expr, KernelExpr):
            return self.apply_kernel(expr)
        elif isinstance(expr, SignalExpr):
            return self.apply_signal(expr)
        elif isinstance(expr, KernelExpr2D):
            return self.apply_kernel2d(expr)
        elif isinstance(expr, SignalExpr2D):
            return self.apply_signal2d(expr)
        elif isinstance(expr, (KernelMatrixExpr, SignalVectorExpr)):
            # Transforms do not yet handle vector/matrix expressions; pass through unchanged.
            return [expr]
        else:
            raise TypeError(f"Unknown expression type: {type(expr)}")

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        return []

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        return []

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        return []

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        return []

def ArithTransform(cls):
    cls.apply_kernel_impl = cls.apply_kernel
    cls.apply_signal_impl = cls.apply_signal
    cls.apply_signal2d_impl = cls.apply_signal2d
    cls.apply_kernel2d_impl = cls.apply_kernel2d

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:  # type: ignore
        if expr.ty == Type.Arith:
            return self.apply_kernel_impl(expr)
        else:
            return []

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:  # type: ignore
        if expr.ty == Type.Arith:
            return self.apply_signal_impl(expr)
        else:
            return []

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:  # type: ignore
        if expr.ty == Type.Arith:
            return self.apply_signal2d_impl(expr)
        else:
            return []

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:  # type: ignore
        if expr.ty == Type.Arith:
            return self.apply_kernel2d_impl(expr)
        else:
            return []

    cls.apply_kernel = apply_kernel
    cls.apply_signal = apply_signal
    cls.apply_signal2d = apply_signal2d
    cls.apply_kernel2d = apply_kernel2d
    return cls


def TropTransform(cls):
    cls.apply_kernel_impl = cls.apply_kernel
    cls.apply_signal_impl = cls.apply_signal
    cls.apply_signal2d_impl = cls.apply_signal2d
    cls.apply_kernel2d_impl = cls.apply_kernel2d

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:  # type: ignore
        if expr.ty in (Type.TropMax, Type.TropMin):
            return self.apply_kernel_impl(expr)
        else:
            return []

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:  # type: ignore
        if expr.ty in (Type.TropMax, Type.TropMin):
            return self.apply_signal_impl(expr)
        else:
            return []

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:  # type: ignore
        if expr.ty in (Type.TropMax, Type.TropMin):
            return self.apply_signal2d_impl(expr)
        else:
            return []

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:  # type: ignore
        if expr.ty in (Type.TropMax, Type.TropMin):
            return self.apply_kernel2d_impl(expr)
        else:
            return []

    cls.apply_kernel = apply_kernel
    cls.apply_signal = apply_signal
    cls.apply_signal2d = apply_signal2d
    cls.apply_kernel2d = apply_kernel2d
    return cls


class DebugPrint(Transform):
    def __init__(self) -> None:
        self.apply_kernel = self.apply_signal = self.apply_signal2d = self.apply_kernel2d = self.apply_generic  # type: ignore

    def apply_generic(self, expr):
        print(pp(expr))
        return [expr]


def _dedup_exprs(results: list) -> list:
    if len(results) < 10:
        unique: list = []
        for expr in results:
            if not any(expr == u for u in unique):
                unique.append(expr)
    else:
        unique = list(dict.fromkeys(results))
    return unique


from .combinators import *  # noqa: E402
from .simplify import *  # noqa: E402
from .core import *  # noqa: E402
from .lower import *  # noqa: E402

__all__ = [
    "Transform",
    "DebugPrint",
    "Noop",
    "ConstantFoldAdd",
    "ConstantFoldSub",
    "ConstantFoldConvolve",
    "ConstantFoldNegate",
    "ConstantFold",
    "FuseRecurse",
    "Dilate",
    "DilateTrop",
    "Delay",
    "ComposeRecurse",
    "Factorize",
    "Factorize2D",
    "DilateTVWithSingleOddOrder",
    "InlinePointwiseOps",
    "PushDownAndFuseConvolve",
    "EliminateDegenerate2Dkernels",
    "Seq",
    "Repeat",
    "RepeatUpTo",
    "Optional",
    "Try",
    "Any",
    "Preorder",
    "SaturatePreorder",
    "SaturatePreorderUpTo",
    "Postorder",
    "InferNRows",
    "AnnotateLanes",
    "PushDownConvertLanes",
    "Eliminate2DKernels",
]
