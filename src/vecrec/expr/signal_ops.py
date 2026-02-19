from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import Callable, List

from vecrec.expr.signal import Var2D
from vecrec.util import ElementType


from .base import *

@dataclass
class SignalExprBinOp(SignalExpr):
    a: SignalExpr
    b: SignalExpr
    __match_args__ = ("a", "b")

    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__()
        assert a.ty == b.ty
        assert (
            a.element_type == b.element_type
        ), f"ElementType mismatch in SignalExprBinOp: {a.element_type} vs {b.element_type}"
        self.a = a
        self.b = b
        self.ty = a.ty
        self.element_type = a.element_type

    @classmethod
    def of(cls, exprs: List[SignalExpr]) -> SignalExpr:
        return reduce(lambda a, b: cls(a, b), exprs)


class SAdd(SignalExprBinOp):
    pass


class SSub(SignalExprBinOp):
    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__(a, b)
        assert self.ty == Type.Arith, "Subtraction is only defined for arithmetic type"


class PointwiseMul(SignalExprBinOp):
    pass


class PointwiseDiv(SignalExprBinOp):
    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__(a, b)
        assert self.ty == Type.Arith, "Division is only defined for arithmetic type"


@dataclass
class SNeg(SignalExpr):
    a: SignalExpr
    __match_args__ = ("a",)

    def __init__(self, a: SignalExpr) -> None:
        super().__init__()
        assert a.ty == Type.Arith, "Negation is only defined for arithmetic type"
        self.a = a
        self.ty = a.ty
        self.element_type = a.element_type


@dataclass
class Convolve(SignalExpr):
    a: KernelExpr
    b: SignalExpr
    __match_args__ = ("a", "b")

    def __init__(self, a: KernelExpr, b: SignalExpr) -> None:
        super().__init__()
        assert a.ty == b.ty
        assert (
            a.element_type == b.element_type
        ), f"ElementType mismatch in Convolve: {a.element_type} vs {b.element_type}"
        self.ty = a.ty
        self.element_type = a.element_type
        self.a = a
        self.b = b


@dataclass
class Recurse(SignalExpr):
    a: KernelExpr
    g: SignalExpr
    __match_args__ = ("a", "g")

    def __init__(self, a: KernelExpr, g: SignalExpr) -> None:
        super().__init__()
        assert a.ty == g.ty
        assert (
            a.element_type == g.element_type
        ), f"ElementType mismatch in Recurse: {a.element_type} vs {g.element_type}"
        self.ty = a.ty
        self.element_type = a.element_type
        self.a = a
        self.g = g


# Convert Lanes
@dataclass
class ConvertLanes(SignalExpr):
    a: SignalExpr
    __match_args__ = ("a",)

    def __init__(self, a: SignalExpr) -> None:
        super().__init__()
        self.a = a
        self.ty = a.ty
        self.element_type = a.element_type
        self.lanes = None

    def __repr__(self) -> str:
        return f"ConvertLanes({self.lanes}, {self.a})"

@dataclass
class Convolve2D(SignalExpr):
    """2D convolution: convolve a 2D kernel with a 2D signal (from Repeater). Produces a 1D signal"""

    a: KernelExpr2D
    b: SignalExpr2D
    __match_args__ = ("a", "b")

    def __init__(self, a: KernelExpr2D, b: SignalExpr2D) -> None:
        super().__init__()
        assert a.ty == b.ty
        assert (
            a.element_type == b.element_type
        ), f"ElementType mismatch in Convolve2D: {a.element_type} vs {b.element_type}"
        self.a = a
        self.b = b
        self.ty = a.ty
        self.element_type = a.element_type

@dataclass
class Recurse2D(SignalExpr2D):
    """2D version of Recurse. The callable receives a Var2D representing the output of the previous rows."""

    a: KernelExpr2D
    g: SignalExpr
    __match_args__ = ("a", "g")

    def __init__(self, a: KernelExpr2D, g: SignalExpr) -> None:
        super().__init__()
        assert a.ty == g.ty
        assert (
            a.element_type == g.element_type
        ), f"ElementType mismatch in Recurse: {a.element_type} vs {g.element_type}"
        self.ty = a.ty
        self.element_type = a.element_type
        self.a = a
        self.g = g

@dataclass
class Repeater(SignalExpr2D):
    """
    Turns a 1D signal into a 2D signal by caching the last `n_rows` rows.
    The minimal number of rows to cache is 2, which stores the previous row 
    while computing and storing the current row.

    The callable receives a Var2D representing the previous rows stored in
    the Repeater, with the current horizontal position aligned with the
    stream for the current row. Use Convolve2D with a 2D kernel to
    access data from multiple previous rows.
    
    Streams as a 2D signal: run() returns n vectors at a time (current row
    plus the n-1 previously cached rows).
    """

    a: SignalExpr
    n_rows: int
    prev_rows_var: Var2D

    var_count: int = 1

    def __init__(
        self,
        func: Callable[[Var2D], SignalExpr],
        n_rows: int,
        ty: Type,
        element_type: ElementType,
    ) -> None:
        super().__init__()
        self.n_rows = n_rows
        self.prev_rows_var = Var2D(f"$f{Repeater.var_count}", ty, element_type)
        Repeater.var_count += 1
        self.a = func(self.prev_rows_var)
        self.ty = self.a.ty
        self.element_type = self.a.element_type
    
    @staticmethod
    def make(a: SignalExpr, n_rows: int, prev_rows_var: Var2D) -> Repeater:
        repeater = Repeater.__new__(Repeater)
        super(Repeater, repeater).__init__()
        repeater.a = a
        repeater.n_rows = n_rows
        repeater.prev_rows_var = prev_rows_var
        repeater.ty = a.ty
        repeater.element_type = a.element_type
        return repeater

    def children(self) -> List[KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D]:
        # Only expose the inner expression, not prev_rows_var
        # (it's an internal placeholder, not an external input)
        return [self.a]

@dataclass
class Ith(SignalExpr):
    a: SignalExpr2D
    i: int
    __match_args__ = ("a", "i")

    def __init__(self, a: SignalExpr2D, i: int) -> None:
        super().__init__()
        self.a = a
        self.i = i
        self.ty = Type.Arith
        self.element_type = a.element_type
