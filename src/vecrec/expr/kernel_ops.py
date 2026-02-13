from __future__ import annotations
from abc import abstractmethod
import copy
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import numbers
from typing import Callable, List, Optional, Dict, Sequence, Tuple, overload
import numpy as np

from vecrec.expr.kernel import TIKernel
from vecrec.util import allclose, ElementType
from .base import *

@dataclass
class KernelExprBinOp(KernelExpr):
    a: KernelExpr
    b: KernelExpr
    ty: Type
    element_type: ElementType
    __match_args__ = ("a", "b")

    def __init__(self, a: KernelExpr, b: KernelExpr) -> None:
        super().__init__()
        assert a.ty == b.ty
        assert a.element_type == b.element_type, \
            f"ElementType mismatch in KernelExprBinOp: {a.element_type} vs {b.element_type}"
        self.a = a
        self.b = b
        self.ty = a.ty
        self.element_type = a.element_type

    def time_delay(self, max_delay: int) -> tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        a_delay, _ = self.a.time_delay(max_delay)
        b_delay, _ = self.b.time_delay(max_delay)
        delay = min(a_delay, b_delay)
        return delay, KConvolve(TIKernel.z(self.ty, self.element_type, -delay), self).with_lanes(
            self.lanes
        )

    @classmethod
    def of(cls, exprs: List[KernelExpr]) -> KernelExpr:
        """Combine a list of KernelExpr into a single KernelExpr using the binary operation."""
        return reduce(lambda a, b: cls(a, b), exprs)


class KAdd(KernelExprBinOp):
    pass


class KSub(KernelExprBinOp):
    pass


class KConvolve(KernelExprBinOp):
    def time_delay(self, max_delay: int) -> tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        a_delay, _ = self.a.time_delay(max_delay)
        b_delay, _ = self.b.time_delay(max_delay)
        delay = min(a_delay + b_delay, max_delay)
        return delay, KConvolve(TIKernel.z(self.ty, self.element_type, -delay), self).with_lanes(
            self.lanes
        )


class KNeg(KernelExpr):
    a: KernelExpr
    ty: Type
    element_type: ElementType
    __match_args__ = ("a",)

    def __init__(self, a: KernelExpr) -> None:
        super().__init__()
        self.a = a
        self.ty = a.ty
        self.element_type = a.element_type

    def time_delay(self, max_delay: int) -> tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        delay, e = self.a.time_delay(max_delay)
        return delay, KNeg(e).with_lanes(self.lanes)


@dataclass
class KConvertLanes(KernelExpr):
    a: KernelExpr
    __match_args__ = ("a",)

    def __init__(self, a: KernelExpr) -> None:
        super().__init__()
        self.a = a
        self.ty = a.ty
        self.element_type = a.element_type
        self.lanes = None

    def time_delay(self, max_delay: int) -> tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        delay, e = self.a.time_delay(max_delay)
        assert self.lanes is not None
        return delay, KConvertLanes(e).with_lanes(self.lanes)

    def __repr__(self) -> str:
        return f"ConvertLanes({self.lanes}, {self.a})"
