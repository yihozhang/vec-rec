from __future__ import annotations
from abc import abstractmethod
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import numbers
import threading
from typing import Generator, List, Optional, Tuple, overload

import numpy as np

from vecrec.util import ElementType


class Type(Enum):
    Arith = 1
    TropMax = 2
    TropMin = 3

    def is_zero(self, value: float) -> bool:
        match self:
            case Type.Arith:
                return bool(np.isclose(value, 0.0))
            case Type.TropMax:
                return bool(np.isclose(value, -np.inf))
            case Type.TropMin:
                return bool(np.isclose(value, np.inf))
        assert False, "unreachable"

    def is_one(self, value: float) -> bool:
        match self:
            case Type.Arith:
                return bool(np.isclose(value, 1.0))
            case Type.TropMax:
                return bool(np.isclose(value, 0.0))
            case Type.TropMin:
                return bool(np.isclose(value, 0.0))
        assert False, "unreachable"

    def zero(self) -> float:
        match self:
            case Type.Arith:
                return 0.0
            case Type.TropMax:
                return -np.inf
            case Type.TropMin:
                return np.inf
        assert False, "unreachable"

    def one(self) -> float:
        match self:
            case Type.Arith:
                return 1.0
            case Type.TropMax:
                return 0.0
            case Type.TropMin:
                return 0.0
        assert False, "unreachable"

    def convolve(self, a: List[float], b: List[float]) -> List[float]:
        match self:
            case Type.Arith:
                return np.convolve(a, b).tolist()
            case Type.TropMax:
                max_len = len(a) + len(b) - 1
                result = [-np.inf] * max_len
                for i in range(len(a)):
                    for j in range(len(b)):
                        result[i + j] = max(result[i + j], a[i] + b[j])
                return result
            case Type.TropMin:
                max_len = len(a) + len(b) - 1
                result = [np.inf] * max_len
                for i in range(len(a)):
                    for j in range(len(b)):
                        result[i + j] = min(result[i + j], a[i] + b[j])
                return result
        assert False, "unreachable"

    @overload
    def add(self, a: float, b: float) -> float: ...
    @overload
    def add(self, a: list[float], b: list[float] | float) -> list[float]: ...
    @overload
    def add(self, a: float, b: list[float]) -> list[float]: ...

    def add(self, a, b):
        f = {
            Type.Arith: lambda x, y: x + y,
            Type.TropMax: lambda x, y: max(x, y),
            Type.TropMin: lambda x, y: min(x, y),
        }[self]
        if isinstance(a, list) and isinstance(b, list):
            return [f(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list) and isinstance(b, numbers.Number):
            return [f(x, b) for x in a]
        elif isinstance(a, numbers.Number) and isinstance(b, list):
            return [f(a, y) for y in b]
        elif isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            return f(a, b)
        assert False, "unreachable"

    @overload
    def sub(self, a: List[float], b: List[float]) -> List[float]: ...
    @overload
    def sub(self, a: List[float], b: float) -> List[float]: ...
    @overload
    def sub(self, a: float, b: List[float]) -> List[float]: ...
    def sub(self, a, b):
        assert self == Type.Arith, "Subtraction is only defined for arithmetic type"
        if isinstance(a, list) and isinstance(b, list):
            return [x - y for x, y in zip(a, b)]
        elif isinstance(a, list) and isinstance(b, numbers.Number):
            return [x - b for x in a]
        elif isinstance(a, numbers.Number) and isinstance(b, list):
            return [a - y for y in b]
        elif isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            return a - b  # type: ignore
        assert False, "unreachable"

    @overload
    def mult(self, a: float, b: float) -> float: ...
    @overload
    def mult(self, a: list[float], b: list[float] | float) -> list[float]: ...
    @overload
    def mult(self, a: float, b: list[float]) -> list[float]: ...

    def mult(self, a, b):
        f = {
            Type.Arith: lambda x, y: x * y,
            Type.TropMax: lambda x, y: x + y,
            Type.TropMin: lambda x, y: x + y,
        }[self]
        if isinstance(a, list) and isinstance(b, list):
            return [f(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list) and isinstance(b, numbers.Number):
            return [f(x, b) for x in a]
        elif isinstance(a, numbers.Number) and isinstance(b, list):
            return [f(a, y) for y in b]
        elif isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            return f(a, b)
        assert False, "unreachable"

    @overload
    def div(self, a: float, b: float) -> float: ...
    @overload
    def div(self, a: list[float], b: list[float] | float) -> list[float]: ...
    @overload
    def div(self, a: float, b: list[float]) -> list[float]: ...
    def div(self, a, b):
        assert self == Type.Arith, "Division is only defined for arithmetic type"
        if isinstance(a, list) and isinstance(b, list):
            return [x / y for x, y in zip(a, b)]
        elif isinstance(a, list) and isinstance(b, numbers.Number):
            return [x / b for x in a]
        elif isinstance(a, numbers.Number) and isinstance(b, list):
            return [a / y for y in b]
        elif isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            return a / b  # type: ignore
        assert False, "unreachable"


@dataclass(kw_only=True, match_args=False)
class RecLang:
    lanes: Optional[int] = None

    ty: Type
    element_type: ElementType

    def __init__(self, lanes: Optional[int] = None) -> None:
        self.lanes = lanes

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        
        if id(self) == id(other):
            return True

        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(self.__dict__.items()))))

    def children(
        self,
    ) -> List[KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D | KernelMatrixExpr | SignalVectorExpr | SignalExpr2DVec]:
        """Return the children of the expression."""
        return [
            v
            for v in self.__dict__.values()
            if isinstance(v, (KernelExpr, SignalExpr, KernelExpr2D, SignalExpr2D, KernelMatrixExpr, SignalVectorExpr, SignalExpr2DVec))
        ]

    def is_leaf(self) -> bool:
        """Return whether the expression is a leaf node."""
        for v in self.__dict__.values():
            if isinstance(v, (KernelExpr, SignalExpr, KernelExpr2D, SignalExpr2D, KernelMatrixExpr, SignalVectorExpr, SignalExpr2DVec)):
                return False
        return True

    def with_lanes(self, lanes: Optional[int]) -> RecLang:
        """Return a copy of the expression with the given number of lanes."""
        assert self.lanes is None, "Lanes already set"
        new_expr = copy.copy(self)
        new_expr.lanes = lanes
        return new_expr


@dataclass(match_args=False)
class SignalExpr(RecLang):
    ty: Type
    element_type: ElementType

    def with_lanes(self, lanes: Optional[int]) -> SignalExpr:
        """Return a copy of the expression with the given number of lanes."""
        return super().with_lanes(lanes)  # type: ignore

    def cache_rows(self) -> SignalExpr2D:
        from vecrec.expr.signal_ops import (
            Repeater,
        )  # local import to avoid circular dependency

        return Repeater(lambda _: self, self.ty, self.element_type).with_lanes(self.lanes)


@dataclass(match_args=False)
class SignalExpr2D(RecLang):
    ty: Type
    element_type: ElementType

    def at(self, r: int) -> SignalExpr:
        from vecrec.expr.signal_ops import Ith

        return Ith(self, r)

    def with_lanes(self, lanes: Optional[int]) -> SignalExpr2D:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class KernelExpr(RecLang):
    ty: Type
    element_type: ElementType

    @abstractmethod
    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        """Return the time delay of the kernel and the kernel without the delay."""
        ...

    @abstractmethod
    def to_sparse_repr(self) -> Tuple[int, List[int], List[float] | List[SignalExpr]]: ...

    def with_lanes(self, lanes: Optional[int]) -> KernelExpr:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class KernelExpr2D(RecLang):
    ty: Type
    element_type: ElementType

    def with_lanes(self, lanes: Optional[int]) -> KernelExpr2D:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class KernelMatrixExpr(RecLang):
    """Base class for an n×n matrix of kernel expressions."""

    ty: Type
    element_type: ElementType
    n: int

    def __init__(self, ty: Type, element_type: ElementType, n: int, lanes: Optional[int] = None) -> None:
        super().__init__(lanes)
        self.ty = ty
        self.element_type = element_type
        self.n = n

    def with_lanes(self, lanes: Optional[int]) -> KernelMatrixExpr:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class SignalVectorExpr(RecLang):
    """Base class for an n-vector of signal streams (each run() produces n vec_type values)."""

    ty: Type
    element_type: ElementType
    n: int

    def __init__(self, ty: Type, element_type: ElementType, n: int, lanes: Optional[int] = None) -> None:
        super().__init__(lanes)
        self.ty = ty
        self.element_type = element_type
        self.n = n

    def at(self, i: int) -> SignalExpr:
        """Extract component i as a 1D SignalExpr."""
        from vecrec.expr.signal_vec import VectorAt  # local import to avoid circular dependency

        return VectorAt(self, i)

    def with_lanes(self, lanes: Optional[int]) -> SignalVectorExpr:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class SignalExpr2DVec(RecLang):
    """Base class for a vector of n 2D signal streams (each component is a SignalExpr2D)."""

    ty: Type
    element_type: ElementType
    n: int

    def __init__(self, ty: Type, element_type: ElementType, n: int, lanes: Optional[int] = None) -> None:
        super().__init__(lanes)
        self.ty = ty
        self.element_type = element_type
        self.n = n

    def at(self, i: int) -> SignalExpr2D:
        """Extract component i as a SignalExpr2D."""
        from vecrec.expr.signal_ops import Signal2DAt  # local import to avoid circular dependency

        return Signal2DAt(self, i)

    def with_lanes(self, lanes: Optional[int]) -> SignalExpr2DVec:
        return super().with_lanes(lanes)  # type: ignore


# ---------------------------------------------------------------------------
# Default-type context manager
# ---------------------------------------------------------------------------

_type_context: threading.local = threading.local()


def _resolve_types(ty: Optional[Type], element_type: Optional[ElementType]) -> Tuple[Type, ElementType]:
    """Resolve ty/element_type, falling back to the current with_types() context."""
    resolved_ty = ty if ty is not None else getattr(_type_context, "ty", None)
    resolved_et = element_type if element_type is not None else getattr(_type_context, "element_type", None)
    if resolved_ty is None:
        raise ValueError("Type must be specified explicitly or via the with_types() context manager")
    if resolved_et is None:
        raise ValueError("ElementType must be specified explicitly or via the with_types() context manager")
    return resolved_ty, resolved_et


@contextmanager
def with_types(ty: Type, element_type: ElementType) -> Generator[None, None, None]:
    """Context manager that sets default Type and ElementType for leaf AST nodes.

    Example::

        with with_types(Type.Arith, ElementType.Float):
            kernel = TIKernel([0, 1.8, -0.9])
            signal = Var("x")
            expr = Recurse(kernel, signal)
    """
    old_ty: Optional[Type] = getattr(_type_context, "ty", None)
    old_et: Optional[ElementType] = getattr(_type_context, "element_type", None)
    _type_context.ty = ty
    _type_context.element_type = element_type
    try:
        yield
    finally:
        _type_context.ty = old_ty
        _type_context.element_type = old_et
