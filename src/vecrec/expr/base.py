from __future__ import annotations
from abc import abstractmethod
import copy
from enum import Enum
import numbers
from typing import List, Optional, Tuple, overload

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
        print(a, b)
        assert False, "unreachable"


class RecLang:
    lanes: Optional[int] = None

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((type(self), tuple(sorted(self.__dict__.items()))))

    def children(self) -> List[KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D]:
        """Return the children of the expression."""
        return [
            v for v in self.__dict__.values() 
            if isinstance(v, (KernelExpr, SignalExpr, KernelExpr2D, SignalExpr2D))
        ]

    def is_leaf(self) -> bool:
        """Return whether the expression is a leaf node."""
        for v in self.__dict__.values():
            if isinstance(v, (KernelExpr, SignalExpr)):
                return False
        return True

    def with_lanes(self, lanes: Optional[int]) -> RecLang:
        """Return a copy of the expression with the given number of lanes."""
        assert self.lanes is None, "Lanes already set"
        new_expr = copy.copy(self)
        new_expr.lanes = lanes
        return new_expr


class SignalExpr(RecLang):
    ty: Type
    element_type: ElementType

    def with_lanes(self, lanes: Optional[int]) -> SignalExpr:
        """Return a copy of the expression with the given number of lanes."""
        return super().with_lanes(lanes)  # type: ignore

class SignalExpr2D(RecLang):
    ty: Type
    element_type: ElementType

    def with_lanes(self, lanes: Optional[int]) -> SignalExpr2D:
        return super().with_lanes(lanes) # type: ignore

class KernelExpr(RecLang):
    ty: Type
    element_type: ElementType

    @abstractmethod
    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        """Return the time delay of the kernel and the kernel without the delay."""
        ...

    def with_lanes(self, lanes: Optional[int]) -> KernelExpr:
        return super().with_lanes(lanes)  # type: ignore

class KernelExpr2D(RecLang):
    ty: Type
    element_type: ElementType

    def with_lanes(self, lanes: Optional[int]) -> KernelExpr2D:
        return super().with_lanes(lanes)  # type: ignore
