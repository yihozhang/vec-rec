from __future__ import annotations
from abc import abstractmethod
import copy
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import numbers
from typing import List, Optional, Dict, Sequence, Tuple, overload
import numpy as np

from vecrec.util import allclose

__all__ = [
    "RecLang",
    "SignalExpr",
    "KernelExpr",
    "KernelConstant",
    "TIKernel",
    "TVKernel",
    "KAdd",
    "KSub",
    "KNeg",
    "KConvolve",
    "Num",
    "SignalExprBinOp",
    "SAdd",
    "SSub",
    "PointwiseMul",
    "PointwiseDiv",
    "SNeg",
    "Convolve",
    "Recurse",
    "Var",
    "Type",
    "ConvertLanes",
    "KConvertLanes",
]

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
            return a - b # type: ignore
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
        print(a,b)
        assert False, "unreachable"


class RecLang:
    lanes: Optional[int] = None

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((type(self), tuple(sorted(self.__dict__.items()))))
    
    def children(self) -> List[KernelExpr | SignalExpr]:
        """Return the children of the expression."""
        return [
            v
            for v in self.__dict__.values()
            if isinstance(v, (KernelExpr, SignalExpr))
        ]
    
    def is_leaf(self) -> bool:
        """Return whether the expression is a leaf node."""
        for v in self.__dict__.values():
            if isinstance(v, (KernelExpr, SignalExpr)):
                return False
        return True

    def with_lanes(self, lanes: int) -> RecLang:
        """Return a copy of the expression with the given number of lanes."""
        assert self.lanes is None, "Lanes already set"
        new_expr = copy.copy(self)
        new_expr.lanes = lanes
        return new_expr

class SignalExpr(RecLang):
    ty: Type

    def with_lanes(self, lanes: int) -> SignalExpr:
        """Return a copy of the expression with the given number of lanes."""
        return super().with_lanes(lanes) # type: ignore


class KernelExpr(RecLang):
    ty: Type

    @abstractmethod
    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        """Return the time delay of the kernel and the kernel without the delay."""
        ...
    
    def with_lanes(self, lanes: int) -> KernelExpr:
        return super().with_lanes(lanes) # type: ignore


class TIKernel(KernelExpr):
    @staticmethod
    def z(ty: Type, n: int = -1) -> TIKernel:
        """Return the delay kernel z^n."""
        if n >= 0:
            raise ValueError("n must be negative")
        return TIKernel([0.0] * (-n) + [ty.one()], ty)

    @staticmethod
    def i(ty: Type) -> TIKernel:
        """Return the identity kernel."""
        return TIKernel([ty.one()], ty)

    ty: Type
    data: List[float]
    __match_args__ = ("data",)

    def __init__(self, data: list[float], ty: Type):
        super().__init__()
        for i in range(len(data)):
            if ty.is_zero(data[i]):
                data[i] = ty.zero()
        while len(data) > 0 and ty.is_zero(data[-1]):
            data.pop()
        self.data = data
        self.ty = ty

    def promote(self) -> TVKernel:
        return TVKernel([Num(x, self.ty) for x in self.data], self.ty)

    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        for i, v in enumerate(self.data[:max_delay]):
            if not self.ty.is_zero(v):
                return i, TIKernel(self.data[i:], self.ty)
        if max_delay < len(self.data):
            return max_delay, TIKernel(self.data[max_delay:], self.ty)
        assert False, "max_delay exceeds kernel length"

    def to_sparse_repr(self) -> Tuple[int, List[int], List[float]]:
        """Returns a tuple of (num_nonzero, indices, values) representing the sparse form of the kernel."""
        indices = []
        values = []
        for i, v in enumerate(self.data):
            if not self.ty.is_zero(v):
                indices.append(i)
                values.append(v)

        return len(indices), indices, values

    def __getitem__(self, index: int) -> float:
        if len(self.data) <= index:
            return 0.0
        return self.data[index]

    def __hash__(self):
        return hash(tuple(self.data))

    def copy(self) -> TIKernel:
        return TIKernel(self.data.copy(), self.ty)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        data = " ".join([f"{x:.8f}" for x in self.data])
        return f"TIKernel([ {data} ])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TIKernel):
            return False
        return allclose(self.data, other.data)

    def num_terms(self) -> int:
        """Return the number of non-zero terms in the signal."""
        return sum(1 for x in self.data if not self.ty.is_zero(x))

    @overload
    def __mul__(self, other: TIKernel | float) -> TIKernel: ...

    @overload
    def __mul__(self, other: TVKernel) -> TVKernel: ...

    def __mul__(self, other):
        if isinstance(other, TVKernel):
            return self.promote() * other
        elif isinstance(other, TIKernel):
            if len(self) == 0 or len(other) == 0:
                return TIKernel([], self.ty)
            return TIKernel(self.ty.convolve(self.data, other.data), self.ty)
        else:
            return TIKernel(self.ty.mult(self.data, other), self.ty)

    __rmul__ = __mul__

    @overload
    def __add__(self, other: TIKernel) -> TIKernel: ...

    @overload
    def __add__(self, other: TVKernel) -> TVKernel: ...

    def __add__(self, other):
        if isinstance(other, TVKernel):
            return self.promote() + other
        assert self.ty == other.ty

        max_len = max(len(self), len(other))
        a_data = self.data + [self.ty.zero()] * (max_len - len(self))
        b_data = other.data + [other.ty.zero()] * (max_len - len(other))
        return TIKernel(self.ty.add(a_data, b_data), self.ty)

    @overload
    def __sub__(self, other: TIKernel) -> TIKernel: ...

    @overload
    def __sub__(self, other: TVKernel) -> TVKernel: ...

    def __sub__(self, other):
        if isinstance(other, TVKernel):
            return self.promote() - other
        assert self.ty == other.ty
        
        max_len = max(len(self), len(other))
        a_data = self.data + [self.ty.zero()] * (max_len - len(self))
        b_data = other.data + [other.ty.zero()] * (max_len - len(other))
        return TIKernel(self.ty.sub(a_data, b_data), self.ty)

    def __neg__(self) -> TIKernel:
        assert self.ty == Type.Arith, "Negation is only defined for arithmetic type"
        return TIKernel([-x for x in self.data], self.ty)

class TVKernel(KernelExpr):
    data: Sequence[SignalExpr]
    ty: Type
    __match_args__ = ("data",)

    def __init__(self, data: Sequence[SignalExpr], ty: Type):
        super().__init__()
        self.data = data
        self.ty = ty

    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        for i, v in enumerate(self.data[:max_delay]):
            match v:
                case Num(value) if self.ty.is_zero(value):
                    continue
                case _:
                    return i, TVKernel(self.data[i:], self.ty)
        if max_delay < len(self.data):
            return max_delay, TVKernel(self.data[max_delay:], self.ty)
        assert False, "max_delay exceeds kernel length"

    def to_sparse_repr(self) -> Tuple[int, List[int], List[SignalExpr]]:
        """Returns a tuple of (num_nonzero, indices, values) representing the sparse form of the kernel."""
        indices = []
        values = []
        for i, v in enumerate(self.data):
            match v:
                case Num(value) if self.ty.is_zero(value):
                    continue
                case _:
                    indices.append(i)
                    values.append(v)
        return len(indices), indices, values

    def __add__(self, other: TIKernel | TVKernel) -> TVKernel:
        if isinstance(other, TIKernel):
            return self + other.promote()
        assert self.ty == other.ty

        max_len = max(len(self.data), len(other.data))
        data = []
        for i in range(max_len):
            a = self.data[i] if i < len(self.data) else Num(self.ty.zero(), self.ty)
            b = other.data[i] if i < len(other.data) else Num(self.ty.zero(), self.ty)
            data.append(SAdd(a, b))
        return TVKernel(data, self.ty)

    def __sub__(self, other: TIKernel | TVKernel) -> TVKernel:
        return self + (-other)

    def __neg__(self) -> TVKernel:
        assert self.ty == Type.Arith, "Negation is only defined for arithmetic type"
        data = [SNeg(x) for x in self.data]
        return TVKernel(data, self.ty)

    def __mul__(self, other: TIKernel | TVKernel | float) -> TVKernel:
        if isinstance(other, TIKernel):
            return self * other.promote()
        elif isinstance(other, TVKernel):
            max_len = len(self.data) + len(other.data) - 1
            data = []
            for i in range(max_len):
                terms = []
                for j in range(len(self.data)):
                    if 0 <= i - j < len(other.data):
                        terms.append(
                            PointwiseMul(
                                self.data[j],
                                Convolve(TIKernel.z(self.ty, -j), other.data[i - j]),
                            )
                        )
                terms_seq: Sequence[SignalExpr] = terms
                if terms:
                    term = reduce(lambda x, y: SAdd(x, y), terms_seq)
                else:
                    term = Num(self.ty.zero(), self.ty)
                data.append(term)
            return TVKernel(data, self.ty)
        else:
            data = [PointwiseMul(Num(other, self.ty), x) for x in self.data]
            return TVKernel(data, self.ty)

    def __rmul__(self, other: float) -> TVKernel:
        return self * other

    def __repr__(self) -> str:
        data = ", ".join([str(x) for x in self.data])
        return f"TVKernel([ {data} ])"


KernelConstant = TIKernel | TVKernel


@dataclass
class KernelExprBinOp(KernelExpr):
    a: KernelExpr
    b: KernelExpr
    ty: Type
    __match_args__ = ("a", "b")

    def __init__(self, a: KernelExpr, b: KernelExpr) -> None:
        super().__init__()
        assert a.ty == b.ty
        self.a = a
        self.b = b
        self.ty = a.ty

    def time_delay(self, max_delay: int) -> tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        a_delay, _ = self.a.time_delay(max_delay)
        b_delay, _ = self.b.time_delay(max_delay)
        delay = min(a_delay, b_delay)
        return delay, KConvolve(TIKernel.z(self.ty, -delay), self)

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
        return delay, KConvolve(TIKernel.z(self.ty, -delay), self)


class KNeg(KernelExpr):
    a: KernelExpr
    ty: Type
    __match_args__ = ("a",)

    def __init__(self, a: KernelExpr) -> None:
        super().__init__()
        self.a = a
        self.ty = a.ty

    def time_delay(self, max_delay: int) -> tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        delay, e = self.a.time_delay(max_delay)
        return delay, KNeg(e)

# SignalExpr


class Num(SignalExpr):
    value: float
    ty: Type
    __match_args__ = ("value",)

    def __init__(self, value: float, ty: Type) -> None:
        super().__init__()
        self.value = value
        self.ty = ty

class SignalExprBinOp(SignalExpr):
    a: SignalExpr
    b: SignalExpr
    __match_args__ = ("a", "b")

    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__()
        assert a.ty == b.ty
        self.a = a
        self.b = b
        self.ty = a.ty

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


@dataclass
class Convolve(SignalExpr):
    a: KernelExpr
    b: SignalExpr
    __match_args__ = ("a", "b")

    def __init__(self, a: KernelExpr, b: SignalExpr) -> None:
        super().__init__()
        assert a.ty == b.ty
        self.ty = a.ty
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
        self.ty = a.ty
        self.a = a
        self.g = g


@dataclass
class Var(SignalExpr):
    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Type) -> None:
        super().__init__()
        self.name = name
        self.ty = ty

# Convert Lanes
class ConvertLanes(SignalExpr):
    a: SignalExpr
    __match_args__ = ("a",)

    def __init__(self, a: SignalExpr, lanes: int) -> None:
        super().__init__()
        self.a = a
        self.ty = a.ty
        self.lanes = lanes

class KConvertLanes(KernelExpr):
    a: KernelExpr
    __match_args__ = ("a",)

    def __init__(self, a: KernelExpr, lanes: int) -> None:
        super().__init__()
        self.a = a
        self.ty = a.ty
        self.lanes = lanes
    
    def time_delay(self, max_delay: int) -> tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        delay, e = self.a.time_delay(max_delay)
        assert self.lanes is not None
        return delay, KConvertLanes(e, self.lanes)