from __future__ import annotations
from functools import reduce
from typing import List, Sequence, Tuple, overload

from vecrec.expr.signal import Num
from vecrec.expr.signal_ops import *
from vecrec.util import allclose, ElementType
from .base import Type, KernelExpr, KernelExpr2D, SignalExpr

class TIKernel(KernelExpr):
    @staticmethod
    def z(ty: Type, element_type: ElementType, n: int = -1) -> TIKernel:
        """Return the delay kernel z^n."""
        if n >= 0:
            raise ValueError("n must be negative")
        return TIKernel([0.0] * (-n) + [ty.one()], ty, element_type)

    @staticmethod
    def i(ty: Type, element_type: ElementType) -> TIKernel:
        """Return the identity kernel."""
        return TIKernel([ty.one()], ty, element_type)

    ty: Type
    element_type: ElementType
    data: List[float]
    __match_args__ = ("data",)

    def __init__(self, data: list[float], ty: Type, element_type: ElementType):
        super().__init__()
        for i in range(len(data)):
            if ty.is_zero(data[i]):
                data[i] = ty.zero()
        while len(data) > 0 and ty.is_zero(data[-1]):
            data.pop()
        self.data = data
        self.ty = ty
        self.element_type = element_type

    def promote(self) -> TVKernel:
        promoted = TVKernel([Num(x, self.ty, self.element_type) for x in self.data], self.ty, self.element_type)
        promoted.lanes = self.lanes
        return promoted

    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        for i, v in enumerate(self.data[:max_delay]):
            if not self.ty.is_zero(v):
                return i, TIKernel(self.data[i:], self.ty, self.element_type).with_lanes(self.lanes)
        if max_delay < len(self.data):
            return max_delay, TIKernel(self.data[max_delay:], self.ty, self.element_type).with_lanes(
                self.lanes
            )
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
        copied = TIKernel(self.data.copy(), self.ty, self.element_type)
        copied.lanes = self.lanes
        return copied

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
            assert self.element_type == other.element_type, \
                f"ElementType mismatch in TIKernel multiplication: {self.element_type} vs {other.element_type}"
            if len(self) == 0 or len(other) == 0:
                return TIKernel([], self.ty, self.element_type)
            return TIKernel(self.ty.convolve(self.data, other.data), self.ty, self.element_type)
        else:
            return TIKernel(self.ty.mult(self.data, other), self.ty, self.element_type)

    __rmul__ = __mul__

    @overload
    def __add__(self, other: TIKernel) -> TIKernel: ...

    @overload
    def __add__(self, other: TVKernel) -> TVKernel: ...

    def __add__(self, other):
        if isinstance(other, TVKernel):
            return self.promote() + other
        assert self.ty == other.ty
        assert self.element_type == other.element_type, \
            f"ElementType mismatch in TIKernel addition: {self.element_type} vs {other.element_type}"

        max_len = max(len(self), len(other))
        a_data = self.data + [self.ty.zero()] * (max_len - len(self))
        b_data = other.data + [other.ty.zero()] * (max_len - len(other))
        return TIKernel(self.ty.add(a_data, b_data), self.ty, self.element_type)

    @overload
    def __sub__(self, other: TIKernel) -> TIKernel: ...

    @overload
    def __sub__(self, other: TVKernel) -> TVKernel: ...

    def __sub__(self, other):
        if isinstance(other, TVKernel):
            return self.promote() - other
        assert self.ty == other.ty
        assert self.element_type == other.element_type, \
            f"ElementType mismatch in TIKernel subtraction: {self.element_type} vs {other.element_type}"

        max_len = max(len(self), len(other))
        a_data = self.data + [self.ty.zero()] * (max_len - len(self))
        b_data = other.data + [other.ty.zero()] * (max_len - len(other))
        return TIKernel(self.ty.sub(a_data, b_data), self.ty, self.element_type)

    def __neg__(self) -> TIKernel:
        assert self.ty == Type.Arith, "Negation is only defined for arithmetic type"
        return TIKernel([-x for x in self.data], self.ty, self.element_type)


class TVKernel(KernelExpr):
    data: Sequence[SignalExpr]
    ty: Type
    element_type: ElementType
    __match_args__ = ("data",)

    def __init__(self, data: Sequence[SignalExpr], ty: Type, element_type: ElementType):
        super().__init__()
        self.data = data
        self.ty = ty
        self.element_type = element_type

    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        for i, v in enumerate(self.data[:max_delay]):
            match v:
                case Num(value) if self.ty.is_zero(value):
                    continue
                case _:
                    return i, TVKernel(self.data[i:], self.ty, self.element_type).with_lanes(self.lanes)
        if max_delay < len(self.data):
            return max_delay, TVKernel(self.data[max_delay:], self.ty, self.element_type).with_lanes(
                self.lanes
            )
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
        assert self.element_type == other.element_type, \
            f"ElementType mismatch in TVKernel addition: {self.element_type} vs {other.element_type}"

        max_len = max(len(self.data), len(other.data))
        data = []
        for i in range(max_len):
            a = self.data[i] if i < len(self.data) else Num(self.ty.zero(), self.ty, self.element_type)
            b = other.data[i] if i < len(other.data) else Num(self.ty.zero(), self.ty, self.element_type)
            data.append(SAdd(a, b))
        return TVKernel(data, self.ty, self.element_type)

    def __sub__(self, other: TIKernel | TVKernel) -> TVKernel:
        return self + (-other)

    def __neg__(self) -> TVKernel:
        assert self.ty == Type.Arith, "Negation is only defined for arithmetic type"
        data = [SNeg(x) for x in self.data]
        return TVKernel(data, self.ty, self.element_type)

    def __mul__(self, other: TIKernel | TVKernel | float) -> TVKernel:
        if isinstance(other, TIKernel):
            return self * other.promote()
        elif isinstance(other, TVKernel):
            assert self.element_type == other.element_type, \
                f"ElementType mismatch in TVKernel multiplication: {self.element_type} vs {other.element_type}"
            max_len = len(self.data) + len(other.data) - 1
            data = []
            for i in range(max_len):
                terms = []
                for j in range(len(self.data)):
                    if 0 <= i - j < len(other.data):
                        terms.append(
                            PointwiseMul(
                                self.data[j],
                                Convolve(TIKernel.z(self.ty, self.element_type, -j), other.data[i - j]),
                            )
                        )
                terms_seq: Sequence[SignalExpr] = terms
                if terms:
                    term = reduce(lambda x, y: SAdd(x, y), terms_seq)
                else:
                    term = Num(self.ty.zero(), self.ty, self.element_type)
                data.append(term)
            return TVKernel(data, self.ty, self.element_type)
        else:
            data = [PointwiseMul(Num(other, self.ty, self.element_type), x) for x in self.data]
            return TVKernel(data, self.ty, self.element_type)

    def __rmul__(self, other: float) -> TVKernel:
        return self * other

    def __repr__(self) -> str:
        data = ", ".join([str(x) for x in self.data])
        return f"TVKernel([ {data} ])"


KernelConstant = TIKernel | TVKernel


class TIKernel2D(KernelExpr2D):
    """2D time-invariant kernel. data[row][col] gives the coefficient."""
    data: List[List[float]]
    ty: Type
    element_type: ElementType

    def __init__(self, data: List[List[float]], ty: Type, element_type: ElementType):
        super().__init__()
        # Trim trailing zero rows
        while len(data) > 0 and all(ty.is_zero(v) for v in data[-1]):
            data.pop()
        # Trim trailing zero cols in each row
        for i in range(len(data)):
            while len(data[i]) > 0 and ty.is_zero(data[i][-1]):
                data[i].pop()
        self.data = data
        self.ty = ty
        self.element_type = element_type

    def n_rows(self) -> int:
        return len(self.data)


class TVKernel2D(KernelExpr2D):
    """2D time-varying kernel. data[row][col] gives the signal expression."""
    data: List[List[SignalExpr]]
    ty: Type
    element_type: ElementType

    def __init__(self, data: List[List[SignalExpr]], ty: Type, element_type: ElementType):
        super().__init__()
        self.data = data
        self.ty = ty
        self.element_type = element_type

    def n_rows(self) -> int:
        return len(self.data)

    def children(self) -> List[KernelExpr | SignalExpr]:
        result: List[KernelExpr | SignalExpr] = []
        for row in self.data:
            for v in row:
                result.append(v)
        return result
