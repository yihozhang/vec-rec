from __future__ import annotations
from functools import reduce
from typing import List, Optional, Sequence, Tuple, overload

from vecrec.expr.signal import Num
from vecrec.expr.signal_ops import *
from vecrec.util import allclose, ElementType
from .base import Type, KernelExpr, KernelExpr2D, SignalExpr, _resolve_types


class TIKernel(KernelExpr):
    @staticmethod
    def z(ty: Optional[Type] = None, element_type: Optional[ElementType] = None, n: int = -1) -> TIKernel:
        """Return the delay kernel z^n."""
        if n > 0:
            raise ValueError(f"n must be negative, got {n}")
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        return TIKernel([resolved_ty.zero()] * (-n) + [resolved_ty.one()], resolved_ty, resolved_et)

    @staticmethod
    def i(ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> TIKernel:
        """Return the identity kernel."""
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        return TIKernel([resolved_ty.one()], resolved_ty, resolved_et)

    ty: Type
    element_type: ElementType
    data: List[float]
    __match_args__ = ("data",)

    def __init__(self, data: list[float], ty: Optional[Type] = None, element_type: Optional[ElementType] = None):
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        for i in range(len(data)):
            if resolved_ty.is_zero(data[i]):
                data[i] = resolved_ty.zero()
        while len(data) > 0 and resolved_ty.is_zero(data[-1]):
            data.pop()
        self.data = data

    def promote(self) -> TVKernel:
        promoted = TVKernel(
            [Num(x, self.ty, self.element_type) for x in self.data],
            self.ty,
            self.element_type,
        )
        promoted.lanes = self.lanes
        return promoted

    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        for i, v in enumerate(self.data[:max_delay]):
            if not self.ty.is_zero(v):
                return i, TIKernel(self.data[i:], self.ty, self.element_type).with_lanes(self.lanes)
        if max_delay < len(self.data):
            return max_delay, TIKernel(self.data[max_delay:], self.ty, self.element_type).with_lanes(self.lanes)
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
        return hash((self.lanes, tuple(self.data)))

    def copy(self) -> TIKernel:
        copied = TIKernel(self.data.copy(), self.ty, self.element_type)
        copied.lanes = self.lanes
        return copied

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        data = " ".join([f"{x:.8f}" for x in self.data])
        if self.lanes is not None:
            return f"TIKernel(lanes={self.lanes}, [ {data} ])"
        else:
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
            assert (
                self.element_type == other.element_type
            ), f"ElementType mismatch in TIKernel multiplication: {self.element_type} vs {other.element_type}"
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
        assert (
            self.element_type == other.element_type
        ), f"ElementType mismatch in TIKernel addition: {self.element_type} vs {other.element_type}"

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
        assert (
            self.element_type == other.element_type
        ), f"ElementType mismatch in TIKernel subtraction: {self.element_type} vs {other.element_type}"

        max_len = max(len(self), len(other))
        a_data = self.data + [self.ty.zero()] * (max_len - len(self))
        b_data = other.data + [other.ty.zero()] * (max_len - len(other))
        return TIKernel(self.ty.sub(a_data, b_data), self.ty, self.element_type)

    def __neg__(self) -> TIKernel:
        assert self.ty == Type.Arith, "Negation is only defined for arithmetic type"
        return TIKernel([-x for x in self.data], self.ty, self.element_type)

    def __truediv__(self, other: float) -> TIKernel:
        assert self.ty == Type.Arith, "Division is only defined for arithmetic type"
        return TIKernel([x / other for x in self.data], self.ty, self.element_type)


@dataclass
class TVKernel(KernelExpr):
    data: Sequence[SignalExpr]
    ty: Type
    element_type: ElementType
    __match_args__ = ("data",)

    def __init__(
        self, data: Sequence[SignalExpr], ty: Optional[Type] = None, element_type: Optional[ElementType] = None
    ):
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.data = data

    def children(self):
        return self.data

    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        """Return the time delay of the kernel."""
        for i, v in enumerate(self.data[:max_delay]):
            match v:
                case Num(value) if self.ty.is_zero(value):
                    continue
                case _:
                    return i, TVKernel(self.data[i:], self.ty, self.element_type).with_lanes(self.lanes)
        if max_delay < len(self.data):
            return max_delay, TVKernel(self.data[max_delay:], self.ty, self.element_type).with_lanes(self.lanes)
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
        assert (
            self.element_type == other.element_type
        ), f"ElementType mismatch in TVKernel addition: {self.element_type} vs {other.element_type}"

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
            assert (
                self.element_type == other.element_type
            ), f"ElementType mismatch in TVKernel multiplication: {self.element_type} vs {other.element_type}"
            max_len = len(self.data) + len(other.data) - 1
            data = []
            for i in range(max_len):
                terms = []
                for j in range(len(self.data)):
                    if 0 <= i - j < len(other.data):
                        terms.append(
                            PointwiseMul(
                                self.data[j],
                                Convolve(
                                    TIKernel.z(self.ty, self.element_type, -j),
                                    other.data[i - j],
                                ),
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
        if self.lanes is not None:
            return f"TVKernel(lanes={self.lanes}, [ {data} ])"
        else:
            return f"TVKernel([ {data} ])"

    def __hash__(self):
        return hash((self.lanes, tuple(self.data)))

    def __truediv__(self, other: float) -> TVKernel:
        assert self.ty == Type.Arith, "Division is only defined for arithmetic type"
        data = [PointwiseDiv(x, Num(other, self.ty, self.element_type)) for x in self.data]
        return TVKernel(data, self.ty, self.element_type)


KernelConstant = TIKernel | TVKernel


@dataclass(unsafe_hash=True)
class TIKernel2D(KernelExpr2D):
    """2D time-invariant kernel. data[row][col] gives the coefficient."""

    data: List[List[float]]
    ty: Type
    element_type: ElementType

    __match_args__ = ("data",)

    def __init__(self, data: List[List[float]], ty: Optional[Type] = None, element_type: Optional[ElementType] = None):
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        # Trim trailing zero rows
        while len(data) > 0 and all(resolved_ty.is_zero(v) for v in data[-1]):
            data.pop()
        # Trim trailing zero cols in each row
        for i in range(len(data)):
            while len(data[i]) > 0 and resolved_ty.is_zero(data[i][-1]):
                data[i].pop()
        self.data = data

    def n_rows(self) -> int:
        return len(self.data)

    def promote(self) -> TVKernel2D:
        data: List[List[SignalExpr]] = [[Num(v, self.ty, self.element_type) for v in row] for row in self.data]
        promoted = TVKernel2D(data, self.ty, self.element_type)
        promoted.lanes = self.lanes
        return promoted

    def __neg__(self) -> TIKernel2D:
        assert self.ty == Type.Arith, "Negation is only defined for arithmetic type"
        return TIKernel2D([[-v for v in row] for row in self.data], self.ty, self.element_type)

    @overload
    def __add__(self, other: TIKernel2D) -> TIKernel2D: ...

    @overload
    def __add__(self, other: TVKernel2D) -> TVKernel2D: ...

    def __add__(self, other):
        if isinstance(other, TVKernel2D):
            return self.promote() + other
        assert isinstance(other, TIKernel2D)
        assert self.ty == other.ty
        assert (
            self.element_type == other.element_type
        ), f"ElementType mismatch in TIKernel2D addition: {self.element_type} vs {other.element_type}"
        max_rows = max(len(self.data), len(other.data))
        result = []
        for r in range(max_rows):
            a_row = self.data[r] if r < len(self.data) else []
            b_row = other.data[r] if r < len(other.data) else []
            max_cols = max(len(a_row), len(b_row))
            a_pad = a_row + [self.ty.zero()] * (max_cols - len(a_row))
            b_pad = b_row + [self.ty.zero()] * (max_cols - len(b_row))
            result.append(self.ty.add(a_pad, b_pad))
        return TIKernel2D(result, self.ty, self.element_type)

    @overload
    def __sub__(self, other: TIKernel2D) -> TIKernel2D: ...

    @overload
    def __sub__(self, other: TVKernel2D) -> TVKernel2D: ...

    def __sub__(self, other):
        if isinstance(other, TVKernel2D):
            return self.promote() - other
        assert isinstance(other, TIKernel2D)
        assert self.ty == other.ty
        assert (
            self.element_type == other.element_type
        ), f"ElementType mismatch in TIKernel2D subtraction: {self.element_type} vs {other.element_type}"
        max_rows = max(len(self.data), len(other.data))
        result = []
        for r in range(max_rows):
            a_row = self.data[r] if r < len(self.data) else []
            b_row = other.data[r] if r < len(other.data) else []
            max_cols = max(len(a_row), len(b_row))
            a_pad = a_row + [self.ty.zero()] * (max_cols - len(a_row))
            b_pad = b_row + [self.ty.zero()] * (max_cols - len(b_row))
            result.append(self.ty.sub(a_pad, b_pad))
        return TIKernel2D(result, self.ty, self.element_type)

    @overload
    def __mul__(self, other: TIKernel2D) -> TIKernel2D: ...

    @overload
    def __mul__(self, other: TVKernel2D | float) -> TVKernel2D: ...

    def __mul__(self, other):
        if isinstance(other, TVKernel2D):
            return self.promote() * other
        elif isinstance(other, TIKernel2D):
            assert (
                self.element_type == other.element_type
            ), f"ElementType mismatch in TIKernel2D multiplication: {self.element_type} vs {other.element_type}"
            if len(self.data) == 0 or len(other.data) == 0:
                return TIKernel2D([], self.ty, self.element_type)
            rows_a, rows_b = len(self.data), len(other.data)
            max_cols_a = max((len(row) for row in self.data), default=0)
            max_cols_b = max((len(row) for row in other.data), default=0)
            if max_cols_a == 0 or max_cols_b == 0:
                return TIKernel2D([], self.ty, self.element_type)
            result_rows = rows_a + rows_b - 1
            result_cols = max_cols_a + max_cols_b - 1
            result = [[self.ty.zero()] * result_cols for _ in range(result_rows)]
            for i, row_a in enumerate(self.data):
                for j, a_val in enumerate(row_a):
                    if self.ty.is_zero(a_val):
                        continue
                    for k, row_b in enumerate(other.data):
                        for l, b_val in enumerate(row_b):
                            result[i + k][j + l] = self.ty.add(
                                result[i + k][j + l],
                                self.ty.mult(a_val, b_val),
                            )
            return TIKernel2D(result, self.ty, self.element_type)
        else:
            return TIKernel2D(
                [[self.ty.mult(v, other) for v in row] for row in self.data],
                self.ty,
                self.element_type,
            )

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> TIKernel2D:
        assert self.ty == Type.Arith, "Division is only defined for arithmetic type"
        data = [[self.ty.div(v, other) for v in row] for row in self.data]
        return TIKernel2D(data, self.ty, self.element_type)


@dataclass(unsafe_hash=True)
class TVKernel2D(KernelExpr2D):
    """2D time-varying kernel. data[row][col] gives the signal expression."""

    data: List[List[SignalExpr]]
    ty: Type
    element_type: ElementType

    __match_args__ = ("data",)

    def __init__(
        self, data: List[List[SignalExpr]], ty: Optional[Type] = None, element_type: Optional[ElementType] = None
    ):
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.data = data

    def n_rows(self) -> int:
        return len(self.data)

    def __neg__(self) -> TVKernel2D:
        assert self.ty == Type.Arith, "Negation is only defined for arithmetic type"
        data: List[List[SignalExpr]] = [[SNeg(v) for v in row] for row in self.data]
        return TVKernel2D(data, self.ty, self.element_type)

    def __add__(self, other: TIKernel2D | TVKernel2D) -> TVKernel2D:
        if isinstance(other, TIKernel2D):
            return self + other.promote()
        assert self.ty == other.ty
        assert (
            self.element_type == other.element_type
        ), f"ElementType mismatch in TVKernel2D addition: {self.element_type} vs {other.element_type}"
        max_rows = max(len(self.data), len(other.data))
        result: List[List[SignalExpr]] = []
        for r in range(max_rows):
            a_row = self.data[r] if r < len(self.data) else []
            b_row = other.data[r] if r < len(other.data) else []
            max_cols = max(len(a_row), len(b_row))
            row_result: List[SignalExpr] = []
            for c in range(max_cols):
                a_val: SignalExpr = a_row[c] if c < len(a_row) else Num(self.ty.zero(), self.ty, self.element_type)
                b_val: SignalExpr = b_row[c] if c < len(b_row) else Num(self.ty.zero(), self.ty, self.element_type)
                row_result.append(SAdd(a_val, b_val))
            result.append(row_result)
        return TVKernel2D(result, self.ty, self.element_type)

    def __sub__(self, other: TIKernel2D | TVKernel2D) -> TVKernel2D:
        return self + (-other)

    def __mul__(self, other: TIKernel2D | TVKernel2D | float) -> TVKernel2D:
        if isinstance(other, TIKernel2D):
            return self * other.promote()
        elif isinstance(other, TVKernel2D):
            raise AssertionError(
                "TVKernel2D * TVKernel2D requires row delays on SignalExpr, which are not yet supported"
            )
        else:
            data: List[List[SignalExpr]] = [
                [PointwiseMul(Num(other, self.ty, self.element_type), v) for v in row] for row in self.data
            ]
            return TVKernel2D(data, self.ty, self.element_type)

    def __rmul__(self, other: float) -> TVKernel2D:
        return self * other

    def __truediv__(self, other: float) -> TVKernel2D:
        assert self.ty == Type.Arith, "Division is only defined for arithmetic type"
        data: List[List[SignalExpr]] = []
        for row in self.data:
            data.append([PointwiseDiv(v, Num(other, self.ty, self.element_type)) for v in row])
        return TVKernel2D(data, self.ty, self.element_type)
