from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .base import KernelMatrixExpr, SignalExpr, SignalVectorExpr


@dataclass(unsafe_hash=True)
class SignalVector(SignalVectorExpr):
    """A vector of n SignalExpr values that are computed independently.

    ``data[i]`` is the i-th component signal.
    """

    data: Tuple[SignalExpr, ...]
    __match_args__ = ("data",)

    def __init__(self, data: Sequence[SignalExpr]) -> None:
        signals = tuple(data)
        n = len(signals)
        assert n > 0, "SignalVector must have at least one signal"
        ty = signals[0].ty
        element_type = signals[0].element_type
        assert all(s.ty == ty for s in signals), "All signals must have the same type"
        assert all(s.element_type == element_type for s in signals), "All signals must have the same element_type"
        super().__init__(ty, element_type, n)
        self.data = signals

    def children(self) -> List[SignalExpr]:  # type: ignore[override]
        return list(self.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SignalVector):
            return False
        return self.n == other.n and self.lanes == other.lanes and self.data == other.data

    def __repr__(self) -> str:
        return f"SignalVector(n={self.n}, data={self.data})"


@dataclass(unsafe_hash=True)
class ConvolveV(SignalVectorExpr):
    """Vectorized FIR: out = A ⊛ g, where A is an n×n kernel matrix and g is an n-vector.

    ``out[i] = sum_j A[i][j] ⊛ g[j]``
    """

    a: KernelMatrixExpr
    b: SignalVectorExpr
    __match_args__ = ("a", "b")

    def __init__(self, a: KernelMatrixExpr, b: SignalVectorExpr) -> None:
        assert a.n == b.n, f"KernelMatrix and SignalVector size mismatch: {a.n} vs {b.n}"
        assert a.ty == b.ty, f"Type mismatch in ConvolveV: {a.ty} vs {b.ty}"
        assert a.element_type == b.element_type, f"ElementType mismatch in ConvolveV: {a.element_type} vs {b.element_type}"
        super().__init__(a.ty, a.element_type, a.n)
        self.a = a
        self.b = b

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConvolveV):
            return False
        return self.lanes == other.lanes and self.a == other.a and self.b == other.b

    def __repr__(self) -> str:
        return f"ConvolveV({self.a}, {self.b})"


@dataclass(unsafe_hash=True)
class RecurseV(SignalVectorExpr):
    """Vectorized IIR: F = A ⊛ F + G, where F,G are n-vectors and A is n×n.

    ``F[i][t] = sum_j sum_k A[i][j][k] * F[j][t-k] + G[i][t]``

    Each kernel ``A[i][j]`` must have a causal delay ≥ lanes (same constraint as
    scalar Recurse) so that vectorized batches are independent.
    """

    a: KernelMatrixExpr
    g: SignalVectorExpr
    __match_args__ = ("a", "g")

    def __init__(self, a: KernelMatrixExpr, g: SignalVectorExpr) -> None:
        assert a.n == g.n, f"KernelMatrix and SignalVector size mismatch: {a.n} vs {g.n}"
        assert a.ty == g.ty, f"Type mismatch in RecurseV: {a.ty} vs {g.ty}"
        assert a.element_type == g.element_type, f"ElementType mismatch in RecurseV: {a.element_type} vs {g.element_type}"
        super().__init__(a.ty, a.element_type, a.n)
        self.a = a
        self.g = g

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecurseV):
            return False
        return self.lanes == other.lanes and self.a == other.a and self.g == other.g

    def __repr__(self) -> str:
        return f"RecurseV({self.a}, {self.g})"


@dataclass(unsafe_hash=True)
class VectorAt(SignalExpr):
    """Extract component ``i`` from a SignalVectorExpr as a 1D SignalExpr."""

    a: SignalVectorExpr
    i: int
    __match_args__ = ("a", "i")

    def __init__(self, a: SignalVectorExpr, i: int) -> None:
        assert 0 <= i < a.n, f"VectorAt index {i} out of range for n={a.n}"
        super().__init__(a.ty, a.element_type)
        self.a = a
        self.i = i

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorAt):
            return False
        return self.lanes == other.lanes and self.a == other.a and self.i == other.i

    def __repr__(self) -> str:
        return f"VectorAt({self.a}, {self.i})"
