from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from vecrec.expr.kernel import TIKernel
from vecrec.util import ElementType

from .base import KernelExpr, KernelMatrixExpr, Type


@dataclass(unsafe_hash=True)
class KernelMatrix(KernelMatrixExpr):
    """An n×n matrix of KernelExpr entries.

    ``data[i][j]`` is the kernel for row i (output i) and column j (input j).
    Entries may be any KernelExpr; zero feedback on a path can be represented
    by a TIKernel([]) (the constructor strips all trailing zeros so
    TIKernel([0.0]) becomes TIKernel([])).
    """

    data: Tuple[Tuple[KernelExpr, ...], ...]
    __match_args__ = ("data",)

    def __init__(
        self,
        data: Sequence[Sequence[KernelExpr]],
    ) -> None:
        rows: List[Tuple[KernelExpr, ...]] = [tuple(row) for row in data]
        n = len(rows)
        assert n > 0, "KernelMatrix must have at least one row"
        assert all(
            len(row) == n for row in rows
        ), f"KernelMatrix must be square, got rows of lengths {[len(r) for r in rows]}"

        ty = rows[0][0].ty
        element_type = rows[0][0].element_type
        assert all(k.ty == ty for row in rows for k in row), "All kernels in KernelMatrix must have the same type"
        assert all(
            k.element_type == element_type for row in rows for k in row
        ), "All kernels in KernelMatrix must have the same element_type"

        super().__init__(ty, element_type, n)
        self.data = tuple(rows)
    
    @staticmethod
    def i(n: int, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> KernelMatrix:
        """Return the identity kernel matrix."""
        identity_kernel = TIKernel.i(ty, element_type)
        return KernelMatrix([[identity_kernel if i == j else TIKernel([], ty, element_type) for j in range(n)] for i in range(n)])

    def children(self) -> List[KernelExpr]:  # type: ignore[override]
        return [k for row in self.data for k in row]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KernelMatrix):
            return False
        return self.n == other.n and self.lanes == other.lanes and self.data == other.data

    def __repr__(self) -> str:
        return f"KernelMatrix(n={self.n}, data={self.data})"
    
    def __sub__(self, other: KernelMatrix) -> KernelMatrix:
        assert self.n == other.n and self.ty == other.ty and self.element_type == other.element_type, "KernelMatrix subtraction requires matching dimensions and types"
        return KernelMatrix(
            [[self.data[i][j] - other.data[i][j] for j in range(self.n)] for i in range(self.n)], # type: ignore
        )
    
    def __add__(self, other: KernelMatrix) -> KernelMatrix:
        assert self.n == other.n and self.ty == other.ty and self.element_type == other.element_type, "KernelMatrix addition requires matching dimensions and types"
        return KernelMatrix(
            # TODO: get rid of KAdd, KSub, KConvolve and use +/-/* directly
            [[self.data[i][j] + other.data[i][j] for j in range(self.n)] for i in range(self.n)], # type: ignore
        )
    
    def __neg__(self):
        return KernelMatrix(
            [[-k for k in row] for row in self.data], # type: ignore
        )
    
    def __matmul__(self, other: KernelMatrix) -> KernelMatrix:
        assert self.n == other.n and self.ty == other.ty and self.element_type == other.element_type, "KernelMatrix multiplication requires matching dimensions and types"
        return KernelMatrix(
            [[sum(self.data[i][k] * other.data[k][j] for k in range(self.n)) for j in range(self.n)] for i in range(self.n)], # type: ignore
        )
    
    def __mul__(self, other: float | int) -> KernelMatrix:
        return KernelMatrix(
            [[k * other for k in row] for row in self.data], # type: ignore
        )
    
    __rmul__ = __mul__
