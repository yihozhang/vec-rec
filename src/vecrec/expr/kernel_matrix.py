from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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

    def children(self) -> List[KernelExpr]:  # type: ignore[override]
        return [k for row in self.data for k in row]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KernelMatrix):
            return False
        return self.n == other.n and self.lanes == other.lanes and self.data == other.data

    def __repr__(self) -> str:
        return f"KernelMatrix(n={self.n}, data={self.data})"
