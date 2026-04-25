from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .base import *
from .base import _resolve_types
from vecrec.util import ElementType

@dataclass(unsafe_hash=True)
class Num(SignalExpr):
    value: float
    ty: Type
    element_type: ElementType
    __match_args__ = ("value",)

    def __init__(self, value: float, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.value = value

@dataclass(unsafe_hash=True)
class Impulse(SignalExpr):
    """Produces `value` at time 0 and the zero element of the semiring thereafter.
    Useful for setting up initial conditions of recurrences."""
    value: float
    ty: Type
    element_type: ElementType
    __match_args__ = ("value",)

    def __init__(self, value: float, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.value = value

@dataclass(unsafe_hash=True)
class Var(SignalExpr):
    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.name = name

@dataclass(unsafe_hash=True)
class RVar2D(SignalExpr2D):
    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.name = name

class RVec2D:
    """Container for n RVar2D objects used as the prev-state argument inside RepeaterVec bodies.
    Not a RecLang node — just a plain Python helper."""

    vars: tuple[RVar2D, ...]

    def __init__(self, vars: list[RVar2D]) -> None:
        self.vars = tuple(vars)

    def at(self, i: int) -> RVar2D:
        return self.vars[i]


@dataclass(unsafe_hash=True)
class Var2D(SignalExpr2D):
    """A user-provided 2D input array. Exposes n_rows consecutive rows simultaneously.
    n_rows is inferred by InferNRows from Ith usages; do not set manually."""
    name: str
    n_rows: Optional[int]
    __match_args__ = ("name", "n_rows")

    def __init__(self, name: str, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.name = name
        self.n_rows = None
