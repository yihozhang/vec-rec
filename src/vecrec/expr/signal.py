from __future__ import annotations
from dataclasses import dataclass

from .base import *
from vecrec.util import ElementType

@dataclass(unsafe_hash=True)
class Num(SignalExpr):
    value: float
    ty: Type
    element_type: ElementType
    __match_args__ = ("value",)

    def __init__(self, value: float, ty: Type, element_type: ElementType = ElementType.Float) -> None:
        super().__init__(ty, element_type)
        self.value = value

@dataclass(unsafe_hash=True)
class Impulse(SignalExpr):
    """Produces `value` at time 0 and the zero element of the semiring thereafter.
    Useful for setting up initial conditions of recurrences."""
    value: float
    ty: Type
    element_type: ElementType
    __match_args__ = ("value",)

    def __init__(self, value: float, ty: Type, element_type: ElementType = ElementType.Float) -> None:
        super().__init__(ty, element_type)
        self.value = value

@dataclass(unsafe_hash=True)
class Var(SignalExpr):
    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Type, element_type: ElementType) -> None:
        super().__init__(ty, element_type)
        self.name = name

@dataclass(unsafe_hash=True)
class RVar2D(SignalExpr2D):
    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Type, element_type: ElementType) -> None:
        super().__init__(ty, element_type)
        self.name = name
