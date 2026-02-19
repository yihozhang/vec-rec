from __future__ import annotations
from dataclasses import dataclass

from .base import *
from vecrec.util import ElementType


class Num(SignalExpr):
    value: float
    ty: Type
    element_type: ElementType
    __match_args__ = ("value",)

    def __init__(self, value: float, ty: Type, element_type: ElementType = ElementType.Float) -> None:
        super().__init__()
        self.value = value
        self.ty = ty
        self.element_type = element_type

@dataclass
class Var(SignalExpr):
    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Type, element_type: ElementType) -> None:
        super().__init__()
        self.name = name
        self.ty = ty
        self.element_type = element_type

@dataclass
class RVar2D(SignalExpr2D):
    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Type, element_type: ElementType) -> None:
        super().__init__()
        self.name = name
        self.ty = ty
        self.element_type = element_type
