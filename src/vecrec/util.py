from enum import Enum
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vecrec.expr import Type

__all__ = ["ElementType", "allclose"]

class ElementType(Enum):
    Float = 1
    I32 = 2
    I64 = 4

    def bit_width(self) -> int:
        match self:
            case ElementType.Float:
                return 32
            case ElementType.I32:
                return 32
            case ElementType.I64:
                return 64

    def to_str(self) -> str:
        match self:
            case ElementType.Float:
                return "float"
            case ElementType.I32:
                return "int32_t"
            case ElementType.I64:
                return "int64_t"
    
    def val_to_str(self, val: float, ty: "Type | None" = None) -> str:
        # Handle infinity for tropical types
        if ty is not None and np.isinf(val):
            if self == ElementType.Float:
                return "-INFINITY" if val < 0 else "INFINITY"
            elif self == ElementType.I32:
                return "INT32_MIN" if val < 0 else "INT32_MAX"
            elif self == ElementType.I64:
                return "INT64_MIN" if val < 0 else "INT64_MAX"

        # Regular value handling
        match self:
            case ElementType.Float:
                return f"{val}"
            case ElementType.I32:
                return str(int(val))
            case ElementType.I64:
                return str(int(val))

def allclose(a, b, tol=1e-6):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > tol:
            return False
    return True
