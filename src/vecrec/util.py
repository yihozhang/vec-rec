from enum import Enum
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vecrec.expr import Type

__all__ = ["ElementType", "allclose", "get_native_bits"]


class ElementType(Enum):
    Float = 1
    I32 = 2
    I64 = 4
    I8 = 8

    def bit_width(self) -> int:
        match self:
            case ElementType.Float:
                return 32
            case ElementType.I32:
                return 32
            case ElementType.I64:
                return 64
            case ElementType.I8:
                return 8

    def to_str(self) -> str:
        match self:
            case ElementType.Float:
                return "float"
            case ElementType.I32:
                return "int32_t"
            case ElementType.I64:
                return "int64_t"
            case ElementType.I8:
                return "int8_t"

    def val_to_str(self, val: float, ty: "Type | None" = None) -> str:
        # Handle infinity for tropical types
        if ty is not None and np.isinf(val):
            if self == ElementType.Float:
                return "-INFINITY" if val < 0 else "INFINITY"
            elif self == ElementType.I32:
                return "(INT32_MIN/8)" if val < 0 else "(INT32_MAX/8)"
            elif self == ElementType.I64:
                return "(INT64_MIN/8)" if val < 0 else "(INT64_MAX/8)"
            elif self == ElementType.I8:
                # Use /2 (not /8) as sentinel; saturating arithmetic prevents overflow
                return "(INT8_MIN/2)" if val < 0 else "(INT8_MAX/2)"

        # Regular value handling
        match self:
            case ElementType.Float:
                return f"{val}"
            case ElementType.I32:
                return str(int(val))
            case ElementType.I64:
                return str(int(val))
            case ElementType.I8:
                return str(int(val))


def allclose(a, b, tol=1e-6):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > tol:
            return False
    return True


def get_native_bits() -> int:
    # run lscpu and grep for avx512
    import subprocess

    try:
        output = subprocess.check_output("lscpu", text=True)
        for line in output.splitlines():
            if "avx512" in line.lower():
                return 512
    except subprocess.CalledProcessError:
        pass
    return 256
