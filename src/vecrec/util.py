from enum import Enum

class ElementType(Enum):
    Float = 1
    I32 = 2
    U32 = 3
    I64 = 4
    U64 = 5

    def bit_width(self) -> int:
        match self:
            case ElementType.Float:
                return 32
            case ElementType.I32:
                return 32
            case ElementType.U32:
                return 32
            case ElementType.I64:
                return 64
            case ElementType.U64:
                return 64

    def to_str(self) -> str:
        match self:
            case ElementType.Float:
                return "float"
            case ElementType.I32:
                return "int32_t"
            case ElementType.U32:
                return "uint32_t"
            case ElementType.I64:
                return "int64_t"
            case ElementType.U64:
                return "uint64_t"

def allclose(a, b, tol=1e-6):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > tol:
            return False
    return True