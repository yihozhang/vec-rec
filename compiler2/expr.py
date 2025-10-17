from __future__ import annotations
from functools import reduce
from typing import List, Optional, Dict, Sequence, Tuple, overload
import numpy as np

__all__ = [
    "RecLang",
    "SignalExpr",
    "KernelExpr",
    "KernelConstant",
    "TIKernel",
    "TVKernel",
    "KAdd",
    "KSub",
    "KNeg",
    "KConvolve",
    "Num",
    "SAdd",
    "SSub",
    "PointwiseMul",
    "SNeg",
    "Convolve",
    "Recurse",
    "Var",
]

class RecLang:
    pass

class SignalExpr(RecLang):
    pass

class KernelExpr(RecLang):
    pass

class TIKernel(KernelExpr):
    @staticmethod
    def z(n: int = -1) -> TIKernel:
        """Return the delay kernel z^n."""
        if n >= 0:
            raise ValueError("n must be negative")
        return TIKernel([0.0] * (-n) + [1.0])
    
    @staticmethod
    def i() -> TIKernel:
        """Return the identity kernel."""
        return TIKernel([1.0])

    data: np.ndarray
    __match_args__ = ("data",)
    def __init__(self, data: list[float] | np.ndarray):
        super().__init__()
        assert isinstance(data, (list, np.ndarray))
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        mask = np.isclose(data, 0.0)
        data[mask] = 0.0
        data = np.trim_zeros(data, 'b')
        self.data = data
        
    def promote(self) -> TVKernel:
        return TVKernel([Num(x) for x in self.data])
    
    def __getitem__(self, index: int) -> float:
        if len(self.data) <= index:
            return 0.0
        return self.data[index]

    def __hash__(self):
        return hash(tuple(self.data))

    def copy(self) -> TIKernel:
        return TIKernel(self.data.copy())
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self) -> str:
        data = " ".join([f"{x:.8f}" for x in self.data])
        return f"TIKernel([ {data} ])"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TIKernel):
            return False
        return np.allclose(self.data, other.data)
    
    def num_terms(self) -> int:
        """Return the number of non-zero terms in the signal."""
        return int(np.count_nonzero(self.data))
    
    @overload
    def __mul__(self, other: TIKernel | float) -> TIKernel:
        ...
    
    @overload
    def __mul__(self, other: TVKernel) -> TVKernel:
        ...    
    
    def __mul__(self, other):
        if isinstance(other, TVKernel):
            return self.promote() * other
        elif isinstance(other, TIKernel):
            if len(self) == 0 or len(other) == 0:
                return TIKernel([])
            return TIKernel(np.convolve(self.data, other.data))
        else:
            return TIKernel(self.data * other)
    
    __rmul__ = __mul__
    
    @overload
    def __add__(self, other: TIKernel) -> TIKernel:
        ...
    
    @overload
    def __add__(self, other: TVKernel) -> TVKernel:
        ...
    
    def __add__(self, other):
        if isinstance(other, TVKernel):
            return self.promote() + other
        max_len = max(len(self), len(other))
        a_data = np.pad(self.data, (0, max_len - len(self)), 'constant')
        b_data = np.pad(other.data, (0, max_len - len(other)), 'constant')
        return TIKernel(a_data + b_data)
    
    @overload
    def __sub__(self, other: TIKernel) -> TIKernel:
        ...
    
    @overload
    def __sub__(self, other: TVKernel) -> TVKernel:
        ...
    
    def __sub__(self, other):
        if isinstance(other, TVKernel):
            return self.promote() - other
        max_len = max(len(self), len(other))
        a_data = np.pad(self.data, (0, max_len - len(self)), 'constant')
        b_data = np.pad(other.data, (0, max_len - len(other)), 'constant')
        return TIKernel(a_data - b_data)
    
    def __neg__(self) -> TIKernel:
        return TIKernel(-self.data)

class TVKernel(KernelExpr):
    data: Sequence[SignalExpr]
    __match_args__ = ("data",)
    def __init__(self, data: Sequence[SignalExpr]):
        super().__init__()
        self.data = data
    
    def __add__(self, other: TIKernel | TVKernel) -> TVKernel:
        if isinstance(other, TIKernel):
            return self + other.promote()
        max_len = max(len(self.data), len(other.data))
        data = []
        for i in range(max_len):
            a = self.data[i] if i < len(self.data) else Num(0.0)
            b = other.data[i] if i < len(other.data) else Num(0.0)
            data.append(SAdd(a, b))
        return TVKernel(data)
    
    def __sub__(self, other: TIKernel | TVKernel) -> TVKernel:
        return self + (-other)
    
    def __neg__(self) -> TVKernel:
        data = [SNeg(x) for x in self.data]
        return TVKernel(data)
    
    def __mul__(self, other: TIKernel | TVKernel | float) -> TVKernel:
        if isinstance(other, TIKernel):
            return self * other.promote()
        elif isinstance(other, TVKernel):
            max_len = len(self.data) + len(other.data) - 1
            data = []
            for i in range(max_len):
                terms = []
                for j in range(len(self.data)):
                    if 0 <= i - j < len(other.data):
                        terms.append(PointwiseMul(self.data[j], Convolve(TIKernel.z(-j), other.data[i - j])))
                terms_seq: Sequence[SignalExpr] = terms
                if terms:
                    term = reduce(lambda x, y: SAdd(x, y), terms_seq)
                else:
                    term = Num(0.0)
                data.append(term)
            return TVKernel(data)
        else:
            data = [PointwiseMul(Num(other), x) for x in self.data]
            return TVKernel(data)
    
    def __rmul__(self, other: float) -> TVKernel:
        return self * other

KernelConstant = TIKernel | TVKernel

class KAdd(KernelExpr):
    a: KernelExpr
    b: KernelExpr
    __match_args__ = ("a", "b")
    def __init__(self, a: KernelExpr, b: KernelExpr) -> None:
        super().__init__()
        self.a = a
        self.b = b

class KSub(KernelExpr):
    a: KernelExpr
    b: KernelExpr
    __match_args__ = ("a", "b")
    def __init__(self, a: KernelExpr, b: KernelExpr) -> None:
        super().__init__()
        self.a = a
        self.b = b

class KNeg(KernelExpr):
    a: KernelExpr
    __match_args__ = ("a",)
    def __init__(self, a: KernelExpr) -> None:
        super().__init__()
        self.a = a

class KConvolve(KernelExpr):
    a: KernelExpr
    b: KernelExpr
    __match_args__ = ("a", "b")
    def __init__(self, a: KernelExpr, b: KernelExpr) -> None:
        super().__init__()
        self.a = a
        self.b = b    

# SignalExpr

class Num(SignalExpr):
    value: float
    __match_args__ = ("value",)
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

class SAdd(SignalExpr):
    a: SignalExpr
    b: SignalExpr
    __match_args__ = ("a", "b")
    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__()
        self.a = a
        self.b = b
        
class SSub(SignalExpr):
    a: SignalExpr
    b: SignalExpr
    __match_args__ = ("a", "b")
    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__()
        self.a = a
        self.b = b

class PointwiseMul(SignalExpr):
    a: SignalExpr
    b: SignalExpr
    __match_args__ = ("a", "b")
    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__()
        self.a = a
        self.b = b

class SNeg(SignalExpr):
    a: SignalExpr
    __match_args__ = ("a",)
    def __init__(self, a: SignalExpr) -> None:
        super().__init__()
        self.a = a

class Convolve(SignalExpr):
    a: KernelExpr
    b: SignalExpr
    __match_args__ = ("a", "b")
    def __init__(self, a: KernelExpr, b: SignalExpr) -> None:
        super().__init__()
        self.a = a
        self.b = b
    

class Recurse(SignalExpr):
    a: KernelExpr
    g: SignalExpr
    __match_args__ = ("a", "g")
    def __init__(self, a: KernelExpr, g: SignalExpr) -> None:
        super().__init__()
        self.a = a
        self.g = g

class Var(SignalExpr):
    name: str
    __match_args__ = ("name",)
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


