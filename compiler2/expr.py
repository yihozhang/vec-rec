from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import numpy as np

__all__ = ["RecLang", "TIKernel", "Convolve", "Recurse", "Var", "Add", "Sub", "Neg"]

class RecLang:
    pass

class TIKernel(RecLang):
    @staticmethod
    def z(n: int = 1) -> TIKernel:
        """Return the delay kernel z^n."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return TIKernel([0.0] * n + [1.0])
    
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
    
    # def __mul__(self, other: TIKernel | float) -> TIKernel:
    #     if isinstance(other, TIKernel):
    #         if len(self) == 0 or len(other) == 0:
    #             return TIKernel([])
    #         return TIKernel(np.convolve(self.data, other.data))
    #     else:
    #         return TIKernel(self.data * other)
    
    # __rmul__ = __mul__
    
    # def __add__(self, other: TIKernel) -> TIKernel:
    #     max_len = max(len(self), len(other))
    #     a_data = np.pad(self.data, (0, max_len - len(self)), 'constant')
    #     b_data = np.pad(other.data, (0, max_len - len(other)), 'constant')
    #     return TIKernel(a_data + b_data)
    
    # def __sub__(self, other: TIKernel) -> TIKernel:
    #     max_len = max(len(self), len(other))
    #     a_data = np.pad(self.data, (0, max_len - len(self)), 'constant')
    #     b_data = np.pad(other.data, (0, max_len - len(other)), 'constant')
    #     return TIKernel(a_data - b_data)
    
    # def __neg__(self) -> TIKernel:
    #     return TIKernel(-self.data)

class Add(RecLang):
    a: RecLang
    b: RecLang
    __match_args__ = ("a", "b")
    def __init__(self, a: RecLang, b: RecLang) -> None:
        super().__init__()
        self.a = a
        self.b = b

class Sub(RecLang):
    a: RecLang
    b: RecLang
    __match_args__ = ("a", "b")
    def __init__(self, a: RecLang, b: RecLang) -> None:
        super().__init__()
        self.a = a
        self.b = b

class Neg(RecLang):
    a: RecLang
    __match_args__ = ("a",)
    def __init__(self, a: RecLang) -> None:
        super().__init__()
        self.a = a

class Convolve(RecLang):
    a: RecLang
    b: RecLang
    __match_args__ = ("a", "b")
    def __init__(self, a: RecLang, b: RecLang) -> None:
        super().__init__()
        self.a = a
        self.b = b

class Recurse(RecLang):
    a: RecLang
    g: RecLang
    __match_args__ = ("a", "g")
    def __init__(self, a: RecLang, g: RecLang) -> None:
        super().__init__()
        self.a = a
        self.g = g

class Var(RecLang):
    name: str
    __match_args__ = ("name",)
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


