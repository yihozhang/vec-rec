from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from egglog import *
import numpy as np

egraph = EGraph()

class Kernel:
    data: np.array
    def __init__(self, data: list[float] | np.ndarray):
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

    def copy(self) -> Kernel:
        return Kernel(self.data.copy())
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self) -> str:
        data = " ".join([f"{x:.8f}" for x in self.data])
        return f"Kernel([ {data} ])"
    
    def num_terms(self) -> int:
        """Return the number of non-zero terms in the signal."""
        return len(self.data) - self.data.count(0.)
    
    def __mul__(self, other: Kernel | float) -> Kernel:
        if isinstance(other, Kernel):
            if len(self) == 0 or len(other) == 0:
                return Kernel([])
            return Kernel(np.convolve(self.data, other.data))
        else:
            return Kernel(self.data * other)
    
    __rmul__ = __mul__
    
    def __add__(self, other: Kernel) -> Kernel:
        max_len = max(len(self), len(other))
        a_data = np.pad(self.data, (0, max_len - len(self)), 'constant')
        b_data = np.pad(other.data, (0, max_len - len(other)), 'constant')
        return Kernel(a_data + b_data)
    
    def __sub__(self, other: Kernel) -> Kernel:
        max_len = max(len(self), len(other))
        a_data = np.pad(self.data, (0, max_len - len(self)), 'constant')
        b_data = np.pad(other.data, (0, max_len - len(other)), 'constant')
        return Kernel(a_data - b_data)
    
    def __neg__(self) -> Kernel:
        return Kernel(-self.data)

@py_eval_fn
def kernel_conv(a: Kernel, b: Kernel) -> Kernel:
    return a * b

@py_eval_fn
def kernel_add(a: Kernel, b: Kernel) -> Kernel:
    return a + b

@py_eval_fn
def kernel_sub(a: Kernel, b: Kernel) -> Kernel:
    return a - b

@py_eval_fn
def dilate_kernel(k: Kernel) -> Tuple[Kernel, Kernel]:
    stride = 2
    while True:
        even = np.zeros(len(k), dtype=np.float64)
        even[0::stride] = k.data[0::stride]
        odd = k.data - even

        even = Kernel(even)
        odd = Kernel(odd)

        if len(odd) == 0:
            stride *= 2
            continue

        f = even - odd
        i = -even * even + 2 * even + odd * odd
        return f, i


_0 = py_eval_fn(lambda t: t[0])
_1 = py_eval_fn(lambda t: t[1])

class KernelExpr(Expr):
    @method(egg_fn="K1D")
    def __init__(self, coeffs: PyObject) -> None: ...
    
    @classmethod
    def z(cls, n: i64Like) -> KernelExpr: ...
    @classmethod
    def identity(cls) -> KernelExpr: ...

    def __mul__(self, other: KernelExpr) -> KernelExpr: ...
    def __add__(self, other: KernelExpr) -> KernelExpr: ...
    def __sub__(self, other: KernelExpr) -> KernelExpr: ...

class Signal(Expr):
    def __init__(self, name: StringLike) -> None: ...
    def convolve(self, kernel: KernelExpr) -> Signal: ...
    def recurse(self, kernel: KernelExpr) -> Signal: ...
    def __add__(self, other: Signal) -> Signal: ...


a, b = vars_("a b", PyObject)
tuple, = vars_("tuple", PyObject)
A, B, C = vars_("A B C", KernelExpr)
F, G, H = vars_("F G H", Signal)

# Generic properties
egraph.register(
    # A * (B * F) = (A * B) * F
    rewrite(
        F.convolve(B).convolve(A)
    ).to(
        F.convolve(A * B)
    ),
)

# Constant folding
egraph.register(
    rewrite(KernelExpr(a) * KernelExpr(b)).to(KernelExpr(kernel_conv(a, b))),
    rewrite(KernelExpr(a) + KernelExpr(b)).to(KernelExpr(kernel_add(a, b))),
    rewrite(KernelExpr(a) - KernelExpr(b)).to(KernelExpr(kernel_sub(a, b))),
)
# Arithmetic specific
egraph.register(
    # F / (1-B) / (1-A) = F / (1 - A - B + A*B)
    rewrite(
        F.recurse(B).recurse(A)
    ).to(
        F.recurse(A + B - A * B)
    ),

    rewrite(
        F.recurse(KernelExpr(a))
    ).to(
        F.convolve(KernelExpr(_0(tuple))).recurse(KernelExpr(_1(tuple))),
        eq(tuple).to(dilate_kernel(a))
    ),
)

n = egraph.let("n", KernelExpr(Kernel([0, 1, 0, 1])))
m = egraph.let("m", KernelExpr(Kernel([0, 1, 1])))
expr1 = egraph.let("expr1", Signal("input").convolve(n).convolve(m))
# expr2 = egraph.let("expr2", Signal("input").recurse(n).recurse(m))

expr2 = egraph.let("expr2", Signal("input").recurse(m))

egraph.run(3)
print(egraph.extract(expr1))
print(egraph.extract(expr2))
