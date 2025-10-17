from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from egglog import *
import numpy as np

from expr import TIKernel
# from transform import *

egraph = EGraph()


@py_eval_fn
def kernel_conv(a: TIKernel, b: TIKernel) -> TIKernel:
    return a * b

@py_eval_fn
def kernel_add(a: TIKernel, b: TIKernel) -> TIKernel:
    return a + b

@py_eval_fn
def kernel_sub(a: TIKernel, b: TIKernel) -> TIKernel:
    return a - b

@py_eval_fn
def dilate_kernel(k: TIKernel) -> Tuple[TIKernel, TIKernel]:
    stride = 2
    while True:
        even = np.zeros(len(k), dtype=np.float64)
        even[0::stride] = k.data[0::stride]
        odd = k.data - even

        even = TIKernel(even)
        odd = TIKernel(odd)

        if len(odd) == 0:
            stride *= 2
            continue

        f = even - odd
        i = -even * even + 2 * even + odd * odd
        return f, i


_0 = py_eval_fn(lambda t: t[0])
_1 = py_eval_fn(lambda t: t[1])

class TIKernelExpr(Expr):
    @method(egg_fn="K1D")
    def __init__(self, coeffs: PyObject) -> None: ...
    
    @classmethod
    def z(cls, n: i64Like) -> TIKernelExpr: ...
    @classmethod
    def identity(cls) -> TIKernelExpr: ...

    def __mul__(self, other: TIKernelExpr) -> TIKernelExpr: ...
    def __add__(self, other: TIKernelExpr) -> TIKernelExpr: ...
    def __sub__(self, other: TIKernelExpr) -> TIKernelExpr: ...

class Signal(Expr):
    def __init__(self, name: StringLike) -> None: ...
    def convolve(self, kernel: TIKernelExpr) -> Signal: ...
    def recurse(self, kernel: TIKernelExpr) -> Signal: ...
    def __add__(self, other: Signal) -> Signal: ...


a, b = vars_("a b", PyObject)
tuple, = vars_("tuple", PyObject)
A, B, C = vars_("A B C", TIKernelExpr)
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
    rewrite(TIKernelExpr(a) * TIKernelExpr(b)).to(TIKernelExpr(kernel_conv(a, b))),
    rewrite(TIKernelExpr(a) + TIKernelExpr(b)).to(TIKernelExpr(kernel_add(a, b))),
    rewrite(TIKernelExpr(a) - TIKernelExpr(b)).to(TIKernelExpr(kernel_sub(a, b))),
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
        F.recurse(TIKernelExpr(a))
    ).to(
        F.convolve(TIKernelExpr(_0(tuple))).recurse(TIKernelExpr(_1(tuple))),
        eq(tuple).to(dilate_kernel(a))
    ),
)

n = egraph.let("n", TIKernelExpr(TIKernel([0, 1, 0, 1])))
m = egraph.let("m", TIKernelExpr(TIKernel([0, 1, 1])))
expr1 = egraph.let("expr1", Signal("input").convolve(n).convolve(m))
# expr2 = egraph.let("expr2", Signal("input").recurse(n).recurse(m))

expr2 = egraph.let("expr2", Signal("input").recurse(m))

egraph.run(3)
print(egraph.extract(expr1))
print(egraph.extract(expr2))
