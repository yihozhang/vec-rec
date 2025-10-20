from __future__ import annotations
from functools import reduce
import numpy as np
from typing import List, Optional, Dict, Sequence, Tuple
from abc import abstractmethod
from compiler2.expr import *
from compiler2.factorize import factorize_polynomial
# from compiler2.expr import Add

class Transform:
    @abstractmethod
    def apply(self, expr: RecLang) -> List[RecLang]:
        pass

# Constant folding
class ConstantFoldAdd(Transform):
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case KAdd(a, b) if isinstance(a, KernelConstant) and isinstance(b, KernelConstant):
                return [a + b]
            case _:
                return []

class ConstantFoldConvolve(Transform):
    """Constant fold convolution of kernels."""
    
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case KConvolve(a, b) if isinstance(a, KernelConstant) and isinstance(b, KernelConstant):
                return [a * b]
            case _:
                return []

class ConstantFoldNegate(Transform):
    """Constant fold negation of time invariant kernels."""
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case KNeg(a) if isinstance(a, TIKernel):
                return [-a]
            case _:
                return []

# IIRs

class FuseRecurse(Transform):
    """Fuse nested IIRs"""
    
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Recurse(a, Recurse(b, g)):
                return [Recurse(KSub(KAdd(a, b), KConvolve(a, b)), g)]
            case _:
                return []

class Dilate(Transform):
    """Dilate an IIR"""

    def op(self, k: TIKernel) -> Tuple[TIKernel, TIKernel]:
        stride = 2
        while True:
            even = np.zeros(len(k), dtype=np.float64)
            even[0::stride] = k.data[0::stride]
            odd = k.data - even

            even_k = TIKernel(even)
            odd_k = TIKernel(odd)

            if len(odd_k) == 0:
                stride *= 2
                continue

            f = even_k - odd_k
            i = -even_k * even_k + 2 * even_k + odd_k * odd_k
            return f, i

    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                f, i = self.op(a)
                return [Recurse(f, Convolve(i, g))]
            case _:
                return []

class Delay(Transform):
    """Delay an IIR. One particular usage of this is time varying convolution"""
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Recurse(a, g):
                return [Recurse(KConvolve(a, a), Convolve(KAdd(TIKernel.i(), a), g))]
            case _:
                return []

class ComposeRecurse(Transform):
    """Compose two IIRs R(a, g) + R(b, h)"""
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case SAdd(Recurse(a, g), Recurse(b, h)):
                c = KSub(KAdd(a, b), KConvolve(a, b))
                w = SAdd(Convolve(KSub(TIKernel.i(), b), g), Convolve(KSub(TIKernel.i(), a), h))
                return [Recurse(c, w)]
            case _:
                return []

class Factorize(Transform):
    """Factorize a TIKernel into products of first-order and second-order factors."""

    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case TIKernel(a):
                factors: Sequence[KernelExpr] = factorize_polynomial(a)
                assert len(factors) > 0
                e = reduce(lambda acc, factor: KConvolve(factor, acc), factors)
                return [e]
            case _:
                return []

## Time-varying kernels

class DilateTVWithSingleOddOrder(Transform):
    def all_nonzeros(self, a: Sequence[SignalExpr]) -> List[int]:
        inzs = []
        for i, v in enumerate(a):
            if v != Num(0):
                inzs.append(i)
        return inzs

    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Recurse(TVKernel(a), g):
                even: List[SignalExpr] = [Num(0)] * len(a)
                odd: List[SignalExpr] = [Num(0)] * len(a)
                even[0::2] = a[0::2]
                odd[1::2] = a[1::2]
                inzs = self.all_nonzeros(odd)
                if len(inzs) == 1:
                    inz = inzs[0]
                    A = TVKernel(odd)
                    B = TVKernel(even)
                    C = KConvolve(
                        TVKernel([Num(0)] * (inz - 1) + [PointwiseDiv(odd[inz], Convolve(TIKernel.z(-inz), odd[inz]))]),
                        B
                    )
                    exprs = [B, C, KConvolve(A, A), KNeg(KConvolve(C, B))]
                    expr = Recurse(KAdd.of(exprs), Convolve(KAdd.of([TIKernel.i(), A, KNeg(C)]), g))
                    return [expr]
                elif len(inzs) == 0:
                    pass
                    return []
                else:
                    return []


            case _:
                return []

# TODO: test dilate single odd order
