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

def is_kernel(k: RecLang) -> bool:
    match k:
        case TIKernel(_):
            return True
        case TVKernel(_):
            return True
        case _:
            return False

class ConstantFoldAdd(Transform):
    """Constant fold addition of kernels."""
    def add_tv(self, a: TVKernel, b: TVKernel) -> TVKernel:
        max_len = max(len(a.data), len(b.data))
        data = []
        for i in range(max_len):
            data.append(Add(
                a.data[i] if i < len(a.data) else TIKernel([]),
                b.data[i] if i < len(b.data) else TIKernel([])
            ))
        return TVKernel(data)

    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Add(a, b) if isinstance(a, TIKernel) and isinstance(b, TIKernel):
                return [a + b]
            case Add(a, b) if isinstance(a, Kernel) and isinstance(b, Kernel):
                a_tv = a.promote() if isinstance(a, TIKernel) else a
                b_tv = b.promote() if isinstance(b, TIKernel) else b
                return [self.add_tv(a_tv, b_tv)]
            case _:
                return []

class ConstantFoldConvolve(Transform):
    """Constant fold convolution of kernels."""
    
    def convolve_ti(self, a: TIKernel, b: TIKernel) -> TIKernel:
        if len(a) == 0 or len(b) == 0:
            return TIKernel([])
        return TIKernel(np.convolve(a.data, b.data))

    def convolve_tv(self, a: TVKernel, b: TVKernel) -> TVKernel:
        max_len = len(a.data) + len(b.data) - 1
        data = []
        for i in range(max_len):
            terms = []
            for j in range(len(a.data)):
                if 0 <= i - j < len(b.data):
                    terms.append(PointwiseMul(a.data[j], Convolve(TIKernel.z(-j), b.data[i - j])))
            terms_seq: Sequence[RecLang] = terms
            if terms:
                term = reduce(lambda x, y: Add(x, y), terms_seq)
            else:
                term = TIKernel([])
            data.append(term)
        return TVKernel(data)

    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Convolve(a, b) if isinstance(a, TIKernel) and isinstance(b, TIKernel):
                return [a * b]
            case Convolve(a, b) if isinstance(a, Kernel) and isinstance(b, Kernel):
                a_tv = a.promote() if isinstance(a, TIKernel) else a
                b_tv = b.promote() if isinstance(b, TIKernel) else b
                return [self.convolve_tv(a_tv, b_tv)]
            case _:
                return []

class ConstantFoldNegate(Transform):
    def negate_ti(self, a: TIKernel) -> TIKernel:
        return TIKernel(-a.data)

    """Constant fold negation of time invariant kernels."""
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Neg(a) if isinstance(a, TIKernel):
                return [-a]
            case _:
                return []

class FuseRecurse(Transform):
    """Fuse nested IIRs"""
    
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Recurse(a, Recurse(b, g)):
                return [Recurse(Sub(Add(a, b), Convolve(a, b)), g)]
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
    """Delay an IIR"""
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Recurse(a, g):
                return [Recurse(Convolve(a, a), Convolve(Add(TIKernel.i(), a), g))]
            case _:
                return []

class ComposeRecurse(Transform):
    """Compose two IIRs R(a, g) + R(b, h)"""
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Add(Recurse(a, g), Recurse(b, h)):
                c = Sub(Add(a, b), Convolve(a, b))
                w = Add(Convolve(Sub(TIKernel.i(), b), g), Convolve(Sub(TIKernel.i(), a), h))
                return [Recurse(c, w)]
            case _:
                return []

class Factorize(Transform):
    """Factorize a TIKernel into products of first-order and second-order factors."""

    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case TIKernel(a):
                factors: Sequence[RecLang] = factorize_polynomial(a)
                assert len(factors) > 0
                e = reduce(lambda acc, factor: Convolve(factor, acc), factors)
                return [e]
            case _:
                return []

