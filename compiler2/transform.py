from __future__ import annotations
from functools import reduce
import numpy as np
from typing import List, Optional, Dict, Tuple
from abc import abstractmethod
from compiler2.expr import *
from compiler2.factorize import factorize_polynomial
# from compiler2.expr import Add

class Transform:
    @abstractmethod
    def apply(self, expr: RecLang) -> List[RecLang]:
        pass

class ConstantFoldAdd(Transform):
    def op(self, a: TIKernel, b: TIKernel) -> TIKernel:
        max_len = max(len(a), len(b))
        a_data = np.pad(a.data, (0, max_len - len(a)), 'constant')
        b_data = np.pad(b.data, (0, max_len - len(b)), 'constant')
        return TIKernel(a_data + b_data)

    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Add(TIKernel(a), TIKernel(b)):
                return [self.add(TIKernel(a), TIKernel(b))]
            case _:
                return []

class ConstantFoldConvolve(Transform):
    def op(self, a: TIKernel, b: TIKernel) -> TIKernel:
        if len(a) == 0 or len(b) == 0:
            return TIKernel([])
        return TIKernel(np.convolve(a.data, b.data))

    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Convolve(TIKernel(a), TIKernel(b)):
                return [self.convolve(TIKernel(a), TIKernel(b))]
            case _:
                return []

class ConstantFoldNegate(Transform):
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Neg(TIKernel(a)):
                return [TIKernel(-a)]
            case _:
                return []

class FuseRecurse(Transform):
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Recurse(a, Recurse(b, g)):
                return [Recurse(Sub(Add(a, b), Convolve(a, b)), g)]
            case _:
                return []

class Dilate(Transform):
    def op(self, k: TIKernel) -> Tuple[TIKernel, TIKernel]:
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

    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Recurse(a, g):
                f, i = self.op(a)
                return [Recurse(f, Convolve(i, g))]
            case _:
                return []

class Delay(Transform):
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Recurse(a, g):
                return [Recurse(Convolve(a, a), Convolve(Add(TIKernel.i(), a), g))]
            case _:
                return []

class ComposeRecurse(Transform):
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case Add(Recurse(a, g), Recurse(b, h)):
                c = Sub(Add(a, b), Convolve(a, b))
                w = Add(Convolve(Sub(TIKernel.i(), b), g), Convolve(Sub(TIKernel.i(), a), h))
                return [Recurse(c, w)]
            case _:
                return []

class Factorize(Transform):
    def apply(self, expr: RecLang) -> List[RecLang]:
        match expr:
            case TIKernel(a):
                factors = factorize_polynomial(a)
                assert len(factors) > 0
                e = reduce(lambda acc, factor: Convolve(TIKernel(factor), acc), factors)
                return e
            case _:
                return []
