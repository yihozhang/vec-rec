from __future__ import annotations
from functools import partial, reduce
import itertools
import numpy as np
from typing import Callable, List, Optional, Dict, Protocol, Sequence, Tuple, overload
from abc import abstractmethod
from compiler2.expr import *
from compiler2.factorize import factorize_polynomial

class Transform:
    @abstractmethod
    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        pass

# Constant folding
class ConstantFoldAdd(Transform):
    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        match expr:
            case KAdd(a, b) if isinstance(a, KernelConstant) and isinstance(b, KernelConstant):
                return [a + b]
            case _:
                return []

class ConstantFoldConvolve(Transform):
    """Constant fold convolution of kernels."""
    
    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        match expr:
            case KConvolve(a, b) if isinstance(a, KernelConstant) and isinstance(b, KernelConstant):
                return [a * b]
            case _:
                return []

class ConstantFoldNegate(Transform):
    """Constant fold negation of time invariant kernels."""
    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        match expr:
            case KNeg(a) if isinstance(a, TIKernel):
                return [-a]
            case _:
                return []

# IIRs

class FuseRecurse(Transform):
    """Fuse nested IIRs"""
    
    def apply(self, expr: RecLang) -> Sequence[RecLang]:
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

    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                f, i = self.op(a)
                return [Recurse(f, Convolve(i, g))]
            case _:
                return []

class Delay(Transform):
    """Delay an IIR. One particular usage of this is time varying convolution"""
    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        match expr:
            case Recurse(a, g):
                return [Recurse(KConvolve(a, a), Convolve(KAdd(TIKernel.i(), a), g))]
            case _:
                return []

class ComposeRecurse(Transform):
    """Compose two IIRs R(a, g) + R(b, h)"""
    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        match expr:
            # TODO: this transformation is wrong for time-varying kernels
            # TODO: look at other transformations
            case SAdd(Recurse(a, g), Recurse(b, h)):
                c = KSub(KAdd(a, b), KConvolve(a, b))
                w = SAdd(Convolve(KSub(TIKernel.i(), b), g), Convolve(KSub(TIKernel.i(), a), h))
                return [Recurse(c, w)]
            case _:
                return []

class Factorize(Transform):
    """Factorize a TIKernel into products of first-order and second-order factors."""

    def apply(self, expr: RecLang) -> Sequence[RecLang]:
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

    def apply(self, expr: RecLang) -> Sequence[RecLang]:
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
                    # TODO: dilate with stride greater than two
                    raise NotImplementedError
                else:
                    return []


            case _:
                return []

class ApplySequence(Transform):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = transforms

    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        results: List[RecLang] = [expr]
        for transform in self.transforms:
            results = [next for res in results for next in transform.apply(res)]
        return results

class Try(Transform):
    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        results = self.transform.apply(expr)
        return results if len(results) > 0 else [expr]

class ApplyParallel(Transform):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = transforms

    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        return [next for transform in self.transforms for next in transform.apply(expr)]

ConstantFold = ApplyParallel([
    ConstantFoldAdd(),
    ConstantFoldConvolve(),
    ConstantFoldNegate(),
])

class Preorder(Transform):
    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def apply(self, expr: RecLang) -> Sequence[RecLang]:
        def cartesian(constructor, lists: Sequence[Sequence[RecLang]]) -> Sequence[RecLang]:
            return [constructor(*args) for args in itertools.product(*lists)]
        
        results = []
        for expr in self.transform.apply(expr):
            match expr:
                case TIKernel(_) | TVKernel(_) | Var(_) | Num(_):
                    results.append(expr)
                case KAdd(a, b):
                    results += cartesian(KAdd, [self.apply(a), self.apply(b)])
                case KSub(a, b):
                    results += cartesian(KSub, [self.apply(a), self.apply(b)])
                case KNeg(a):
                    results += cartesian(KNeg, [self.apply(a)])
                case KConvolve(a, b):
                    results += cartesian(KConvolve, [self.apply(a), self.apply(b)])
                case SAdd(a, b):
                    results += cartesian(SAdd, [self.apply(a), self.apply(b)])
                case SSub(a, b):
                    results += cartesian(SSub, [self.apply(a), self.apply(b)])
                case PointwiseMul(a, b):
                    results += cartesian(PointwiseMul, [self.apply(a), self.apply(b)])
                case PointwiseDiv(a, b):
                    results += cartesian(PointwiseDiv, [self.apply(a), self.apply(b)])
                case SNeg(a):
                    results += cartesian(SNeg, [self.apply(a)])
                case Convolve(a, b):
                    results += cartesian(Convolve, [self.apply(a), self.apply(b)])
                case Recurse(a, g):
                    results += cartesian(Recurse, [self.apply(a), self.apply(g)])
                case _:
                    raise NotImplementedError(f"Preorder traversal not implemented for {expr}")
        return results


# @overload
# def preorder_traverse(func) -> Callable[SignalExpr, SignalExpr]:
#     ...

# @overload
# def preorder_traverse(func), expr: KernelExpr) -> KernelExpr:
#     ...

class ExprMapping(Protocol):
    @overload
    def __call__(self, expr: SignalExpr) -> SignalExpr: ...
    @overload
    def __call__(self, expr: KernelExpr) -> KernelExpr: ...

class ExprMappingAmb(Protocol):
    @overload
    def __call__(self, expr: SignalExpr) -> List[SignalExpr]: ...
    @overload
    def __call__(self, expr: KernelExpr) -> List[KernelExpr]: ...

def preorder_traverse(func) -> ExprMapping:
    @overload
    def go(expr: SignalExpr) -> SignalExpr: ...
    @overload
    def go(expr: KernelExpr) -> KernelExpr: ...

    def go(expr):
        expr = func(expr)
        match expr:
            case KAdd(a, b):
                return KAdd(go(a), go(b))
            case KSub(a, b):
                return KSub(go(a), go(b))
            case KConvolve(a, b):
                return KConvolve(go(a), go(b))
            case KNeg(a):
                return KNeg(go(a))
            case SAdd(a, b):
                return SAdd(go(a), go(b))
            case SSub(a, b):
                return SSub(go(a), go(b))
            case Convolve(a, b):
                return Convolve(go(a), go(b))
            case SNeg(a):
                return SNeg(go(a))
            case Recurse(a, g):
                return Recurse(go(a), go(g))
            case Convolve(a, b):
                return Convolve(go(a), go(b))
            case _:
                return expr
    return go