from __future__ import annotations
from functools import reduce
from typing import List, Sequence, Tuple
import typing
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr, Type
from vecrec.expr.kernel import TIKernel2D
from vecrec.factorize import factorize_polynomial, factorize_kernel2d
from vecrec.util import ElementType
from . import Transform, ArithTransform, TropTransform

__all__ = [
    "FuseRecurse",
    "Dilate",
    "DilateTrop",
    "Delay",
    "ComposeRecurse",
    "Factorize",
    "Factorize2D",
    "DilateTVWithSingleOddOrder",
]


@ArithTransform
class FuseRecurse(Transform):
    """Fuse nested IIRs"""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, Recurse(b, g)):
                return [Recurse(KSub(KAdd(a, b), KConvolve(a, b)), g)]
            case _:
                return []


@ArithTransform
class Dilate(Transform):
    """Dilate an IIR"""

    max_bits: int

    def __init__(self, max_bits: int = 2**63):
        self.max_bits = max_bits

    def max_lanes(self, element_type: ElementType) -> int:
        """Get the maximal number of lanes possible given the type"""
        return self.max_bits // element_type.bit_width()

    def op(self, k: TIKernel) -> typing.Optional[Tuple[TIKernel, TIKernel]]:
        assert k.ty == Type.Arith
        max_stride = self.max_lanes(k.element_type)

        stride = 2
        while stride <= max_stride:
            even = [0.0] * len(k)
            even[0::stride] = k.data[0::stride]
            odd = [a - b for a, b in zip(k.data, even)]

            even_k = TIKernel(even, Type.Arith, k.element_type)
            odd_k = TIKernel(odd, Type.Arith, k.element_type)

            if len(odd_k) == 0:
                stride *= 2
                continue

            f = TIKernel.i(Type.Arith, k.element_type) - even_k + odd_k
            i = -even_k * even_k + 2 * even_k + odd_k * odd_k
            return f, i
        return None

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                result = self.op(a)
                if not result:
                    return []
                f, i = result
                return [Recurse(i, Convolve(f, g))]
            case _:
                return []


@TropTransform
class DilateTrop(Transform):
    """Dilate a tropical IIR"""

    max_bits: int

    def __init__(self, max_bits: int = 2**63):
        self.max_bits = max_bits
    def max_lanes(self, element_type: ElementType) -> int:
        """Get the maximal number of lanes possible given the type"""
        return self.max_bits // element_type.bit_width()

    def op(self, k: TIKernel) -> typing.Optional[Tuple[TIKernel, TIKernel]]:
        assert k.ty == Type.TropMax or k.ty == Type.TropMin
        max_stride = self.max_lanes(k.element_type)

        stride = 2
        while stride <= max_stride:
            even = [k.ty.zero()] * len(k)
            even[0::stride] = k.data[0::stride]
            odd = [k.ty.zero() if a == b else a for a, b in zip(k.data, even)]

            even_k = TIKernel(even, k.ty, k.element_type)
            odd_k = TIKernel(odd, k.ty, k.element_type)

            if len(odd_k) == 0:
                stride *= 2
                continue

            f = TIKernel.i(k.ty, k.element_type) + odd_k
            i = even_k + odd_k * odd_k
            return f, i
        return None

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                result = self.op(a)
                if not result:
                    return []
                f, i = result
                return [Recurse(i, Convolve(f, g))]
            case _:
                return []


class Delay(Transform):
    """Delay an IIR by identifying leading coefficient v at feedback p and negating by v*z^-p"""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                pos, val = next(
                    ((i, v) for i, v in enumerate(a.data) if not a.ty.is_zero(v)),
                    (len(a), 0),
                )
                if pos == len(a):
                    return []
                new_a = a.copy()
                new_a.data[pos] = expr.ty.zero()
                coeff = TIKernel.z(expr.ty, expr.element_type, -pos) * val
                return [
                    Recurse(
                        new_a + coeff * a,
                        Convolve(TIKernel.i(expr.ty, expr.element_type) + coeff, g),
                    )
                ]
            case _:
                return []


@ArithTransform
class ComposeRecurse(Transform):
    """Compose two IIRs R(a, g) + R(b, h)"""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            # Requires commutativity
            # G/(I-A) + H/(I-B) = (I-B)G/(I-B)(I-A) + (I-A)H/(I-A)(I-B) = ((I-B)G + (I-A)H) / (I - A)(I - B)
            case SAdd(Recurse(TIKernel(_) as a, g), Recurse(TIKernel(_) as b, h)):
                c = KSub(KAdd(a, b), KConvolve(a, b))
                w = SAdd(
                    Convolve(KSub(TIKernel.i(Type.Arith, a.element_type), b), g),
                    Convolve(KSub(TIKernel.i(Type.Arith, b.element_type), a), h),
                )
                return [Recurse(c, w)]
            case _:
                return []


@ArithTransform
class Factorize(Transform):
    """Factorize a TIKernel into products of first-order and second-order factors."""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Convolve(TIKernel(a) as kernel, g):
                if len(a) < 2:
                    return []  # Constant kernel, nothing to factorize
                factors: Sequence[KernelExpr] = factorize_polynomial(
                    a, kernel.element_type
                )
                if len(factors) == 0:
                    return []
                # e = reduce(lambda acc, factor: KConvolve(factor, acc), factors)
                e = reduce(lambda acc, factor: Convolve(factor, acc), factors, g)
                return [e]
            case _:
                return []


@ArithTransform
class Factorize2D(Transform):
    """Factorize a separable 2D TIKernel into row and column 1D kernels.

    If the coefficient matrix of TIKernel2D has rank 1, it can be written as
    an outer product: A[r][c] = col_vec[r] * row_vec[c].  In that case:
        Convolve2D(A, S) → Convolve(row_kernel, Convolve2D(col_kernel_2d, S))
    where col_kernel_2d is a single-column 2D kernel.  This splits a 2D
    convolution into two independent 1D passes, reducing complexity from
    O(rows * cols) to O(rows + cols) per output sample.
    """

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Convolve2D(TIKernel2D(_) as kernel, s):
                # Skip single-column kernels: already in factored form; factorizing
                # them again would create an infinite preorder-traversal loop.
                n_cols = max((len(row) for row in kernel.data), default=0)
                if n_cols <= 1:
                    return []
                result = factorize_kernel2d(kernel)
                if result is None:
                    return []
                row_kernel, col_kernel_2d = result
                # Convolve2D goes inside.
                return [Convolve(row_kernel, Convolve2D(col_kernel_2d, s))]
            case _:
                return []


@ArithTransform
class DilateTVWithSingleOddOrder(Transform):
    def all_nonzeros(self, a: Sequence[SignalExpr]) -> List[int]:
        inzs = []
        for i, v in enumerate(a):
            if not (isinstance(v, Num) and v.ty.is_zero(v.value)):
                inzs.append(i)
        return inzs

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(TVKernel(a) as kernel, g):
                even: List[SignalExpr] = [
                    Num(0, Type.Arith, kernel.element_type)
                ] * len(a)
                odd: List[SignalExpr] = [Num(0, Type.Arith, kernel.element_type)] * len(
                    a
                )
                even[0::2] = a[0::2]
                odd[1::2] = a[1::2]
                inzs = self.all_nonzeros(odd)
                if len(inzs) == 1:
                    inz = inzs[0]
                    A = TVKernel(odd, Type.Arith, kernel.element_type)
                    B = TVKernel(even, Type.Arith, kernel.element_type)
                    C = KConvolve(
                        TVKernel(
                            [Num(0, Type.Arith, kernel.element_type)] * (inz - 1)
                            + [
                                PointwiseDiv(
                                    odd[inz],
                                    Convolve(
                                        TIKernel.z(
                                            Type.Arith, kernel.element_type, -inz
                                        ),
                                        odd[inz],
                                    ),
                                )
                            ],
                            Type.Arith,
                            kernel.element_type,
                        ),
                        B,
                    )
                    exprs = [B, C, KConvolve(A, A), KNeg(KConvolve(C, B))]
                    expr = Recurse(
                        KAdd.of(exprs),
                        Convolve(
                            KAdd.of(
                                [
                                    TIKernel.i(Type.Arith, kernel.element_type),
                                    A,
                                    KNeg(C),
                                ]
                            ),
                            g,
                        ),
                    )
                    return [expr]
                elif len(inzs) == 0:
                    # TODO: dilate with stride greater than two
                    raise NotImplementedError
                else:
                    return []

            case _:
                return []
