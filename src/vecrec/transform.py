from __future__ import annotations
from functools import reduce
import itertools
import copy
from typing import List, Sequence, Tuple, overload
import typing
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr, Type
from vecrec.expr.base import KernelExpr2D, SignalExpr2D
from vecrec.expr.signal import RVar2D, Var2D
from vecrec.expr.signal_ops import Repeater, Convolve2D, Ith
from vecrec.expr.kernel import TIKernel2D, TVKernel2D
from vecrec.factorize import factorize_polynomial
from vecrec.util import ElementType


__all__ = [
    "Transform",
    "Noop",
    "ConstantFoldAdd",
    "ConstantFoldSub",
    "ConstantFoldConvolve",
    "ConstantFoldNegate",
    "ConstantFold",
    "FuseRecurse",
    "Dilate",
    "DilateTrop",
    "Delay",
    "ComposeRecurse",
    "Factorize",
    "DilateTVWithSingleOddOrder",
    "PushConvolveBelowRecurse",
    "Seq",
    "Repeat",
    "RepeatUpTo",
    "Optional",
    "Try",
    "Any",
    "Preorder",
    "SaturatePreorder",
    "SaturatePreorderUpTo",
    "Postorder",
    "AnnotateLanes",
    "PushDownConvertLanes",
    "Eliminate2DKernels",
    "DebugPrint",
]


class Transform:
    @overload
    def apply_generic(self, expr: KernelExpr) -> Sequence[KernelExpr]: ...
    @overload
    def apply_generic(self, expr: SignalExpr) -> Sequence[SignalExpr]: ...
    @overload
    def apply_generic(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]: ...
    @overload
    def apply_generic(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]: ...

    def apply_generic(self, expr):
        if isinstance(expr, KernelExpr):
            return self.apply_kernel(expr)
        elif isinstance(expr, SignalExpr):
            return self.apply_signal(expr)
        elif isinstance(expr, KernelExpr2D):
            return self.apply_kernel2d(expr)
        elif isinstance(expr, SignalExpr2D):
            return self.apply_signal2d(expr)
        else:
            raise TypeError(f"Unknown expression type: {type(expr)}")

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        return []

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        return []

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        return []

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        return []


class Noop(Transform):
    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        return [expr]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        return [expr]

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        return [expr]

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        return [expr]


# Constant folding
class ConstantFoldAdd(Transform):
    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        match expr:
            case KAdd(a, b) if isinstance(a, KernelConstant) and isinstance(
                b, KernelConstant
            ):
                return [a + b]
            case _:
                return []

    def apply_signal(self, expr):
        match expr:
            case SAdd(Num(a), b) | SAdd(b, Num(a)) if expr.ty.is_zero(a):
                return [b]
            case SSub(b, Num(a)) if expr.ty.is_zero(a):
                return [b]
            case _:
                return []


class ConstantFoldSub(Transform):
    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        match expr:
            case KSub(a, b) if isinstance(a, KernelConstant) and isinstance(
                b, KernelConstant
            ):
                return [a - b]
            case _:
                return []


class ConstantFoldConvolve(Transform):
    """Constant fold convolution of kernels."""

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        match expr:
            case KConvolve(a, b) if isinstance(a, KernelConstant) and isinstance(
                b, KernelConstant
            ):
                return [a * b]
            case _:
                return []

    def apply_signal(self, expr):
        match expr:
            case Convolve(TIKernel(a), b) if all(expr.ty.is_zero(v) for v in a):
                return [Num(0, expr.ty, expr.element_type)]
            case Convolve(TIKernel(a), b) if expr.ty.is_one(a[0]) and all(
                expr.ty.is_zero(v) for v in a[1:]
            ):
                return [b]
            case Convolve(TIKernel(a), b) if all(expr.ty.is_zero(v) for v in a):
                return [Num(0, expr.ty, expr.element_type)]
            case _:
                return []


def ArithTransform(cls):
    cls.apply_kernel_impl = cls.apply_kernel
    cls.apply_signal_impl = cls.apply_signal
    cls.apply_signal2d_impl = cls.apply_signal2d
    cls.apply_kernel2d_impl = cls.apply_kernel2d

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:  # type: ignore
        if expr.ty == Type.Arith:
            return self.apply_kernel_impl(expr)
        else:
            return []

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:  # type: ignore
        if expr.ty == Type.Arith:
            return self.apply_signal_impl(expr)
        else:
            return []

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:  # type: ignore
        if expr.ty == Type.Arith:
            return self.apply_signal2d_impl(expr)
        else:
            return []

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:  # type: ignore
        if expr.ty == Type.Arith:
            return self.apply_kernel2d_impl(expr)
        else:
            return []

    cls.apply_kernel = apply_kernel
    cls.apply_signal = apply_signal
    cls.apply_signal2d = apply_signal2d
    cls.apply_kernel2d = apply_kernel2d
    return cls


def TropTransform(cls):
    cls.apply_kernel_impl = cls.apply_kernel
    cls.apply_signal_impl = cls.apply_signal
    cls.apply_signal2d_impl = cls.apply_signal2d
    cls.apply_kernel2d_impl = cls.apply_kernel2d

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:  # type: ignore
        if expr.ty in (Type.TropMax, Type.TropMin):
            return self.apply_kernel_impl(expr)
        else:
            return []

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:  # type: ignore
        if expr.ty in (Type.TropMax, Type.TropMin):
            return self.apply_signal_impl(expr)
        else:
            return []

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:  # type: ignore
        if expr.ty in (Type.TropMax, Type.TropMin):
            return self.apply_signal2d_impl(expr)
        else:
            return []

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:  # type: ignore
        if expr.ty in (Type.TropMax, Type.TropMin):
            return self.apply_kernel2d_impl(expr)
        else:
            return []

    cls.apply_kernel = apply_kernel
    cls.apply_signal = apply_signal
    cls.apply_signal2d = apply_signal2d
    cls.apply_kernel2d = apply_kernel2d
    return cls


@ArithTransform
class ConstantFoldNegate(Transform):
    """Constant fold negation of time invariant kernels."""

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        match expr:
            case KNeg(a) if isinstance(a, TIKernel):
                return [-a]
            case _:
                return []


# IIRs


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

    def op(self, k: TIKernel) -> Tuple[TIKernel, TIKernel]:
        assert k.ty == Type.TropMax or k.ty == Type.TropMin
        stride = 2
        while True:
            even = [k.ty.zero()] * len(k)
            even[0::stride] = k.data[0::stride]
            print(even)
            print(k.data)
            odd = [k.ty.zero() if a == b else a for a, b in zip(k.data, even)]

            even_k = TIKernel(even, k.ty, k.element_type)
            odd_k = TIKernel(odd, k.ty, k.element_type)

            if len(odd_k) == 0:
                stride *= 2
                continue

            f = TIKernel.i(k.ty, k.element_type) + odd_k
            i = even_k + odd_k * odd_k
            return f, i

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                f, i = self.op(a)
                return [Recurse(i, Convolve(f, g))]
            case _:
                return []


# TODO: dilation for 2D IIRs
# TODO: delay for tropical IIRs


@ArithTransform
class Delay(Transform):
    """Delay an IIR by identifying leading coefficient v at feedback p and negating by v*z^-p"""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                assert a.ty == Type.Arith
                pos, val = next(
                    ((i, v) for i, v in enumerate(a.data) if not a.ty.is_zero(v)),
                    (len(a), 0),
                )
                if val == 0:
                    return []
                coeff = TIKernel.z(Type.Arith, expr.element_type, -pos) * val
                return [
                    Recurse(
                        a + coeff * (a - TIKernel.i(Type.Arith, expr.element_type)),
                        Convolve(TIKernel.i(Type.Arith, expr.element_type) + coeff, g),
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


## Time-varying kernels
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


class PushConvolveBelowRecurse(Transform):
    """Push Convolve below Recurse, since they commute.

    Convolve(h, Recurse(a, g)) = Recurse(a, Convolve(h, g))

    Both Convolve (FIR) and Recurse (IIR) are LTI operations, so their order
    can be swapped: h * (g / (I - a)) = (h * g) / (I - a).
    """

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Convolve(h, Recurse(a, g)):
                return [Recurse(a, Convolve(h, g))]
            case Convolve2D(h, Recurse2D(a, g)):
                # It seems both ways make sense.
                # Doing a recurse first allow convolution to reuse the materialization
                # Ask: Why do we want to push down convolution?
                raise NotImplementedError()
                # return [Recurse2D(a, Convolve2D(h, g))]
            case _:
                return []


def _dedup_exprs(results):
    if len(results) < 10:
        unique: list = []
        for expr in results:
            if not any(expr == u for u in unique):
                unique.append(expr)
    else:
        # ChatGPT told me this preserves order compared to `unique = list(set(results))`
        unique = list(dict.fromkeys(results))
    return unique


class Seq(Transform):
    def __init__(self, *transforms: Transform) -> None:
        self.transforms = transforms
        self.apply_kernel = self.apply_kernel2d = self.apply_signal = self.apply_signal2d = self.apply_generic  # type: ignore

    @typing.no_type_check
    def apply_generic(self, expr):
        results = [expr]
        for transform in self.transforms:
            results = _dedup_exprs(
                [next for res in results for next in transform.apply_generic(res)]
            )
        return results


class Repeat(Transform):
    def __init__(self, times: int, *transforms: Transform) -> None:
        self.transform = Seq(*transforms)
        self.times = times
        self.apply_kernel = self.apply_kernel2d = self.apply_signal = self.apply_signal2d = self.apply_generic  # type: ignore

    @typing.no_type_check
    def apply_generic(self, expr):
        results = [expr]
        for _ in range(self.times):
            results = _dedup_exprs(
                [next for res in results for next in self.transform.apply_generic(res)]
            )
        return results


class RepeatUpTo(Transform):
    def __init__(self, max_times: int, *transforms: Transform) -> None:
        self.transform = Seq(*transforms)
        self.max_times = max_times
        self.apply_kernel = self.apply_kernel2d = self.apply_signal = self.apply_signal2d = self.apply_generic  # type: ignore

    @typing.no_type_check
    def apply_generic(self, expr):
        results = [[expr]]
        for _ in range(self.max_times):
            new_results = []
            for res in results[-1]:
                new_results += self.transform.apply_generic(res)
            new_results = _dedup_exprs(new_results)
            results.append(new_results)
        return _dedup_exprs([s for res in results for s in res])


class Optional(Transform):
    """Apply a transform optionally, keeping the original expression if the transform doesn't produce it."""

    def __init__(self, transform: Transform) -> None:
        self.transform = transform
        self.apply_kernel = self.apply_kernel2d = self.apply_signal = self.apply_signal2d = self.apply_generic  # type: ignore

    def apply_generic(self, expr):
        results = self.transform.apply_generic(expr)
        # Check if the original expression is already in the results
        if any(expr == res for res in results):
            return results
        # Otherwise, prepend the original expression
        return [expr] + list(results)


class Try(Transform):
    def __init__(self, transform: Transform) -> None:
        self.transform = transform
        self.apply_kernel = self.apply_kernel2d = self.apply_signal = self.apply_signal2d = self.apply_generic  # type: ignore

    def apply_generic(self, expr):
        results = self.transform.apply_generic(expr)
        return results if len(results) > 0 else [expr]


class Any(Transform):
    def __init__(self, *transforms: Transform) -> None:
        self.transforms = transforms
        self.apply_kernel = self.apply_kernel2d = self.apply_signal = self.apply_signal2d = self.apply_generic  # type: ignore

    def apply_generic(self, expr):
        results = [
            next
            for transform in self.transforms
            for next in transform.apply_generic(expr)
        ]
        return _dedup_exprs(results)


ConstantFold = Any(
    ConstantFoldAdd(),
    ConstantFoldSub(),
    ConstantFoldConvolve(),
    ConstantFoldNegate(),
)


class Preorder(Transform):
    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        def cartesian(constructor, lists, lanes) -> Sequence[KernelExpr]:  # type: ignore
            return [
                constructor(*args).with_lanes(lanes)
                for args in itertools.product(*lists)
            ]

        results: List[KernelExpr] = []
        for expr in self.transform.apply_kernel(expr):
            match expr:
                case TIKernel(_) | TVKernel(_):
                    results.append(expr)
                case _:
                    results += cartesian(
                        type(expr),
                        [self.apply_generic(child) for child in expr.children()],
                        expr.lanes,
                    )
        return results

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        def cartesian(constructor, lists, lanes) -> Sequence[SignalExpr]:  # type: ignore
            return [
                constructor(*args).with_lanes(lanes)
                for args in itertools.product(*lists)
            ]

        results: List[SignalExpr] = []
        for expr in self.transform.apply_signal(expr):
            match expr:
                case Var(_) | Num(_) | Impulse(_):
                    results.append(expr)
                case Ith(_):
                    # Special handling for Ith to preserve the integer index
                    for new_a in self.apply_generic(expr.a):
                        new_expr = copy.copy(expr)
                        new_expr.a = new_a
                        results.append(new_expr)
                case _:
                    results += cartesian(
                        type(expr),
                        [self.apply_generic(child) for child in expr.children()],
                        expr.lanes,
                    )
        return results

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        def cartesian(constructor, lists, lanes) -> Sequence[KernelExpr2D]:  # type: ignore
            return [
                constructor(*args).with_lanes(lanes)
                for args in itertools.product(*lists)
            ]

        results: List[KernelExpr2D] = []
        for expr in self.transform.apply_generic(expr):
            match expr:
                case TIKernel2D(_) | TVKernel2D(_):
                    results.append(expr)
                case _:
                    results += cartesian(
                        type(expr),
                        [self.apply_generic(child) for child in expr.children()],
                        expr.lanes,
                    )
        return results

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        def cartesian(constructor, lists, lanes):
            return [
                constructor(*args).with_lanes(lanes)
                for args in itertools.product(*lists)
            ]

        results: List[SignalExpr2D] = []
        for expr in self.transform.apply_generic(expr):
            match expr:
                case RVar2D(_) | Var2D(_):
                    results.append(expr)
                case Repeater(_):
                    for new_a in self.apply_generic(expr.a):
                        new_expr = copy.copy(expr)
                        new_expr.a = new_a
                        results.append(new_expr)
                case Recurse2D(_, _):
                    results += cartesian(
                        type(expr),
                        [self.apply_generic(child) for child in expr.children()],
                        expr.lanes,
                    )
                case _:
                    raise NotImplementedError(
                        "Preorder traversal of non-RVar2D, non-Var2D, non-Repeater SignalExpr2D not implemented"
                    )
        return results


class Postorder(Transform):
    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        def cartesian(constructor, lists, lanes) -> Sequence[KernelExpr]:  # type: ignore
            return [
                constructor(*args).with_lanes(lanes)
                for args in itertools.product(*lists)
            ]

        results: Sequence[KernelExpr]
        match expr:
            case TIKernel(_) | TVKernel(_):
                results = [expr]
            case _:
                results = cartesian(
                    type(expr),
                    [self.apply_generic(child) for child in expr.children()],
                    expr.lanes,
                )
        return [e for result in results for e in self.transform.apply_kernel(result)]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        def cartesian(constructor, lists, lanes) -> Sequence[SignalExpr]:  # type: ignore
            return [
                constructor(*args).with_lanes(lanes)
                for args in itertools.product(*lists)
            ]

        results: Sequence[SignalExpr]
        match expr:
            case Var(_) | Num(_) | Impulse(_):
                results = [expr]
            case Ith(e, i):
                results = [
                    Ith(new_e, i).with_lanes(expr.lanes)
                    for new_e in self.apply_signal2d(e)
                ]
            case _:
                results = cartesian(
                    type(expr),
                    [self.apply_generic(child) for child in expr.children()],
                    expr.lanes,
                )
        return [e for result in results for e in self.transform.apply_signal(result)]

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        def cartesian(constructor, lists, lanes) -> Sequence[KernelExpr2D]:  # type: ignore
            return [
                constructor(*args).with_lanes(lanes)
                for args in itertools.product(*lists)
            ]

        results: Sequence[KernelExpr2D]
        match expr:
            case TIKernel2D(_) | TVKernel2D(_):
                results = [expr]
            case _:
                results = cartesian(
                    type(expr),
                    [self.apply_generic(child) for child in expr.children()],
                    expr.lanes,
                )
        return [e for result in results for e in self.transform.apply_kernel2d(result)]

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        def cartesian(constructor, lists, lanes):
            return [
                constructor(*args).with_lanes(lanes)
                for args in itertools.product(*lists)
            ]

        results: List[SignalExpr2D] = []
        match expr:
            case RVar2D(_) | Var2D(_):
                results = [expr]
            case Repeater(_):
                for new_a in self.apply_generic(expr.a):
                    new_expr = copy.copy(expr)
                    new_expr.a = new_a
                    results.append(new_expr)
            case Recurse2D(_, _):
                results = cartesian(
                    type(expr),
                    [self.apply_generic(child) for child in expr.children()],
                    expr.lanes,
                )
                return results
            case _:
                raise NotImplementedError(
                    "Postorder traversal of non-RVar2D, non-Var2D, non-Repeater SignalExpr2D not implemented"
                )
        return [e for result in results for e in self.transform.apply_signal2d(result)]


class SaturatePreorder(Transform):
    transform: Transform

    class TryWithTag(Transform):
        def __init__(self, transform: Transform) -> None:
            self.transform = transform
            self.success = False
            self.apply_kernel = self.apply_signal = self.apply_signal2d = self.apply_kernel2d = self.apply_generic  # type: ignore

        def apply_generic(self, expr):
            results = self.transform.apply_generic(expr)
            if len(results) > 0:
                self.success = True
                return results
            else:
                return [expr]

    def __init__(self, transform: Transform) -> None:
        self.transform = transform
        self.apply_kernel = self.apply_signal = self.apply_signal2d = self.apply_kernel2d = self.apply_generic  # type: ignore

    @typing.no_type_check
    def apply_generic(self, expr):
        worklist = [expr]
        results = []
        while worklist:
            expr = worklist.pop()
            try_ = self.TryWithTag(self.transform)
            rewritten = Preorder(try_).apply_generic(expr)
            if not try_.success:
                assert len(rewritten) == 1
                results.append(expr)
            else:
                worklist.extend(rewritten)
        return _dedup_exprs(results)


class SaturatePreorderUpTo(SaturatePreorder):
    @typing.no_type_check
    def apply_generic(self, expr):
        worklist = [expr]
        results = [expr]
        while worklist:
            expr = worklist.pop()
            try_ = self.TryWithTag(self.transform)
            rewritten = Preorder(try_).apply_generic(expr)
            if try_.success:
                results.extend(rewritten)
                worklist.extend(rewritten)
        return _dedup_exprs(results)


class Eliminate2DKernels(Transform):
    """Eliminate 2D kernels by converting them to 1D kernels and modifying the signal expressions accordingly.
    The only places where 2D kernels appear are in Convolve2D and Recurse2D.
    """

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        if isinstance(expr, Convolve2D):
            assert isinstance(expr.a, TIKernel2D) or isinstance(
                expr.a, TVKernel2D
            ), "Kernel should be canonicalized before 2D kernel elimination"
            if isinstance(expr.a, TIKernel2D):
                kernels: List[KernelExpr] = [
                    TIKernel(row, expr.a.ty, expr.a.element_type) for row in expr.a.data
                ]
            else:
                kernels = [
                    TVKernel(row, expr.a.ty, expr.a.element_type) for row in expr.a.data
                ]
            return [
                SAdd.of(
                    [
                        Convolve(kernel, Ith(expr.b, i))
                        for i, kernel in enumerate(kernels)
                    ]
                )
            ]
        else:
            return []

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        if isinstance(expr, Recurse2D):
            # Recurse2D(A, G) = Repeater(\f', Recurse(A[0], A[1:] x f' +  G))
            assert isinstance(expr.a, TIKernel2D) or isinstance(
                expr.a, TVKernel2D
            ), "Kernel should be canonicalized before 2D kernel elimination"

            kernel_first = (
                TIKernel(expr.a.data[0], expr.a.ty, expr.a.element_type)
                if isinstance(expr.a, TIKernel2D)
                else TVKernel(expr.a.data[0], expr.a.ty, expr.a.element_type)
            )
            kernel_rest = (
                TIKernel2D(expr.a.data[1:], expr.a.ty, expr.a.element_type)
                if isinstance(expr.a, TIKernel2D)
                else TVKernel2D(expr.a.data[1:], expr.a.ty, expr.a.element_type)
            )
            new_expr = Repeater(
                lambda f_: Recurse(
                    kernel_first, SAdd(Convolve2D(kernel_rest, f_), expr.g)
                ),
                len(expr.a.data),
                expr.a.ty,
                expr.a.element_type,
            )
            return [new_expr]
        else:
            return []


class AnnotateLanes(Transform):
    max_bits: int

    def __init__(self, max_bits: int) -> None:
        self.max_bits = max_bits

    def max_lanes(self, element_type: ElementType) -> int:
        """Get the maximal number of lanes possible given the type"""
        return self.max_bits // element_type.bit_width()

    @overload
    def convert_lanes(self, e: SignalExpr, lanes: int) -> SignalExpr: ...
    @overload
    def convert_lanes(self, e: KernelExpr, lanes: int) -> KernelExpr: ...
    @overload
    def convert_lanes(self, e: SignalExpr2D, lanes: int) -> SignalExpr2D: ...

    def convert_lanes(self, e, lanes):
        if isinstance(e, SignalExpr2D):
            # For 2D signal expressions, just ensure lanes are set correctly
            # TODO: ConvertLanes2D has not been implemented
            if e.lanes != lanes:
                return e.with_lanes(lanes)
            else:
                return e

        if e.lanes != lanes:
            new_expr = (
                ConvertLanes(e) if isinstance(e, SignalExpr) else KConvertLanes(e)
            )
            new_expr.lanes = lanes
            return new_expr
        else:
            return e

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        """Handle signal expressions that involve 2D data (Repeater, Convolve2D)."""
        element_type = expr.element_type
        lanes = self.max_lanes(element_type)

        match expr:
            case Repeater(a, _n_rows, prev_rows_var):
                new_repeater: Repeater = expr.with_lanes(lanes)  # type: ignore
                new_repeater.a = self.convert_lanes(self.apply_signal(a)[0], lanes)
                new_repeater.prev_rows_var = prev_rows_var.with_lanes(lanes)  # type: ignore
                return [new_repeater]
            case RVar2D(name):
                return [expr.with_lanes(lanes)]
            case Var2D(name, n_rows):
                return [expr.with_lanes(lanes)]
            case Recurse2D(a, g):
                raise NotImplementedError("Not implemented")
            case _:
                raise NotImplementedError(
                    "Lane annotation for non-Repeater, non-RVar2D, non-Var2D SignalExpr2D not implemented"
                )

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        raise NotImplementedError(
            "Lane annotation for 2D kernel expressions not implemented"
        )

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        assert not isinstance(
            expr, Convolve2D
        ), "2D convolutions should be eliminated before lane annotation"
        element_type = expr.element_type

        if isinstance(expr, Ith):
            lanes = self.max_lanes(element_type)
            new_a = self.convert_lanes(self.apply_generic(expr.a)[0], lanes)
            new_ith_expr = Ith(new_a, expr.i)
            new_ith_expr.lanes = lanes
            return [new_ith_expr]

        match expr:
            case Recurse(a, g):
                lanes, _ = a.time_delay(self.max_lanes(element_type))
            case _:
                lanes = self.max_lanes(element_type)

        if expr.is_leaf():
            return [expr.with_lanes(lanes)]
        else:
            args = []
            for child in expr.children():
                assert not isinstance(
                    child, KernelExpr2D
                ), "2D kernels should be eliminated before lane annotation"
                args.append(self.convert_lanes(self.apply_generic(child)[0], lanes))

            new_expr = type(expr)(*args)  # type: ignore
            new_expr.lanes = lanes

            return [new_expr]

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        element_type = expr.element_type
        max_lanes = self.max_lanes(element_type)
        if isinstance(expr, TVKernel):
            data = expr.data

            def convert_lanes(e):
                if e.lanes != max_lanes:
                    new_expr = ConvertLanes(e)
                    new_expr.lanes = max_lanes
                    return new_expr
                else:
                    return e

            new_data = [convert_lanes(self.apply_signal(sig)[0]) for sig in data]
            new_expr: KernelExpr = TVKernel(new_data, expr.ty, element_type)
            new_expr.lanes = max_lanes
            return [new_expr]
        elif expr.is_leaf():
            return [expr.with_lanes(max_lanes)]
        else:
            args = [self.apply_generic(child)[0] for child in expr.children()]
            for arg in args:
                assert arg.lanes == max_lanes
            new_expr = type(expr)(*args)  # type: ignore
            new_expr.lanes = max_lanes
            return [new_expr]


class PushDownConvertLanesImpl(Transform):
    # Try to make the expression compute that many lanes at a time.
    # (T, int) -> T for T in {SignalExpr, KernelExpr, SignalExpr2D, KernelExpr2D}
    def narrow_lanes(self, expr, lanes):
        assert not isinstance(
            expr, KernelExpr2D
        ), "2D kernels should be eliminated before pushing down lane annotations"
        assert expr.lanes is not None
        assert lanes <= expr.lanes

        if isinstance(expr, ConvertLanes) or isinstance(expr, KConvertLanes):
            if expr.a.lanes == lanes:
                return expr.a

            new_expr = copy.copy(expr)
            new_expr.lanes = lanes
            return new_expr
        elif isinstance(expr, KernelExpr):
            if isinstance(expr, TIKernel):
                new_kernel_expr: KernelExpr = TIKernel(
                    expr.data, expr.ty, expr.element_type
                )
                new_kernel_expr.lanes = lanes
                return new_kernel_expr
            elif isinstance(expr, TVKernel):
                # TODO: we can also use a ConvertLanes here
                data = [self.narrow_lanes(child, lanes) for child in expr.data]
                return TVKernel(data, expr.ty, expr.element_type).with_lanes(lanes)
            else:
                assert False, "unreachable"
        elif isinstance(expr, Var):
            new_signal_expr: SignalExpr = Var(expr.name, expr.ty, expr.element_type)
            new_signal_expr.lanes = lanes
            return new_signal_expr
        elif isinstance(expr, Impulse):
            new_signal_expr = Impulse(expr.value, expr.ty, expr.element_type)
            new_signal_expr.lanes = lanes
            return new_signal_expr
        elif isinstance(expr, Repeater):
            raise NotImplementedError(
                "TODO: This happens only after we implement ConvertLanes2D"
            )
        elif isinstance(expr, Ith):
            new_ith_expr = copy.copy(expr)
            new_ith_expr.lanes = lanes
            return new_ith_expr
        else:
            assert isinstance(expr, SignalExpr)
            if expr.is_leaf():
                new_signal_expr = copy.copy(expr)
            else:
                args = [self.narrow_lanes(child, lanes) for child in expr.children()]
                new_signal_expr = type(expr)(*args)

            new_signal_expr.lanes = lanes

            return new_signal_expr

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        if not isinstance(expr, KConvertLanes):
            return [expr]

        assert expr.lanes is not None and expr.a.lanes is not None
        if expr.lanes < expr.a.lanes:
            return [self.narrow_lanes(expr.a, expr.lanes)]

        return [expr]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        results: List[SignalExpr] = [expr]
        if not isinstance(expr, ConvertLanes):
            return results

        assert expr.lanes is not None and expr.a.lanes is not None
        if expr.lanes < expr.a.lanes:
            results.append(self.narrow_lanes(expr.a, expr.lanes))

        return results

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        raise NotImplementedError("2D kernel should have been eliminated")

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        # TODO: KConvertLanes2D has not been implemented.
        return [expr]


def PushDownConvertLanes() -> Transform:
    return Postorder(PushDownConvertLanesImpl())

class DebugPrint(Transform):
    def __init__(self) -> None:
        self.apply_kernel = self.apply_signal = self.apply_signal2d = self.apply_kernel2d = self.apply_generic  # type: ignore

    def apply_generic(self, expr):
        print(pp(expr))
        return [expr]
