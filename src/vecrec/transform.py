from __future__ import annotations
from functools import partial, reduce
import itertools
import copy
import numpy as np
from typing import Callable, List, Dict, Protocol, Sequence, Tuple, overload
from abc import abstractmethod
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr, Type
from vecrec.factorize import factorize_polynomial
from vecrec.util import ElementType


class Transform:
    @overload
    def apply_generic(self, expr: KernelExpr) -> Sequence[KernelExpr]: ...
    @overload
    def apply_generic(self, expr: SignalExpr) -> Sequence[SignalExpr]: ...

    def apply_generic(self, expr):
        if isinstance(expr, KernelExpr):
            return self.apply_kernel(expr)
        elif isinstance(expr, SignalExpr):
            return self.apply_signal(expr)
        else:
            raise TypeError(f"Unknown expression type: {type(expr)}")

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        return []

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        return []


class Noop(Transform):
    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        return [expr]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
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


def ArithTransform(cls):
    cls.apply_kernel_impl = cls.apply_kernel

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:  # type: ignore
        if expr.ty == Type.Arith:
            return self.apply_kernel_impl(expr)
        else:
            return []

    cls.apply_kernel = apply_kernel
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

    def op(self, k: TIKernel) -> Tuple[TIKernel, TIKernel]:
        assert k.ty == Type.Arith
        stride = 2
        while True:
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

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                f, i = self.op(a)
                return [Recurse(i, Convolve(f, g))]
            case _:
                return []


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
                factors: Sequence[KernelExpr] = factorize_polynomial(a, kernel.element_type)
                assert len(factors) > 0
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
                even: List[SignalExpr] = [Num(0, Type.Arith, kernel.element_type)] * len(a)
                odd: List[SignalExpr] = [Num(0, Type.Arith, kernel.element_type)] * len(a)
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
                                    Convolve(TIKernel.z(Type.Arith, kernel.element_type, -inz), odd[inz]),
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
                        Convolve(KAdd.of([TIKernel.i(Type.Arith, kernel.element_type), A, KNeg(C)]), g),
                    )
                    return [expr]
                elif len(inzs) == 0:
                    # TODO: dilate with stride greater than two
                    raise NotImplementedError
                else:
                    return []

            case _:
                return []


class Seq(Transform):
    def __init__(self, *transforms: Transform) -> None:
        self.transforms = transforms

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        results: List[KernelExpr] = [expr]
        for transform in self.transforms:
            results = [next for res in results for next in transform.apply_kernel(res)]
        return results

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        results: List[SignalExpr] = [expr]
        for transform in self.transforms:
            results = [next for res in results for next in transform.apply_signal(res)]
        return results


class Repeat(Transform):
    def __init__(self, times: int, *transforms: Transform) -> None:
        self.transform = Seq(*transforms)
        self.times = times

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        results: List[KernelExpr] = [expr]
        for _ in range(self.times):
            results = [
                next for res in results for next in self.transform.apply_kernel(res)
            ]
        return results

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        results: List[SignalExpr] = [expr]
        for _ in range(self.times):
            results = [
                next for res in results for next in self.transform.apply_signal(res)
            ]
        return results


class RepeatUpTo(Transform):
    def __init__(self, max_times: int, *transforms: Transform) -> None:
        self.transform = Seq(*transforms)
        self.max_times = max_times

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        results: List[List[KernelExpr]] = [[expr]]
        for _ in range(self.max_times):
            new_results: List[KernelExpr] = []
            for res in results[-1]:
                new_results += self.transform.apply_kernel(res)
            results.append(new_results)
        return [s for res in results for s in res]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        results: List[List[SignalExpr]] = [[expr]]
        for _ in range(self.max_times):
            new_results: List[SignalExpr] = []
            for res in results[-1]:
                new_results += self.transform.apply_signal(res)
            results.append(new_results)
        return [s for res in results for s in res]


def Optional(transform: Transform) -> Transform:
    return Any(Noop(), transform)


class Try(Transform):
    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        results = self.transform.apply_kernel(expr)
        return results if len(results) > 0 else [expr]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        results = self.transform.apply_signal(expr)
        return results if len(results) > 0 else [expr]


class Any(Transform):
    def __init__(self, *transforms: Transform) -> None:
        self.transforms = transforms

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        return [
            next
            for transform in self.transforms
            for next in transform.apply_kernel(expr)
        ]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        return [
            next
            for transform in self.transforms
            for next in transform.apply_signal(expr)
        ]


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
                case Var(_) | Num(_):
                    results.append(expr)
                case _:
                    results += cartesian(
                        type(expr),
                        [self.apply_generic(child) for child in expr.children()],
                        expr.lanes,
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
            case Var(_) | Num(_):
                results = [expr]
            case _:
                results = cartesian(
                    type(expr),
                    [self.apply_generic(child) for child in expr.children()],
                    expr.lanes,
                )
        return [e for result in results for e in self.transform.apply_signal(result)]


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

    def convert_lanes(self, e, lanes):
        if e.lanes != lanes:
            new_expr = (
                ConvertLanes(e) if isinstance(e, SignalExpr) else KConvertLanes(e)
            )
            new_expr.lanes = lanes
            return new_expr
        else:
            return e

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        element_type = expr.element_type
        match expr:
            case Recurse(a, g):
                lanes, _ = a.time_delay(self.max_lanes(element_type))
            case _:
                lanes = self.max_lanes(element_type)

        if expr.is_leaf():
            return [expr.with_lanes(lanes)]
        else:
            args = [
                self.convert_lanes(self.apply_generic(child)[0], lanes)
                for child in expr.children()
            ]
            new_expr = type(expr)(*args)
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
            new_expr = type(expr)(*args)
            new_expr.lanes = max_lanes
            return [new_expr]


class PushDownConvertLanesImpl(Transform):
    # Try to make the expression compute that many lanes at a time.
    @overload
    def narrow_lanes(self, expr: KernelExpr, lanes: int) -> KernelExpr: ...
    @overload
    def narrow_lanes(self, expr: SignalExpr, lanes: int) -> SignalExpr: ...

    def narrow_lanes(self, expr, lanes):
        assert expr.lanes is not None
        assert lanes <= expr.lanes

        if isinstance(expr, ConvertLanes) or isinstance(expr, ConvertLanes):
            if expr.a.lanes == lanes:
                return expr.a

            new_expr = copy.copy(expr)
            new_expr.lanes = lanes
            return new_expr
        elif isinstance(expr, KernelExpr):
            if isinstance(expr, TIKernel):
                new_kernel_expr: KernelExpr = TIKernel(expr.data, expr.ty, expr.element_type)
                new_kernel_expr.lanes = lanes
                return new_kernel_expr
            elif isinstance(expr, TVKernel):
                # There are two ways we can do it:
                # 1. Narrow down the lanes of each signal
                # 2. Use a ConvertLanes instead.
                raise NotImplementedError("Narrowing lanes of TVKernel not implemented")
            else:
                assert False, "unreachable"
        elif isinstance(expr, Var):
            new_signal_expr: SignalExpr = Var(expr.name, expr.ty, expr.element_type)
            new_signal_expr.lanes = lanes
            return new_signal_expr
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


def PushDownConvertLanes():
    return Postorder(PushDownConvertLanesImpl())


# class UnrollToMaxLanes(Transform):
#     def __init__(self, max_bits: int) -> None:
#         self.max_bits = max_bits

#     def max_lanes(self, element_type: ElementType) -> int:
#         """Get the maximal number of lanes possible given the type"""
#         return self.max_bits // element_type.bit_width()

#     def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
#         assert expr.lanes is not None
#         # TODO: ElementType.Float should be obtained from expr, not made up.
#         max_lanes = self.max_lanes(ElementType.Float)
#         assert expr.lanes <= max_lanes
#         if expr.lanes < max_lanes:
#             new_expr = ConvertLanes(expr)
#             new_expr.lanes = max_lanes
#             return [new_expr]
#         else:
#             return [expr]
