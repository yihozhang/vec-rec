from __future__ import annotations
from typing import List, Sequence
import typing
import itertools
import copy
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr
from vecrec.expr.base import KernelExpr2D, SignalExpr2D
from vecrec.expr.signal import RVar2D, Var2D
from vecrec.expr.signal_ops import Repeater, Ith, Recurse2D
from vecrec.expr.signal_vec import VectorAt
from . import Transform, _dedup_exprs

__all__ = [
    "Noop",
    "Seq",
    "Repeat",
    "RepeatUpTo",
    "Optional",
    "Try",
    "Any",
    "Preorder",
    "Postorder",
    "SaturatePreorder",
    "SaturatePreorderUpTo",
]


class Noop(Transform):
    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        return [expr]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        return [expr]

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        return [expr]

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        return [expr]


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
                case TIKernel(_):
                    results.append(expr)
                case TVKernel(data):
                    results += [
                        TVKernel(data, expr.ty, expr.element_type).with_lanes(
                            expr.lanes
                        )
                        for data in itertools.product(
                            *[self.apply_signal(v) for v in data]
                        )
                    ]
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
                case VectorAt(a, i):
                    # Special handling for VectorAt to preserve the integer index
                    for new_vec_a in self.apply_generic(a):
                        results.append(VectorAt(new_vec_a, i).with_lanes(expr.lanes))
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
                case TIKernel2D(_):
                    results.append(expr)
                case TVKernel2D(data):
                    def expand_row(row):
                        return itertools.product(*(self.apply_signal(v) for v in row))

                    row_expansions = [expand_row(row) for row in data]

                    all_combinations = itertools.product(*row_expansions)

                    results += [
                        TVKernel2D([list(row) for row in d], expr.ty, expr.element_type).with_lanes(expr.lanes)
                        for d in all_combinations
                    ]
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
            case TIKernel(_):
                results = [expr]
            case TVKernel(data):
                results = [
                    TVKernel(data, expr.ty, expr.element_type).with_lanes(expr.lanes)
                    for data in itertools.product(*[self.apply_signal(v) for v in data])
                ]
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
            case VectorAt(a, i):
                results = [
                    VectorAt(new_vec_a, i).with_lanes(expr.lanes)
                    for new_vec_a in self.apply_generic(a)
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
            case TIKernel2D(_):
                results = [expr]
            case TVKernel2D(data):
                    def expand_row(row):
                        return itertools.product(*(self.apply_signal(v) for v in row))

                    row_expansions = [expand_row(row) for row in data]

                    all_combinations = itertools.product(*row_expansions)

                    results = [
                        TVKernel2D([list(row) for row in d], expr.ty, expr.element_type).with_lanes(expr.lanes)
                        for d in all_combinations
                    ]
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
