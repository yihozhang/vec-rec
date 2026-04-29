from __future__ import annotations
from typing import List, Sequence
import typing
import itertools
import copy
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr
from vecrec.expr.base import (
    KernelExpr2D,
    KernelMatrixExpr,
    SignalExpr2D,
    SignalVectorExpr,
    SignalExpr2DVec,
)
from vecrec.expr.signal import RVar2D, Var2D
from vecrec.expr.signal_ops import Repeater, Ith, Recurse2D, Signal2DAt, RepeaterVec, WindowMap, WMVar
from vecrec.expr.signal_vec import VectorAt
from . import Transform, _dedup_exprs
from typing import Callable


def _expand_tvkernel2d(
    apply_signal: Callable[[SignalExpr], Sequence[SignalExpr]],
    expr: TVKernel2D,
) -> List[KernelExpr2D]:
    """Return all TVKernel2D variants obtained by applying apply_signal to every entry."""
    return [
        TVKernel2D([list(row) for row in d], expr.ty, expr.element_type).with_lanes(expr.lanes)
        for d in itertools.product(*[itertools.product(*(apply_signal(v) for v in row)) for row in expr.data])
    ]


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
    @typing.no_type_check
    def apply_generic(self, expr):
        return [expr]

    apply_kernel = apply_generic
    apply_signal = apply_generic
    apply_signal2d = apply_generic
    apply_kernel2d = apply_generic
    apply_kernel_matrix = apply_generic
    apply_signal_vector = apply_generic
    apply_signal2dvec = apply_generic


class Seq(Transform):
    def __init__(self, *transforms: Transform) -> None:
        self.transforms = transforms

    @typing.no_type_check
    def apply_generic(self, expr):
        results = [expr]
        for transform in self.transforms:
            results = _dedup_exprs([next for res in results for next in transform.apply_generic(res)])
        return results

    apply_kernel = apply_generic
    apply_signal = apply_generic
    apply_signal2d = apply_generic
    apply_kernel2d = apply_generic
    apply_kernel_matrix = apply_generic
    apply_signal_vector = apply_generic
    apply_signal2dvec = apply_generic


class Repeat(Transform):
    def __init__(self, times: int, *transforms: Transform) -> None:
        self.transform = Seq(*transforms)
        self.times = times

    @typing.no_type_check
    def apply_generic(self, expr):
        results = [expr]
        for _ in range(self.times):
            results = _dedup_exprs([next for res in results for next in self.transform.apply_generic(res)])
        return results

    apply_kernel = apply_generic
    apply_signal = apply_generic
    apply_signal2d = apply_generic
    apply_kernel2d = apply_generic
    apply_kernel_matrix = apply_generic
    apply_signal_vector = apply_generic
    apply_signal2dvec = apply_generic


class RepeatUpTo(Transform):
    def __init__(self, max_times: int, *transforms: Transform) -> None:
        self.transform = Seq(*transforms)
        self.max_times = max_times

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

    apply_kernel = apply_generic
    apply_signal = apply_generic
    apply_signal2d = apply_generic
    apply_kernel2d = apply_generic
    apply_kernel_matrix = apply_generic
    apply_signal_vector = apply_generic
    apply_signal2dvec = apply_generic


class Optional(Transform):
    """Apply a transform optionally, keeping the original expression if the transform doesn't produce it."""

    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def apply_generic(self, expr):
        results = self.transform.apply_generic(expr)
        # Check if the original expression is already in the results
        if any(expr == res for res in results):
            return results
        # Otherwise, prepend the original expression
        return [expr] + list(results)

    apply_kernel = apply_generic
    apply_signal = apply_generic
    apply_signal2d = apply_generic
    apply_kernel2d = apply_generic
    apply_kernel_matrix = apply_generic
    apply_signal_vector = apply_generic
    apply_signal2dvec = apply_generic


class Try(Transform):
    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def apply_generic(self, expr):
        results = self.transform.apply_generic(expr)
        return results if len(results) > 0 else [expr]

    apply_kernel = apply_generic
    apply_signal = apply_generic
    apply_signal2d = apply_generic
    apply_kernel2d = apply_generic
    apply_kernel_matrix = apply_generic
    apply_signal_vector = apply_generic
    apply_signal2dvec = apply_generic


class Any(Transform):
    def __init__(self, *transforms: Transform) -> None:
        self.transforms = transforms

    def apply_generic(self, expr):
        results = [next for transform in self.transforms for next in transform.apply_generic(expr)]
        return _dedup_exprs(results)

    apply_kernel = apply_generic
    apply_signal = apply_generic
    apply_signal2d = apply_generic
    apply_kernel2d = apply_generic
    apply_kernel_matrix = apply_generic
    apply_signal_vector = apply_generic
    apply_signal2dvec = apply_generic


class Preorder(Transform):
    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        def cartesian(constructor, lists, lanes) -> Sequence[KernelExpr]:  # type: ignore
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

        results: List[KernelExpr] = []
        for expr in self.transform.apply_kernel(expr):
            match expr:
                case TIKernel(_):
                    results.append(expr)
                case TVKernel(data):
                    results += [
                        TVKernel(data, expr.ty, expr.element_type).with_lanes(expr.lanes)
                        for data in itertools.product(*[self.apply_signal(v) for v in data])
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
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

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
                case WMVar():
                    results.append(expr)
                case WindowMap():
                    new_bodies = list(self.apply_signal_vector(expr.body))
                    new_inputs_lists = [list(self.apply_generic(inp)) for inp in expr.inputs]
                    for new_inputs in itertools.product(*new_inputs_lists):
                        for new_body in new_bodies:
                            new_wm = WindowMap.make(expr.n, new_inputs, new_body, expr.input_vars)
                            new_wm.lanes = expr.lanes
                            results.append(new_wm)
                case _:
                    results += cartesian(
                        type(expr),
                        [self.apply_generic(child) for child in expr.children()],
                        expr.lanes,
                    )
        return results

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        def cartesian(constructor, lists, lanes) -> Sequence[KernelExpr2D]:  # type: ignore
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

        results: List[KernelExpr2D] = []
        for expr in self.transform.apply_generic(expr):
            match expr:
                case TIKernel2D(_):
                    results.append(expr)
                case TVKernel2D(_):
                    results += _expand_tvkernel2d(self.apply_signal, expr)
                case _:
                    results += cartesian(
                        type(expr),
                        [self.apply_generic(child) for child in expr.children()],
                        expr.lanes,
                    )
        return results

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        def cartesian(constructor, lists, lanes):
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

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
                case Signal2DAt(vec_expr, _comp_idx):
                    for new_vec in self.apply_signal2dvec(vec_expr):
                        new_at: Signal2DAt = copy.copy(expr)
                        new_at.a = new_vec
                        results.append(new_at)
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

    def apply_signal2dvec(self, expr: SignalExpr2DVec) -> Sequence[SignalExpr2DVec]:
        results: List[SignalExpr2DVec] = []
        for e in self.transform.apply_signal2dvec(expr):
            if isinstance(e, RepeaterVec):
                for new_a in self.apply_signal_vector(e.a):
                    new_rv = copy.copy(e)
                    new_rv.a = new_a
                    results.append(new_rv)
            else:
                results.append(e)
        return results

    def apply_kernel_matrix(self, expr: KernelMatrixExpr) -> Sequence[KernelMatrixExpr]:
        results: List[KernelMatrixExpr] = []
        for expr in self.transform.apply_generic(expr):
            match expr:
                case KernelMatrix(data):
                    results += [
                        KernelMatrix([list(row) for row in matrix_data]).with_lanes(expr.lanes)
                        for matrix_data in itertools.product(
                            *[itertools.product(*(self.apply_kernel(k) for k in row)) for row in data]
                        )
                    ]
                case _:
                    raise NotImplementedError()
        return results

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        def cartesian(constructor, lists, lanes) -> Sequence[SignalVectorExpr]:  # type: ignore
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

        results: List[SignalVectorExpr] = []
        for expr in self.transform.apply_generic(expr):
            match expr:
                case SignalVector(data):
                    results += [
                        SignalVector(vec_data).with_lanes(expr.lanes)
                        for vec_data in itertools.product(*[self.apply_signal(s) for s in data])
                    ]
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
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

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
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

        results: Sequence[SignalExpr]
        match expr:
            case Var(_) | Num(_) | Impulse(_):
                results = [expr]
            case Ith(e, i):
                results = [Ith(new_e, i).with_lanes(expr.lanes) for new_e in self.apply_signal2d(e)]
            case VectorAt(a, i):
                results = [VectorAt(new_vec_a, i).with_lanes(expr.lanes) for new_vec_a in self.apply_generic(a)]
            case WMVar():
                results = [expr]
            case WindowMap():
                new_bodies = list(self.apply_signal_vector(expr.body))
                new_inputs_lists = [list(self.apply_generic(inp)) for inp in expr.inputs]
                wm_results: List[SignalExpr] = []
                for new_inputs in itertools.product(*new_inputs_lists):
                    for new_body in new_bodies:
                        new_wm = WindowMap.make(expr.n, new_inputs, new_body, expr.input_vars)
                        new_wm.lanes = expr.lanes
                        wm_results.append(new_wm)
                results = wm_results
            case _:
                results = cartesian(
                    type(expr),
                    [self.apply_generic(child) for child in expr.children()],
                    expr.lanes,
                )
        return [e for result in results for e in self.transform.apply_signal(result)]

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        def cartesian(constructor, lists, lanes) -> Sequence[KernelExpr2D]:  # type: ignore
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

        results: Sequence[KernelExpr2D]
        match expr:
            case TIKernel2D(_):
                results = [expr]
            case TVKernel2D(_):
                results = _expand_tvkernel2d(self.apply_signal, expr)
            case _:
                results = cartesian(
                    type(expr),
                    [self.apply_generic(child) for child in expr.children()],
                    expr.lanes,
                )
        return [e for result in results for e in self.transform.apply_kernel2d(result)]

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        def cartesian(constructor, lists, lanes):
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

        results: List[SignalExpr2D] = []
        match expr:
            case RVar2D(_) | Var2D(_):
                results = [expr]
            case Repeater(_):
                for new_a in self.apply_generic(expr.a):
                    new_expr = copy.copy(expr)
                    new_expr.a = new_a
                    results.append(new_expr)
            case Signal2DAt(vec_expr, _comp_idx):
                for new_vec in self.apply_signal2dvec(vec_expr):
                    new_at: Signal2DAt = copy.copy(expr)
                    new_at.a = new_vec
                    results.append(new_at)
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

    def apply_signal2dvec(self, expr: SignalExpr2DVec) -> Sequence[SignalExpr2DVec]:
        results: List[SignalExpr2DVec]
        if isinstance(expr, RepeaterVec):
            body_results: List[SignalExpr2DVec] = []
            for new_a in self.apply_signal_vector(expr.a):
                new_rv = copy.copy(expr)
                new_rv.a = new_a
                body_results.append(new_rv)
            results = body_results
        else:
            results = [expr]
        return [e for result in results for e in self.transform.apply_signal2dvec(result)]

    def apply_kernel_matrix(self, expr: KernelMatrixExpr) -> Sequence[KernelMatrixExpr]:
        def cartesian(constructor, lists, lanes) -> Sequence[KernelMatrixExpr]:  # type: ignore
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

        results: Sequence[KernelMatrixExpr]
        match expr:
            case KernelMatrix(data):
                results = [
                    KernelMatrix([list(row) for row in matrix_data]).with_lanes(expr.lanes)
                    for matrix_data in itertools.product(
                        *[itertools.product(*(self.apply_kernel(k) for k in row)) for row in data]
                    )
                ]
            case _:
                raise NotImplementedError()
        return [e for result in results for e in self.transform.apply_kernel_matrix(result)]

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        def cartesian(constructor, lists, lanes) -> Sequence[SignalVectorExpr]:  # type: ignore
            return [constructor(*args).with_lanes(lanes) for args in itertools.product(*lists)]

        results: Sequence[SignalVectorExpr]
        match expr:
            case SignalVector(data):
                results = [
                    SignalVector(vec_data).with_lanes(expr.lanes)
                    for vec_data in itertools.product(*[self.apply_signal(s) for s in data])
                ]
            case _:
                results = cartesian(
                    type(expr),
                    [self.apply_generic(child) for child in expr.children()],
                    expr.lanes,
                )
        return [e for result in results for e in self.transform.apply_signal_vector(result)]


class SaturatePreorder(Transform):
    transform: Transform

    class TryWithTag(Transform):
        def __init__(self, transform: Transform) -> None:
            self.transform = transform
            self.success = False

        def apply_generic(self, expr):
            results = self.transform.apply_generic(expr)
            if len(results) > 0:
                self.success = True
                return results
            else:
                return [expr]

        apply_kernel = apply_generic
        apply_signal = apply_generic
        apply_signal2d = apply_generic
        apply_kernel2d = apply_generic
        apply_kernel_matrix = apply_generic
        apply_signal_vector = apply_generic
        apply_signal2dvec = apply_generic

    def __init__(self, transform: Transform) -> None:
        self.transform = transform

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

    apply_kernel = apply_generic
    apply_signal = apply_generic
    apply_signal2d = apply_generic
    apply_kernel2d = apply_generic
    apply_kernel_matrix = apply_generic
    apply_signal_vector = apply_generic
    apply_signal2dvec = apply_generic


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

    apply_kernel = apply_generic
    apply_signal = apply_generic
    apply_signal2d = apply_generic
    apply_kernel2d = apply_generic
    apply_kernel_matrix = apply_generic
    apply_signal_vector = apply_generic
    apply_signal2dvec = apply_generic
