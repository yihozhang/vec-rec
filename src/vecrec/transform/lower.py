from __future__ import annotations
from typing import List, Sequence, overload, cast
import copy
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr
from vecrec.expr.base import KernelExpr2D, SignalExpr2D, KernelMatrixExpr, SignalVectorExpr, SignalExpr2DVec
from vecrec.expr.signal import RVar2D, Var2D
from vecrec.expr.signal_ops import Repeater, Convolve2D, Ith, Recurse2D, Signal2DAt, RepeaterVec, WindowMap, WindowMapVec, WMVar
from vecrec.expr.kernel import TIKernel2D, TVKernel2D
from vecrec.expr.signal_vec import VectorAt, RecurseV, ConvolveV, SignalVector
from vecrec.expr.kernel_matrix import KernelMatrix
from vecrec.util import ElementType
from . import Transform
from .combinators import Postorder

__all__ = [
    "Eliminate2DKernels",
    "InferNRows",
    "AnnotateLanes",
    "PushDownConvertLanes",
]


class Eliminate2DKernels(Transform):
    """Eliminate 2D kernels by converting them to 1D kernels and modifying the signal expressions accordingly.
    The only places where 2D kernels appear are in Convolve2D and Recurse2D.
    """

    def __init__(self) -> None:
        self._memo: dict[int, list[SignalExpr2D]] = {}

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        if isinstance(expr, Convolve2D):
            assert isinstance(expr.a, TIKernel2D) or isinstance(
                expr.a, TVKernel2D
            ), "Kernel should be canonicalized before 2D kernel elimination"
            if isinstance(expr.a, TIKernel2D):
                kernels: List[KernelExpr] = [TIKernel(row, expr.a.ty, expr.a.element_type) for row in expr.a.data]
            else:
                kernels = [TVKernel(row, expr.a.ty, expr.a.element_type) for row in expr.a.data]
            return [SAdd.of([Convolve(kernel, Ith(expr.b, i)) for i, kernel in enumerate(kernels)])]
        else:
            return []

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        if isinstance(expr, Recurse2D):
            key = id(expr)
            if key in self._memo:
                return self._memo[key]

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
                lambda f_: Recurse(kernel_first, SAdd(Convolve2D(kernel_rest, f_), expr.g)),
                expr.a.ty,
                expr.a.element_type,
            )
            result: list[SignalExpr2D] = [new_expr]
            self._memo[key] = result
            return result
        else:
            return []


class InferNRows(Transform):
    """Infer n_rows for all Var2D and Repeater nodes from their Ith usages.

    Must be called after all passes that generate Ith nodes (Eliminate2DKernels,
    PushDownAndFuseConvolve) and before AnnotateLanes / codegen.
    """

    def apply_generic(self, expr):
        # Phase 1: DFS to collect all Var2D/Repeater/RepeaterVec nodes and build rvar maps.
        all_var2d: list[Var2D] = []
        all_repeaters: list[Repeater] = []
        all_repeatervecs: list[RepeaterVec] = []
        rvar_to_repeater: dict[str, Repeater] = {}
        rvar_to_repeatervec: dict[str, RepeaterVec] = {}
        visited_phase1: set[int] = set()

        def collect_nodes(e: RecLang) -> None:
            if id(e) in visited_phase1:
                return
            visited_phase1.add(id(e))
            if isinstance(e, Var2D):
                all_var2d.append(e)
            elif isinstance(e, Repeater):
                all_repeaters.append(e)
                rvar_to_repeater[e.prev_rows_var.name] = e
            elif isinstance(e, RepeaterVec):
                all_repeatervecs.append(e)
                for pv in e.prev_vars:
                    rvar_to_repeatervec[pv.name] = e
            for child in e.children():
                collect_nodes(child)

        collect_nodes(expr)

        # Phase 2: DFS to compute lower bounds for n_rows from Ith nodes.
        var2d_bounds: dict[str, int] = {}
        repeater_bounds: dict[Repeater, int] = {}
        repeatervec_bounds: dict[RepeaterVec, int] = {}
        visited_phase2: set[int] = set()

        def collect_bounds(e: RecLang) -> None:
            if id(e) in visited_phase2:
                return
            visited_phase2.add(id(e))
            if isinstance(e, Ith):
                x, i = e.a, e.i
                if isinstance(x, Var2D):
                    var2d_bounds[x.name] = max(var2d_bounds.get(x.name, 0), i + 1)
                elif isinstance(x, Repeater):
                    repeater_bounds[x] = max(repeater_bounds.get(x, 0), i + 1)
                elif isinstance(x, RVar2D):
                    owner = rvar_to_repeater.get(x.name)
                    if owner is not None:
                        # RVar2D exposes n_rows - 1 slots; Ith(rvar, i) needs owner.n_rows >= i + 2
                        repeater_bounds[owner] = max(repeater_bounds.get(owner, 0), i + 2)
                    else:
                        owner_vec = rvar_to_repeatervec.get(x.name)
                        assert owner_vec is not None, (
                            f"RVar2D '{x.name}' not found in any Repeater or RepeaterVec"
                        )
                        repeatervec_bounds[owner_vec] = max(repeatervec_bounds.get(owner_vec, 0), i + 2)
                elif isinstance(x, Signal2DAt):
                    rv = x.a
                    assert isinstance(rv, RepeaterVec)
                    repeatervec_bounds[rv] = max(repeatervec_bounds.get(rv, 0), i + 1)
            for child in e.children():
                collect_bounds(child)

        collect_bounds(expr)

        # Phase 3: Apply bounds, raise if any node remains unresolved.
        # Pre-compute ALL bounds before any mutation to avoid hash instability.
        final_var2d: list[tuple[Var2D, int]] = [(v, max(v.n_rows or 0, var2d_bounds.get(v.name, 0))) for v in all_var2d]
        final_repeaters: list[tuple[Repeater, int]] = [
            (r, max(r.n_rows or 0, repeater_bounds.get(r, 0))) for r in all_repeaters
        ]
        final_repeatervecs: list[tuple[RepeaterVec, int]] = [
            (rv, max(rv.n_rows or 0, repeatervec_bounds.get(rv, 0))) for rv in all_repeatervecs
        ]

        for v, bound in final_var2d:
            v.n_rows = bound or None
            if not v.n_rows:
                raise ValueError(
                    f"Var2D('{v.name}') has no Ith usages — n_rows cannot be inferred. "
                    "Call InferNRows after Eliminate2DKernels."
                )

        for r, bound in final_repeaters:
            r.n_rows = bound or None
            if not r.n_rows:
                raise ValueError(
                    f"Repeater with prev_rows_var '{r.prev_rows_var.name}' has no Ith usages — "
                    "n_rows cannot be inferred. Call InferNRows after Eliminate2DKernels."
                )

        for rv, bound in final_repeatervecs:
            rv.n_rows = bound or None
            if not rv.n_rows:
                raise ValueError(
                    f"RepeaterVec has no Ith usages — n_rows cannot be inferred."
                )

        return [expr]


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
    @overload
    def convert_lanes(self, e: KernelMatrixExpr, lanes: int) -> KernelMatrixExpr: ...
    @overload
    def convert_lanes(self, e: SignalVectorExpr, lanes: int) -> SignalVectorExpr: ...

    def convert_lanes(self, e, lanes):
        if isinstance(e, SignalExpr2D):
            # For 2D signal expressions, just ensure lanes are set correctly
            # TODO: ConvertLanes2D has not been implemented
            if e.lanes != lanes:
                return e.with_lanes(lanes)
            else:
                return e

        if isinstance(e, KernelMatrixExpr):
            raise NotImplementedError("Lane annotation for mutual recurrences not implemented")

        if e.lanes != lanes:
            new_expr = (
                ConvertLanes(e)
                if isinstance(e, SignalExpr)
                else (
                    KConvertLanes(e)
                    if isinstance(e, KernelExpr)
                    else ConvertLanesV(e) if isinstance(e, SignalVectorExpr) else None
                )
            )
            assert new_expr is not None, "Unsupported expression type for lane conversion"
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
            case RVar2D(_name):
                return [expr.with_lanes(lanes)]
            case Var2D(_name, _n_rows):
                return [expr.with_lanes(lanes)]
            case Signal2DAt(vec_expr, _comp_idx):
                new_vec = self.apply_signal2dvec(vec_expr)[0]
                new_at = copy.copy(expr)
                new_at.a = new_vec
                new_at.lanes = lanes
                return [new_at]
            case Recurse2D(_a, _g):
                raise NotImplementedError("Not implemented")
            case _:
                raise NotImplementedError(
                    "Lane annotation for non-Repeater, non-RVar2D, non-Var2D SignalExpr2D not implemented"
                )

    def apply_signal2dvec(self, expr: SignalExpr2DVec) -> Sequence[SignalExpr2DVec]:
        """Annotate lanes for RepeaterVec and its body."""
        if isinstance(expr, RepeaterVec):
            lanes = self.max_lanes(expr.element_type)
            new_a = self.convert_lanes(self.apply_signal_vector(expr.a)[0], lanes)
            new_prev_vars = cast(
                "tuple[RVar2D, ...]",
                tuple(pv.with_lanes(lanes) for pv in expr.prev_vars),
            )
            new_rv = copy.copy(expr)
            new_rv.a = new_a
            new_rv.prev_vars = new_prev_vars
            new_rv.lanes = lanes
            return [new_rv]
        return [expr.with_lanes(self.max_lanes(expr.element_type))]

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        raise NotImplementedError("Lane annotation for 2D kernel expressions not implemented")

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        assert not isinstance(expr, Convolve2D), "2D convolutions should be eliminated before lane annotation"
        element_type = expr.element_type

        if isinstance(expr, Ith):
            lanes = self.max_lanes(element_type)
            ith_inner = self.convert_lanes(self.apply_generic(expr.a)[0], lanes)
            new_ith_expr = Ith(ith_inner, expr.i)
            new_ith_expr.lanes = lanes
            return [new_ith_expr]

        if isinstance(expr, VectorAt) and isinstance(expr.a, WindowMapVec):
            wmv = expr.a
            # Discover K from the body (same pattern as WindowMap).
            probe = self.apply_signal_vector(wmv.body)[0]
            K = probe.lanes or self.max_lanes(element_type)
            k_ann = AnnotateLanes(K * element_type.bit_width())
            new_body = k_ann.apply_signal_vector(wmv.body)[0]
            new_inputs = tuple(k_ann.apply_signal(inp)[0] for inp in wmv.inputs)
            new_ivars = tuple(
                tuple(cast(WMVar, wv.with_lanes(K)) for wv in ivars)
                for ivars in wmv.input_vars
            )
            new_wmv = WindowMapVec.make(wmv.n_interleave, wmv.n, new_inputs, new_body, new_ivars)
            new_wmv.lanes = K
            new_vec_at = VectorAt(new_wmv, expr.i)
            new_vec_at.lanes = K
            return [new_vec_at]

        if isinstance(expr, VectorAt):
            lanes = self.max_lanes(element_type)
            vec_inner = self.convert_lanes(self.apply_generic(expr.a)[0], lanes)
            new_vec_at = VectorAt(vec_inner, expr.i)
            new_vec_at.lanes = lanes
            return [new_vec_at]

        if isinstance(expr, WindowMap):
            # Probe the body at max lanes to discover K from the RecurseV time_delay.
            probe = self.apply_signal_vector(expr.body)[0]
            K = probe.lanes or self.max_lanes(element_type)
            # Re-annotate body and inputs at exactly K lanes (no ConvertLanes wrappers
            # needed inside the body; everything stays at K lanes throughout).
            k_ann = AnnotateLanes(K * element_type.bit_width())
            new_body = k_ann.apply_signal_vector(expr.body)[0]
            new_inputs = tuple(k_ann.apply_signal(inp)[0] for inp in expr.inputs)
            # WMVar placeholders live in the body's half-rate domain (K lanes).
            new_ivars = tuple(
                tuple(cast(WMVar, wv.with_lanes(K)) for wv in ivars)
                for ivars in expr.input_vars
            )
            new_wm = WindowMap.make(expr.n, new_inputs, new_body, new_ivars)
            new_wm.lanes = K
            return [new_wm]

        match expr:
            case Recurse(a, _g):
                lanes, _ = a.time_delay(self.max_lanes(element_type))
            case _:
                lanes = self.max_lanes(element_type)

        if expr.is_leaf():
            return [expr.with_lanes(lanes)]
        else:
            args = []
            for child in expr.children():
                assert not isinstance(child, KernelExpr2D), "2D kernels should be eliminated before lane annotation"
                args.append(self.convert_lanes(self.apply_generic(child)[0], lanes))  # type: ignore[arg-type]

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

    def apply_kernel_matrix(self, expr: KernelMatrixExpr) -> Sequence[KernelMatrixExpr]:
        lanes = self.max_lanes(expr.element_type)

        if isinstance(expr, KernelMatrix):
            new_rows = []
            for row in expr.data:
                new_row = [self.convert_lanes(self.apply_kernel(k)[0], lanes) for k in row]
                new_rows.append(new_row)
            new_expr: KernelMatrixExpr = KernelMatrix(new_rows)
            new_expr.lanes = lanes
            return [new_expr]

        if expr.is_leaf():
            return [expr.with_lanes(lanes)]

        raise NotImplementedError("Lane annotation for non-KernelMatrix KernelMatrixExpr not implemented")

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        lanes = self.max_lanes(expr.element_type)

        if isinstance(expr, RecurseV | ConvolveV):
            assert isinstance(expr.a, KernelMatrix)
            if isinstance(expr, RecurseV):
                lanes = min(k.time_delay(lanes)[0] for row in expr.a.data for k in row)
            new_kernel_rows = [[self.convert_lanes(self.apply_kernel(k)[0], lanes) for k in row] for row in expr.a.data]
            new_km = KernelMatrix(new_kernel_rows).with_lanes(lanes)

            new_sv = self.convert_lanes(self.apply_signal_vector(expr.g)[0], lanes)

            new_expr: SignalVectorExpr = expr.__class__(new_km, new_sv).with_lanes(lanes)
            return [new_expr]
        elif isinstance(expr, SignalVector):
            new_signals = [self.convert_lanes(self.apply_signal(s)[0], lanes) for s in expr.data]
            new_sv = SignalVector(new_signals).with_lanes(lanes)
            return [new_sv]
        else:
            return [expr.with_lanes(lanes)]


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
                new_kernel_expr: KernelExpr = TIKernel(expr.data, expr.ty, expr.element_type)
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
            raise NotImplementedError("TODO: This happens only after we implement ConvertLanes2D")
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
        raise AssertionError("2D kernel should have been eliminated")

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        # TODO: KConvertLanes2D has not been implemented.
        return [expr]


def PushDownConvertLanes() -> Transform:
    return Postorder(PushDownConvertLanesImpl())
