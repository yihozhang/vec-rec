from __future__ import annotations
from typing import List, Optional, Sequence, overload, cast
import copy
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr
from vecrec.expr.base import KernelExpr2D, SignalExpr2D, KernelMatrixExpr, SignalVectorExpr, SignalExpr2DVec
from vecrec.expr.signal import RVar2D, Var2D, ComputedSignal2D
from vecrec.expr.signal_ops import Repeater, Convolve2D, Ith, Recurse2D, Signal2DAt, RepeaterVec, WindowMap, WindowMapVec, WMVar, ChangeSemiring, SelectMax
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
        all_var2d: list[Var2D | ComputedSignal2D] = []
        all_repeaters: list[Repeater] = []
        all_repeatervecs: list[RepeaterVec] = []
        rvar_to_repeater: dict[str, Repeater] = {}
        rvar_to_repeatervec: dict[str, RepeaterVec] = {}
        visited_phase1: set[int] = set()

        def collect_nodes(e: RecLang) -> None:
            if id(e) in visited_phase1:
                return
            visited_phase1.add(id(e))
            if isinstance(e, (Var2D, ComputedSignal2D)):
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
                if isinstance(x, (Var2D, ComputedSignal2D)):
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
        final_var2d: list[tuple[Var2D | ComputedSignal2D, int]] = [
            (v, max(v.n_rows or 0, var2d_bounds.get(v.name, 0))) for v in all_var2d
        ]
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
                    f"{type(v).__name__}('{v.name}') has no Ith usages — n_rows cannot be inferred. "
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
                # Probe to discover actual K (may be < max_lanes, e.g. Recurse with
                # delayed kernel or WindowMap whose RecurseV limits lanes).
                actual_lanes = (self.apply_signal(a)[0].lanes) or lanes
                # Re-annotate body at exactly K so all interior nodes (RVar2D etc.)
                # land at K lanes and match the Repeater's vec_type.
                k_ann = AnnotateLanes(actual_lanes * element_type.bit_width())
                new_a = k_ann.apply_signal(a)[0]
                new_repeater: Repeater = expr.with_lanes(actual_lanes)  # type: ignore
                new_repeater.a = new_a
                new_repeater.prev_rows_var = prev_rows_var.with_lanes(actual_lanes)  # type: ignore
                return [new_repeater]
            case RVar2D(_name):
                return [expr.with_lanes(lanes)]
            case Var2D(_name, _n_rows):
                return [expr.with_lanes(lanes)]
            case ComputedSignal2D(_name):
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
            ith_inner = self.apply_generic(expr.a)[0]
            actual_lanes = ith_inner.lanes or self.max_lanes(element_type)
            new_ith_expr = Ith(ith_inner, expr.i)
            new_ith_expr.lanes = actual_lanes
            return [new_ith_expr]

        if isinstance(expr, ChangeSemiring):
            target_lanes = self.max_lanes(element_type)
            new_inner = self.convert_lanes(self.apply_signal(expr.a)[0], target_lanes)
            new_cs = ChangeSemiring(new_inner, expr.ty)
            new_cs.lanes = target_lanes
            return [new_cs]

        if isinstance(expr, SelectMax):
            # Pointwise selection: annotate every candidate at max lanes and
            # rebuild explicitly so the integer labels are preserved (the
            # generic type(expr)(*args) path cannot pass labels).
            lanes = self.max_lanes(element_type)
            new_values = [self.convert_lanes(self.apply_signal(v)[0], lanes) for v in expr.values]
            new_sel = SelectMax(new_values, expr.labels)
            new_sel.lanes = lanes
            return [new_sel]

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


def _wrap_signal(expr: SignalExpr, target: int) -> SignalExpr:
    new = ConvertLanes(expr)
    new.lanes = target
    return new


def _wrap_vec(expr: SignalVectorExpr, target: int) -> SignalVectorExpr:
    new = ConvertLanesV(expr)
    new.lanes = target
    return new


def _count_adapters(expr: RecLang) -> int:
    count = 0
    stack: list[RecLang] = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, (ConvertLanes, KConvertLanes, ConvertLanesV)):
            count += 1
        for c in e.children():
            stack.append(c)
    return count


class PushDownConvertLanesImpl(Transform):
    """Variant-generating push-down of ConvertLanes / KConvertLanes / ConvertLanesV.

    For each adapter encountered (in postorder), emit two variants when push-down
    strictly reduces the adapter count: the original, and a relane'd variant where
    the adapter has been pushed through MUST=match operators and binding-variable
    nodes (Repeater/WindowMap/RepeaterVec/WindowMapVec, updating their bound
    variables in lockstep). Recurse-like nodes (Recurse, Recurse2D, RecurseV) are
    fixed barriers — relane wraps them in a fresh adapter when target differs.
    """

    def relane(self, expr: RecLang, target: int) -> RecLang:
        return self._relane(expr, target, {})

    def _external_2d_lanes(self, expr: SignalExpr2D, subst: dict[str, RecLang]) -> Optional[int]:
        """If `expr` is a 2D node bound by an external Repeater/RepeaterVec — i.e.
        an RVar2D not in `subst`, or a Signal2DAt rooted at a RepeaterVec whose
        prev_vars aren't in `subst` — return its native lanes (the binder's
        cache width, which can't be changed here). Otherwise return None.

        Example (affine-gap):

            μ(fv1_0, fv1_1, fv1_2).
                ( M,                                # comp 0, body at 8 lanes
                  I,                                # comp 1, body at 8 lanes
                  rec([-∞,-1], [-∞,-12] × M') )    # comp 2, Recurse forces 1 lane

        The Recurse on comp 2 has time-delay 1, so AnnotateLanes inserts a
        ConvertLanes(M'@8 → 1) around its g. PushDownConvertLanes then tries
        to push that adapter into M', which mentions fv1_0/fv1_1/fv1_2 via
        Ith(rvar, 0). Those RVar2D nodes are bound by the surrounding
        RepeaterVec — *not* by anything we're relane'ing — so their names are
        absent from `subst`. The cache they read is laid out at the outer
        RepeaterVec's 8 lanes; we can't reread it at 1 lane.

        For `Ith(rvar_fv1_0, 0)` with `subst = {}` this returns 8, signalling
        the caller (the `Ith` branch of `_relane`) to keep the Ith at 8 lanes
        and wrap the result in a 1D ConvertLanes(_ → 1) — analogous to how
        Recurse is treated as a fixed barrier. If instead we were inside the
        RepeaterVec's own `_relane` (which adds the freshly-relane'd prev_vars
        to `subst`), `fv1_0` would be in `subst` and this returns None, so
        the normal push-down proceeds."""
        if isinstance(expr, RVar2D):
            if expr.name not in subst:
                return expr.lanes
            return None
        if isinstance(expr, Signal2DAt):
            inner = expr.a
            if isinstance(inner, RepeaterVec) and all(pv.name not in subst for pv in inner.prev_vars):
                return expr.lanes
            return None
        return None

    def _external_vec_lanes(self, expr: SignalVectorExpr, subst: dict[str, RecLang]) -> Optional[int]:
        """SignalVector inside a RepeaterVec body is bound externally when none
        of the enclosing RepeaterVec's prev_vars are in `subst`. We don't have
        a direct handle on the binder here, but in practice this case only
        arises when VectorAt is read from a SignalVector that we shouldn't
        be reaching during relane — be conservative and treat all non-leaf
        SignalVectorExpr as relane'able."""
        return None

    def _relane(self, expr: RecLang, target: int, subst: dict[str, RecLang]) -> RecLang:
        # Fast path: subtree already at target lanes and not an adapter.
        if expr.lanes == target and not isinstance(expr, (ConvertLanes, KConvertLanes, ConvertLanesV)):
            return expr

        # SignalExpr branch
        if isinstance(expr, SignalExpr):
            if isinstance(expr, ConvertLanes):
                return self._relane(expr.a, target, subst)
            if isinstance(expr, ChangeSemiring):
                new_inner_cs = cast(SignalExpr, self._relane(expr.a, target, subst))
                new_cs = ChangeSemiring(new_inner_cs, expr.ty)
                new_cs.lanes = target
                return new_cs
            if isinstance(expr, Recurse):
                return expr if expr.lanes == target else _wrap_signal(expr, target)
            if isinstance(expr, SelectMax):
                new_vals = [cast(SignalExpr, self._relane(v, target, subst)) for v in expr.values]
                new_sel = SelectMax(new_vals, expr.labels)
                new_sel.lanes = target
                return new_sel
            if isinstance(expr, Convolve2D):
                raise AssertionError("Convolve2D should have been eliminated")
            if isinstance(expr, Ith):
                # An RVar2D / Signal2DAt rooted in a RepeaterVec whose binder
                # lies outside this subtree is a fixed barrier — its cache
                # layout was fixed at the binder's lanes and can't be reread
                # at a different width. Keep the Ith at the inner's native
                # lanes and wrap with ConvertLanes when target differs
                # (analogous to how Recurse is handled).
                natural_2d = self._external_2d_lanes(expr.a, subst)
                if natural_2d is not None:
                    new_ith = Ith(expr.a, expr.i)
                    new_ith.lanes = natural_2d
                    if natural_2d == target:
                        return new_ith
                    return _wrap_signal(new_ith, target)
                new_inner_2d = cast(SignalExpr2D, self._relane(expr.a, target, subst))
                new_ith = Ith(new_inner_2d, expr.i)
                new_ith.lanes = target
                return new_ith
            if isinstance(expr, VectorAt):
                natural_v = self._external_vec_lanes(expr.a, subst)
                if natural_v is not None:
                    new_vec_at = VectorAt(expr.a, expr.i)
                    new_vec_at.lanes = natural_v
                    if natural_v == target:
                        return new_vec_at
                    return _wrap_signal(new_vec_at, target)
                new_inner_v = cast(SignalVectorExpr, self._relane(expr.a, target, subst))
                new_vec_at = VectorAt(new_inner_v, expr.i)
                new_vec_at.lanes = target
                return new_vec_at
            if isinstance(expr, WindowMap):
                # Fixed barrier (like Recurse): a WindowMap's lane width K is
                # pinned by its body's RecurseV kernel, which carries exactly K
                # leading zeros so time_delay(K)=K. The WMVars are consumed behind
                # that RecurseV barrier — _relane wraps it rather than descending —
                # so relane'ing the WindowMap's binding vars to a new target would
                # leave the WMVar *uses* at the old lanes, corrupting codegen's
                # Unzip2/Zip2 slice accounting. Wrap in an adapter instead.
                return expr if expr.lanes == target else _wrap_signal(expr, target)
            if isinstance(expr, WMVar):
                if expr.name in subst:
                    return subst[expr.name]
                return WMVar(expr.name, expr.ty, expr.element_type).with_lanes(target)
            if isinstance(expr, Var):
                return Var(expr.name, expr.ty, expr.element_type).with_lanes(target)
            if isinstance(expr, Num):
                return Num(expr.value, expr.ty, expr.element_type).with_lanes(target)
            if isinstance(expr, Impulse):
                return Impulse(expr.value, expr.ty, expr.element_type).with_lanes(target)
            # MUST=match operators: SAdd, SSub, SNeg, PointwiseMul, PointwiseDiv, Convolve.
            new_children_s = [self._relane(c, target, subst) for c in expr.children()]
            new_expr: SignalExpr = type(expr)(*new_children_s)  # type: ignore[arg-type]
            new_expr.lanes = target
            return new_expr

        # KernelExpr branch
        if isinstance(expr, KernelExpr):
            if isinstance(expr, KConvertLanes):
                return self._relane(expr.a, target, subst)
            if isinstance(expr, TIKernel):
                return TIKernel(expr.data, expr.ty, expr.element_type).with_lanes(target)
            if isinstance(expr, TVKernel):
                new_data = [cast(SignalExpr, self._relane(s, target, subst)) for s in expr.data]
                return TVKernel(new_data, expr.ty, expr.element_type).with_lanes(target)
            # MUST=match: KAdd, KSub, KConvolve, KNeg.
            new_children_k = [self._relane(c, target, subst) for c in expr.children()]
            new_kernel: KernelExpr = type(expr)(*new_children_k)  # type: ignore[arg-type]
            new_kernel.lanes = target
            return new_kernel

        # SignalVectorExpr branch
        if isinstance(expr, SignalVectorExpr):
            if isinstance(expr, ConvertLanesV):
                return self._relane(expr.a, target, subst)
            if isinstance(expr, RecurseV):
                return expr if expr.lanes == target else _wrap_vec(expr, target)
            if isinstance(expr, WindowMapVec):
                # Fixed barrier for the same reason as WindowMap above: K is
                # pinned by the body's RecurseV kernel and the WMVars are consumed
                # behind that barrier, so the binding vars can't be relane'd.
                return expr if expr.lanes == target else _wrap_vec(expr, target)
            if isinstance(expr, SignalVector):
                new_data_sv = [cast(SignalExpr, self._relane(s, target, subst)) for s in expr.data]
                new_sv = SignalVector(new_data_sv)
                new_sv.lanes = target
                return new_sv
            if isinstance(expr, ConvolveV):
                new_a_cv = cast(KernelMatrixExpr, self._relane(expr.a, target, subst))
                new_g_cv = cast(SignalVectorExpr, self._relane(expr.g, target, subst))
                new_cv = ConvolveV(new_a_cv, new_g_cv)
                new_cv.lanes = target
                return new_cv
            raise AssertionError(f"relane unsupported for SignalVectorExpr: {type(expr)}")

        # KernelMatrixExpr branch
        if isinstance(expr, KernelMatrixExpr):
            if isinstance(expr, KernelMatrix):
                new_rows = [
                    [cast(KernelExpr, self._relane(k, target, subst)) for k in row] for row in expr.data
                ]
                return KernelMatrix(new_rows).with_lanes(target)
            raise AssertionError(f"relane unsupported for KernelMatrixExpr: {type(expr)}")

        # SignalExpr2D branch (reached via Ith / Signal2DAt)
        if isinstance(expr, SignalExpr2D):
            if isinstance(expr, RVar2D):
                if expr.name in subst:
                    return subst[expr.name]
                return RVar2D(expr.name, expr.ty, expr.element_type).with_lanes(target)
            if isinstance(expr, (Var2D, ComputedSignal2D)):
                new_var2d = copy.copy(expr)
                new_var2d.lanes = target
                return new_var2d
            if isinstance(expr, Repeater):
                old_var = expr.prev_rows_var
                new_var: RVar2D = cast(
                    RVar2D, RVar2D(old_var.name, old_var.ty, old_var.element_type).with_lanes(target)
                )
                body_subst_r: dict[str, RecLang] = dict(subst)
                body_subst_r[old_var.name] = new_var
                new_body_r = cast(SignalExpr, self._relane(expr.a, target, body_subst_r))
                rep = Repeater.make(new_body_r, expr.n_rows, new_var)
                rep.lanes = target
                return rep
            if isinstance(expr, Signal2DAt):
                new_inner_2dvec = cast(SignalExpr2DVec, self._relane(expr.a, target, subst))
                new_at = Signal2DAt(new_inner_2dvec, expr.comp_idx)
                new_at.lanes = target
                return new_at
            if isinstance(expr, Recurse2D):
                raise AssertionError("Recurse2D should have been eliminated")
            raise AssertionError(f"relane unsupported for SignalExpr2D: {type(expr)}")

        # SignalExpr2DVec branch (reached via Signal2DAt)
        if isinstance(expr, SignalExpr2DVec):
            if isinstance(expr, RepeaterVec):
                new_prev_vars: tuple[RVar2D, ...] = tuple(
                    cast(RVar2D, RVar2D(pv.name, pv.ty, pv.element_type).with_lanes(target))
                    for pv in expr.prev_vars
                )
                body_subst_rv: dict[str, RecLang] = dict(subst)
                for old_pv, new_pv in zip(expr.prev_vars, new_prev_vars):
                    body_subst_rv[old_pv.name] = new_pv
                new_body_rv = cast(SignalVectorExpr, self._relane(expr.a, target, body_subst_rv))
                rv = RepeaterVec.make(new_body_rv, expr.n, new_prev_vars, expr.n_rows)
                rv.lanes = target
                return rv
            raise AssertionError(f"relane unsupported for SignalExpr2DVec: {type(expr)}")

        if isinstance(expr, KernelExpr2D):
            raise AssertionError("2D kernels should have been eliminated")

        raise AssertionError(f"relane unknown type: {type(expr)}")

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        if not isinstance(expr, KConvertLanes):
            return [expr]
        assert expr.lanes is not None and expr.a.lanes is not None
        pushed = cast(KernelExpr, self.relane(expr.a, expr.lanes))
        if _count_adapters(pushed) < _count_adapters(expr):
            return [pushed, expr]
        return [expr]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        if not isinstance(expr, ConvertLanes):
            return [expr]
        assert expr.lanes is not None and expr.a.lanes is not None
        pushed = cast(SignalExpr, self.relane(expr.a, expr.lanes))
        if _count_adapters(pushed) < _count_adapters(expr):
            return [pushed, expr]
        return [expr]

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        if not isinstance(expr, ConvertLanesV):
            return [expr]
        assert expr.lanes is not None and expr.a.lanes is not None
        pushed = cast(SignalVectorExpr, self.relane(expr.a, expr.lanes))
        if _count_adapters(pushed) < _count_adapters(expr):
            return [pushed, expr]
        return [expr]

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        return [expr]

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        raise AssertionError("2D kernel should have been eliminated")

    def apply_signal2dvec(self, expr: SignalExpr2DVec) -> Sequence[SignalExpr2DVec]:
        return [expr]

    def apply_kernel_matrix(self, expr: KernelMatrixExpr) -> Sequence[KernelMatrixExpr]:
        return [expr]


def PushDownConvertLanes() -> Transform:
    return Postorder(PushDownConvertLanesImpl())
