from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import Callable, ClassVar, List, Optional, Sequence, Tuple

from vecrec.expr.signal import RVar2D, RVec2D
from vecrec.util import ElementType

from .base import *
from .base import _resolve_types, SignalExpr2DVec, Type


@dataclass(unsafe_hash=True)
class SignalExprBinOp(SignalExpr):
    a: SignalExpr
    b: SignalExpr
    __match_args__ = ("a", "b")

    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__(a.ty, a.element_type)
        assert a.ty == b.ty
        assert (
            a.element_type == b.element_type
        ), f"ElementType mismatch in SignalExprBinOp: {a.element_type} vs {b.element_type}"
        self.a = a
        self.b = b

    @classmethod
    def of(cls, exprs: List[SignalExpr]) -> SignalExpr:
        return reduce(lambda a, b: cls(a, b), exprs)


class SAdd(SignalExprBinOp):
    pass


class SSub(SignalExprBinOp):
    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__(a, b)
        assert self.ty.has_additive_inverse(), "Subtraction is only defined for types with an additive inverse"


class PointwiseMul(SignalExprBinOp):
    pass


class PointwiseDiv(SignalExprBinOp):
    def __init__(self, a: SignalExpr, b: SignalExpr) -> None:
        super().__init__(a, b)
        assert self.ty == Type.Arith, "Division is only defined for arithmetic type"


class CustomPointwiseOp(SignalExprBinOp):
    """Base class for a user-defined binary pointwise (elementwise) operator.

    Extends the built-in pointwise ops (SAdd, SSub, PointwiseMul, PointwiseDiv)
    with operators whose C++ implementation lives in a benchmark-owned struct
    rather than in the template library — the same self-describing pattern as
    ComputedSignal2D, but for a binary operation on two signals instead of a leaf
    input. Because it subclasses SignalExprBinOp it inherits the ``(a, b)``
    constructor and ``children() == [a, b]``, so it flows through every
    transform/schedule/lane-annotation pass exactly like a built-in binop; only
    codegen lowering differs.

    Subclass and set the two class attributes:

      cpp_struct:  name of a C++ struct template implementing the BinOp interface —
                   a default constructor, a
                   ``run(Reg_t &reg, int base, int s1, int s2)`` that writes
                   ``reg[base] = op(reg[s1], reg[s2])``, and a no-op
                   ``reset_and_next_row()``. Codegen instantiates it as
                   ``cpp_struct<vec_type>``.
      cpp_prelude: the full C++ definition of that struct, appended to the
                   generated header exactly once (before the kernels that use it).

    Unlike the built-in ops, no semiring suffix is appended to ``cpp_struct``: a
    custom pointwise op defines its own elementwise behavior directly and is
    semiring-independent. Operands must share ty/element_type (enforced by
    SignalExprBinOp); the op is only valid for element types its C++ body
    supports.
    """

    cpp_struct: ClassVar[str]
    cpp_prelude: ClassVar[str]


class PointwiseModulo(CustomPointwiseOp):
    """Elementwise modulo ``a % b`` — an example custom pointwise operator.

    Implemented purely as a benchmark-owned C++ struct using the vector ``%``
    operator, so it is valid for the integer element types (I8/I16/I32/I64).
    Division by zero is undefined, so the second operand must be nonzero.
    """

    cpp_struct = "PointwiseModulo"
    cpp_prelude = r"""
template <typename vec_type>
struct PointwiseModulo {
    PointwiseModulo() = default;

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int s1, int s2) {
        reg[base] = reg[s1] % reg[s2];
    }

    void reset_and_next_row() {}
};
"""


@dataclass(unsafe_hash=True)
class SNeg(SignalExpr):
    a: SignalExpr
    __match_args__ = ("a",)

    def __init__(self, a: SignalExpr) -> None:
        super().__init__(a.ty, a.element_type)
        assert a.ty.has_additive_inverse(), "Negation is only defined for types with an additive inverse"
        self.a = a


@dataclass(unsafe_hash=True)
class Convolve(SignalExpr):
    a: KernelExpr
    b: SignalExpr
    __match_args__ = ("a", "b")

    def __init__(self, a: KernelExpr, b: SignalExpr) -> None:
        super().__init__(a.ty, a.element_type)
        assert a.ty == b.ty
        assert (
            a.element_type == b.element_type
        ), f"ElementType mismatch in Convolve: {a.element_type} vs {b.element_type}"
        self.a = a
        self.b = b


@dataclass(unsafe_hash=True)
class SelectMax(SignalExpr):
    """Argmax selection over n candidate value signals.

    Reduces a fixed list of candidate value signals to the integer ``label`` of
    the candidate that achieves the (tropical) max at each position. Used to
    build a traceback / pointer stream alongside a max recurrence: the value
    recurrence computes the max; ``SelectMax`` computes which candidate won.

    Reduction semantics ("take the latest best, or take i"): scanning candidates
    left to right, the running best starts at candidate 0 (with ``labels[0]``).
    For each later candidate ``k`` with value ``v_k``::

        if v_k > best:  best = v_k; idx = labels[k]
        else:           keep best and idx

    The strict ``>`` makes earlier candidates win ties, giving a deterministic
    argmax. Only defined for idempotent-add (tropical) semirings.
    """

    values: Tuple[SignalExpr, ...]
    labels: Tuple[int, ...]
    __match_args__ = ("values", "labels")

    def __init__(self, values: Sequence[SignalExpr], labels: Sequence[int]) -> None:
        values_t = tuple(values)
        labels_t = tuple(labels)
        assert len(values_t) > 0, "SelectMax requires at least one candidate"
        assert len(values_t) == len(
            labels_t
        ), f"SelectMax: values/labels length mismatch: {len(values_t)} vs {len(labels_t)}"
        ty = values_t[0].ty
        element_type = values_t[0].element_type
        assert ty.is_add_idempotent(), "SelectMax is only defined for idempotent-add (tropical) semirings"
        assert all(v.ty == ty for v in values_t), "All SelectMax candidates must have the same type"
        assert all(
            v.element_type == element_type for v in values_t
        ), "All SelectMax candidates must have the same element_type"
        super().__init__(ty, element_type)
        self.values = values_t
        self.labels = labels_t

    def children(
        self,
    ) -> List[
        KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D | KernelMatrixExpr | SignalVectorExpr | SignalExpr2DVec
    ]:
        return list(self.values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SelectMax):
            return False
        return self.lanes == other.lanes and self.values == other.values and self.labels == other.labels

    def __repr__(self) -> str:
        return f"SelectMax({self.values}, {self.labels})"


@dataclass(unsafe_hash=True)
class Recurse(SignalExpr):
    a: KernelExpr
    g: SignalExpr
    __match_args__ = ("a", "g")

    def __init__(self, a: KernelExpr, g: SignalExpr) -> None:
        super().__init__(a.ty, a.element_type)
        assert a.ty == g.ty
        assert (
            a.element_type == g.element_type
        ), f"ElementType mismatch in Recurse: {a.element_type} vs {g.element_type}"
        self.a = a
        self.g = g


# Convert Lanes
@dataclass(unsafe_hash=True)
class ConvertLanes(SignalExpr):
    a: SignalExpr
    __match_args__ = ("a",)

    def __init__(self, a: SignalExpr) -> None:
        super().__init__(a.ty, a.element_type)
        self.a = a
        self.lanes = None

    def __repr__(self) -> str:
        return f"ConvertLanes({self.lanes}, {self.a})"


@dataclass(unsafe_hash=True)
class ChangeSemiring(SignalExpr):
    """Re-interpret a signal as belonging to a different semiring.

    The runtime values are unchanged (the underlying element_type is identical);
    only the semiring view changes. This allows an expression to mix operators
    from multiple semirings — e.g. computing a difference in the arithmetic
    semiring (which has a real additive inverse) and feeding the result into a
    recurrence in a tropical or custom semiring.

    The outer ``ty`` is the target semiring; ``a.ty`` is the source semiring.
    """

    a: SignalExpr
    __match_args__ = ("a",)

    def __init__(self, a: SignalExpr, target_ty: Type) -> None:
        super().__init__(target_ty, a.element_type)
        self.a = a


@dataclass(unsafe_hash=True)
class Convolve2D(SignalExpr):
    """2D convolution: convolve a 2D kernel with a 2D signal (from Repeater). Produces a 1D signal"""

    a: KernelExpr2D
    b: SignalExpr2D
    __match_args__ = ("a", "b")

    def __init__(self, a: KernelExpr2D, b: SignalExpr2D) -> None:
        super().__init__(a.ty, a.element_type)
        assert a.ty == b.ty
        assert (
            a.element_type == b.element_type
        ), f"ElementType mismatch in Convolve2D: {a.element_type} vs {b.element_type}"
        self.a = a
        self.b = b


@dataclass(unsafe_hash=True)
class Recurse2D(SignalExpr2D):
    """2D version of Recurse. The callable receives a RVar2D representing the output of the previous rows."""

    a: KernelExpr2D
    g: SignalExpr
    __match_args__ = ("a", "g")

    def __init__(self, a: KernelExpr2D, g: SignalExpr) -> None:
        super().__init__(a.ty, a.element_type)
        assert a.ty == g.ty
        assert (
            a.element_type == g.element_type
        ), f"ElementType mismatch in Recurse: {a.element_type} vs {g.element_type}"
        self.a = a
        self.g = g


@dataclass(unsafe_hash=True)
class Repeater(SignalExpr2D):
    """
    Turns a 1D signal into a 2D signal by caching the last `n_rows` rows.
    The minimal number of rows to cache is 2, which stores the previous row
    while computing and storing the current row.

    The callable receives a RVar2D representing the previous rows stored in
    the Repeater, with the current horizontal position aligned with the
    stream for the current row. Use Convolve2D with a 2D kernel to
    access data from multiple previous rows.

    Streams as a 2D signal: run() returns n vectors at a time (current row
    plus the n-1 previously cached rows).
    """

    a: SignalExpr
    n_rows: Optional[int]
    prev_rows_var: RVar2D
    __match_args__ = ("a", "n_rows", "prev_rows_var")

    var_count: int = 1

    def __init__(
        self,
        func: Callable[[RVar2D], SignalExpr],
        ty: Optional[Type] = None,
        element_type: Optional[ElementType] = None,
    ) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        self.n_rows = None
        self.prev_rows_var = RVar2D(f"$f{Repeater.var_count}", resolved_ty, resolved_et)
        Repeater.var_count += 1
        self.a = func(self.prev_rows_var)
        super().__init__(self.a.ty, self.a.element_type)

    @staticmethod
    def make(a: SignalExpr, n_rows: Optional[int], prev_rows_var: RVar2D) -> Repeater:
        repeater = Repeater.__new__(Repeater)
        super(Repeater, repeater).__init__(a.ty, a.element_type)
        repeater.a = a
        repeater.n_rows = n_rows
        repeater.prev_rows_var = prev_rows_var
        return repeater

    def children(
        self,
    ) -> List[
        KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D | KernelMatrixExpr | SignalVectorExpr | SignalExpr2DVec
    ]:
        # Only expose the inner expression, not prev_rows_var
        # (it's an internal placeholder, not an external input)
        return [self.a]


@dataclass(unsafe_hash=True)
class Ith(SignalExpr):
    a: SignalExpr2D
    i: int
    __match_args__ = ("a", "i")

    def __init__(self, a: SignalExpr2D, i: int) -> None:
        super().__init__(a.ty, a.element_type)
        self.a = a
        self.i = i


@dataclass(unsafe_hash=True)
class Signal2DAt(SignalExpr2D):
    """Extract component comp_idx from a SignalExpr2DVec as a SignalExpr2D."""

    a: SignalExpr2DVec
    comp_idx: int
    __match_args__ = ("a", "comp_idx")

    def __init__(self, a: SignalExpr2DVec, comp_idx: int) -> None:
        super().__init__(a.ty, a.element_type)
        self.a = a
        self.comp_idx = comp_idx

    def children(
        self,
    ) -> List[
        KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D | KernelMatrixExpr | SignalVectorExpr | SignalExpr2DVec
    ]:
        return [self.a]


@dataclass(unsafe_hash=True)
class RepeaterVec(SignalExpr2DVec):
    """
    Multi-component Repeater: caches n signals from the previous row simultaneously.

    The callable receives an RVec2D whose n components (RVar2D) represent the
    previous cached rows for each component.  The callable must return a
    SignalVectorExpr of the same n.  Use Signal2DAt / Ith to read the cached rows.
    """

    a: SignalVectorExpr
    n_rows: Optional[int]
    prev_vars: Tuple[RVar2D, ...]
    __match_args__ = ("a", "n_rows", "prev_vars")

    var_count: int = 1

    def __init__(
        self,
        n: int,
        func: Callable[[RVec2D], SignalVectorExpr],
        ty: Optional[Type] = None,
        element_type: Optional[ElementType] = None,
    ) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        self.n_rows = None
        self.prev_vars = tuple(RVar2D(f"$fv{RepeaterVec.var_count}_{i}", resolved_ty, resolved_et) for i in range(n))
        RepeaterVec.var_count += 1
        self.a = func(RVec2D(list(self.prev_vars)))
        assert self.a.n == n, f"RepeaterVec body returned {self.a.n} components, expected {n}"
        super().__init__(resolved_ty, resolved_et, n)

    @staticmethod
    def make(
        a: SignalVectorExpr,
        n: int,
        prev_vars: Tuple[RVar2D, ...],
        n_rows: Optional[int],
    ) -> RepeaterVec:
        rv: RepeaterVec = RepeaterVec.__new__(RepeaterVec)
        super(RepeaterVec, rv).__init__(a.ty, a.element_type, n)
        rv.a = a
        rv.n = n
        rv.n_rows = n_rows
        rv.prev_vars = prev_vars
        return rv

    def children(
        self,
    ) -> List[
        KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D | KernelMatrixExpr | SignalVectorExpr | SignalExpr2DVec
    ]:
        # Only expose the body, not prev_vars (internal placeholders)
        return [self.a]


@dataclass(unsafe_hash=True)
class WMVar(SignalExpr):
    """Placeholder for one interleaved substream inside a WindowMap body."""

    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Type, element_type: ElementType) -> None:
        super().__init__(ty, element_type)
        self.name = name


@dataclass(unsafe_hash=True)
class WindowMap(SignalExpr):
    """Partition inputs into n interleaved substreams, apply body, zip result back.

    For n=2: substream j=0 = even positions, j=1 = odd positions.
    input_vars[i][j] is the WMVar placeholder for the j-th substream of the i-th input.
    body is a SignalVectorExpr with n components (one per substream).
    """

    n: int
    inputs: Tuple[SignalExpr, ...]
    body: SignalVectorExpr
    input_vars: Tuple[Tuple[WMVar, ...], ...]
    __match_args__ = ("n", "inputs", "body", "input_vars")

    var_count: int = 1

    def __init__(
        self,
        n: int,
        inputs: List[SignalExpr],
        func: Callable[[List[List[WMVar]]], SignalVectorExpr],
    ) -> None:
        assert len(inputs) > 0
        ty = inputs[0].ty
        et = inputs[0].element_type
        name = f"$wm{WindowMap.var_count}"
        WindowMap.var_count += 1
        self.n = n
        self.inputs = tuple(inputs)
        self.input_vars = tuple(tuple(WMVar(f"{name}_{i}_{j}", ty, et) for j in range(n)) for i in range(len(inputs)))
        subs = [list(self.input_vars[i]) for i in range(len(inputs))]
        self.body = func(subs)
        assert self.body.n == n
        super().__init__(ty, et)

    @staticmethod
    def make(
        n: int,
        inputs: Tuple[SignalExpr, ...],
        body: SignalVectorExpr,
        input_vars: Tuple[Tuple[WMVar, ...], ...],
    ) -> "WindowMap":
        wm: WindowMap = WindowMap.__new__(WindowMap)
        ty = inputs[0].ty
        et = inputs[0].element_type
        super(WindowMap, wm).__init__(ty, et)
        wm.n = n
        wm.inputs = inputs
        wm.body = body
        wm.input_vars = input_vars
        return wm

    def children(
        self,
    ) -> List[
        KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D | KernelMatrixExpr | SignalVectorExpr | SignalExpr2DVec
    ]:
        return list(self.inputs) + [self.body]  # type: ignore[return-value]


@dataclass(unsafe_hash=True)
class WindowMapVec(SignalVectorExpr):
    """Vector variant of WindowMap: produces n_out interleaved output streams.

    Partitions each input into n_interleave interleaved substreams, applies
    body which returns n_interleave * n_out components, and zips them back into
    n_out output streams (inherited as SignalVectorExpr.n).

    For n_interleave=2 (even/odd): output component i is the interleaving of
    body components 2i (substream 0 / even) and 2i+1 (substream 1 / odd).
    Requires body.n == n_interleave * n_out.
    """

    n_interleave: int
    inputs: Tuple[SignalExpr, ...]
    body: SignalVectorExpr
    input_vars: Tuple[Tuple[WMVar, ...], ...]
    __match_args__ = ("n_interleave", "inputs", "body", "input_vars")

    var_count: int = 1

    def __init__(
        self,
        n_interleave: int,
        n_out: int,
        inputs: List[SignalExpr],
        func: Callable[[List[List[WMVar]]], SignalVectorExpr],
    ) -> None:
        assert len(inputs) > 0
        ty = inputs[0].ty
        et = inputs[0].element_type
        name = f"$wmv{WindowMapVec.var_count}"
        WindowMapVec.var_count += 1
        self.n_interleave = n_interleave
        self.inputs = tuple(inputs)
        self.input_vars = tuple(
            tuple(WMVar(f"{name}_{i}_{j}", ty, et) for j in range(n_interleave)) for i in range(len(inputs))
        )
        subs = [list(self.input_vars[i]) for i in range(len(inputs))]
        self.body = func(subs)
        assert (
            self.body.n == n_interleave * n_out
        ), f"WindowMapVec body must have n_interleave*n_out={n_interleave * n_out} components, got {self.body.n}"
        super().__init__(ty, et, n_out)

    @staticmethod
    def make(
        n_interleave: int,
        n_out: int,
        inputs: Tuple[SignalExpr, ...],
        body: SignalVectorExpr,
        input_vars: Tuple[Tuple[WMVar, ...], ...],
    ) -> "WindowMapVec":
        wmv: WindowMapVec = WindowMapVec.__new__(WindowMapVec)
        ty = inputs[0].ty
        et = inputs[0].element_type
        super(WindowMapVec, wmv).__init__(ty, et, n_out)
        wmv.n_interleave = n_interleave
        wmv.inputs = inputs
        wmv.body = body
        wmv.input_vars = input_vars
        return wmv

    def children(
        self,
    ) -> List[
        KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D | KernelMatrixExpr | SignalVectorExpr | SignalExpr2DVec
    ]:
        return list(self.inputs) + [self.body]  # type: ignore[return-value]
