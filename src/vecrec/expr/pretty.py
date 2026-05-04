"""Pretty printer for the VecRec IR."""

from __future__ import annotations
from typing import Callable, Sequence

from vecrec.expr.base import (
    RecLang,
)
from vecrec.expr.signal import Num, Impulse, Var, RVar2D, Var2D
from vecrec.expr.signal_ops import (
    SAdd,
    SSub,
    PointwiseMul,
    PointwiseDiv,
    SNeg,
    Convolve,
    Convolve2D,
    Recurse,
    Recurse2D,
    ConvertLanes,
    Repeater,
    Ith,
    Signal2DAt,
    RepeaterVec,
    WMVar,
    WindowMap,
    WindowMapVec,
)
from vecrec.expr.signal_vec import SignalVector, ConvolveV, RecurseV, ConvertLanesV, VectorAt
from vecrec.expr.kernel import TIKernel, TVKernel, TIKernel2D, TVKernel2D
from vecrec.expr.kernel_ops import KAdd, KSub, KConvolve, KNeg, KConvertLanes
from vecrec.expr.kernel_matrix import KernelMatrix

# Precedence levels (higher binds tighter)
_PREC_BINDER = 10  # μ (Repeater)
_PREC_RECURSE = 20  # rec(...)
_PREC_ADD = 30  # +, -
_PREC_MUL = 40  # ×, ·, /
_PREC_UNARY = 50  # unary -
_PREC_INDEX = 60  # a[i]
_PREC_ATOM = 100  # literals, variables, kernels


def _prec(expr: RecLang) -> int:
    match expr:
        case Num(value=v) if v < 0:
            return _PREC_UNARY
        case Num() | Impulse() | Var() | RVar2D() | WMVar():
            return _PREC_ATOM
        case TIKernel() | TVKernel() | TIKernel2D() | TVKernel2D():
            return _PREC_ATOM
        case Ith():
            return _PREC_INDEX
        case SNeg() | KNeg():
            return _PREC_UNARY
        case Convolve() | Convolve2D() | KConvolve():
            return _PREC_MUL
        case PointwiseMul() | PointwiseDiv():
            return _PREC_MUL - 5
        case SAdd() | SSub() | KAdd() | KSub():
            return _PREC_ADD
        case Recurse() | Recurse2D():
            return _PREC_RECURSE
        case Repeater() | RepeaterVec() | WindowMap() | WindowMapVec():
            return _PREC_BINDER
        case Signal2DAt() | VectorAt():
            return _PREC_INDEX
        case SignalVector() | ConvolveV() | RecurseV():
            return _PREC_ATOM
        case ConvertLanes() | ConvertLanesV():
            return _prec(expr.a)
        case KConvertLanes():
            return _prec(expr.a)
    return _PREC_ATOM


def _wrap(inner: str, inner_prec: int, outer_prec: int, *, right: bool = False) -> str:
    """Wrap in parens if the inner expression binds less tightly."""
    # For right operands of non-commutative ops (-, /), also parenthesize at equal precedence.
    needs = inner_prec <= outer_prec if right else inner_prec < outer_prec
    return f"({inner})" if needs else inner


def _fmt_num(v: float) -> str:
    if v != v:  # NaN
        return "NaN"
    if v == float("inf"):
        return "∞"
    if v == float("-inf"):
        return "-∞"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:g}"


# Source - https://stackoverflow.com/a/13875688
# Posted by Eric, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-24, License - CC BY-SA 3.0


def _fmt_superscript(s: str) -> str:
    return "".join(dict(zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")).get(c, c) for c in s)


def _fmt_kernel_row(row: list[float]) -> str:
    return ", ".join(_fmt_num(v) for v in row)


def _all_children(expr: RecLang) -> list[RecLang]:
    """Return all sub-expressions, including those stored in list fields (e.g. TVKernel2D)."""
    match expr:
        case TVKernel2D():
            return [v for row in expr.data for v in row]
        case KernelMatrix():
            return [k for row in expr.data for k in row]
        case _:
            return expr.children()  # type: ignore[return-value]


def _fmt_one(
    expr: RecLang,
    fmt: Callable[[RecLang], str],
    bound: set[int],
) -> str:
    """Format one level of an expression.

    `fmt` is called for each child; `bound` holds the ids of nodes that have
    been assigned a let-binding name (so their effective precedence is _PREC_ATOM).
    """

    def cp(e: RecLang) -> int:
        return _PREC_ATOM if id(e) in bound else _prec(e)

    def w(e: RecLang, outer: int, *, right: bool = False) -> str:
        return _wrap(fmt(e), cp(e), outer, right=right)

    match expr:
        # --- Leaves ---
        case Num(value=v):
            return _fmt_num(v)

        case Impulse(value=v):
            return f"δ({_fmt_num(v)})"

        case Var(name=n):
            return n

        case RVar2D(name=n):
            return n.lstrip("$")

        case Var2D(name=n):
            return n

        # --- Kernels ---
        case TIKernel():
            ty = expr.ty
            if all(ty.is_zero(v) for v in expr.data[:-1]) and ty.is_one(expr.data[-1]):
                return f"z⁻{_fmt_superscript(str(len(expr.data) - 1))}"
            return f"[{_fmt_kernel_row(expr.data)}]"

        case TVKernel():
            vals = ", ".join(fmt(v) for v in expr.data)
            return f"[{vals}]"

        case TIKernel2D():
            rows = "; ".join(_fmt_kernel_row(row) for row in expr.data)
            return f"[{rows}]"

        case TVKernel2D():
            rows = "; ".join(", ".join(fmt(v) for v in row) for row in expr.data)
            return f"[{rows}]"

        # --- Signal binary ops ---
        case SAdd(a, b):
            p = _PREC_ADD
            return f"{w(a, p)} + {w(b, p)}"

        case SSub(a, b):
            p = _PREC_ADD
            return f"{w(a, p)} - {w(b, p, right=True)}"

        case PointwiseMul(a, b):
            p = _PREC_MUL
            return f"{w(a, p)} · {w(b, p)}"

        case PointwiseDiv(a, b):
            p = _PREC_MUL
            return f"{w(a, p)} / {w(b, p, right=True)}"

        case SNeg(a):
            return f"-{w(a, _PREC_UNARY)}"

        # --- Convolutions ---
        case Convolve(a, b):
            p = _PREC_MUL
            return f"{w(a, p)} × {w(b, p)}"

        case Convolve2D(a, b):
            p = _PREC_MUL
            return f"{w(a, p)} × {w(b, p)}"

        # --- Recurrences ---
        case Recurse(a, g):
            return f"rec({fmt(a)}, {fmt(g)})"

        case Recurse2D(a, g):
            return f"rec₂({fmt(a)}, {fmt(g)})"

        # --- Kernel ops ---
        case KAdd(a, b):
            p = _PREC_ADD
            return f"{w(a, p)} + {w(b, p)}"

        case KSub(a, b):
            p = _PREC_ADD
            return f"{w(a, p)} - {w(b, p, right=True)}"

        case KConvolve(a, b):
            p = _PREC_MUL
            return f"{w(a, p)} × {w(b, p)}"

        case KNeg(a):
            return f"-{w(a, _PREC_UNARY)}"

        # --- Indexing ---
        case Ith(a, i):
            return f"{w(a, _PREC_INDEX)}[{i}]"

        # --- Repeater (μ-binder) ---
        case Repeater():
            name = expr.prev_rows_var.name.lstrip("$")
            return f"μ{name}. {fmt(expr.a)}"

        case RepeaterVec():
            names = ", ".join(pv.name.lstrip("$") for pv in expr.prev_vars)
            return f"μ({names}). {fmt(expr.a)}"

        # --- Signal vector indexing ---
        case Signal2DAt(a, i):
            return f"{w(a, _PREC_INDEX)}[{i}]"

        case VectorAt(a, i):
            return f"{w(a, _PREC_INDEX)}[{i}]"

        # --- Signal vectors ---
        case SignalVector(data):
            parts = ", ".join(fmt(s) for s in data)
            return f"({parts})"

        case ConvolveV(a, g):
            p = _PREC_MUL
            return f"{w(a, p)} × {w(g, p)}"

        case RecurseV(a, g):
            return f"rec({fmt(a)}, {fmt(g)})"

        case KernelMatrix():
            rows = "; ".join(", ".join(fmt(k) for k in row) for row in expr.data)
            return f"[{rows}]"

        # --- WindowMap binder ---
        case WMVar(name):
            return name.lstrip("$")

        case WindowMap():
            inputs_str = ", ".join(fmt(inp) for inp in expr.inputs)
            return f"wmap[{expr.n}]({inputs_str}; {fmt(expr.body)})"

        case WindowMapVec():
            inputs_str = ", ".join(fmt(inp) for inp in expr.inputs)
            return f"wmapv[{expr.n_interleave}, {expr.n}]({inputs_str}; {fmt(expr.body)})"

        # --- Lane conversions (transparent) ---
        case ConvertLanes(a):
            return fmt(a)

        case ConvertLanesV(a):
            return fmt(a)

        case KConvertLanes(a):
            return fmt(a)

    return repr(expr)


def pps(exprs: Sequence[RecLang]) -> list[str]:
    """Pretty-print a list of IR expressions."""
    return [pp(expr) for expr in exprs]


def pp(expr: RecLang) -> str:
    """Pretty-print an IR expression, using let-bindings for shared subexpressions.

    Two subexpressions are considered "shared" when they are structurally equal
    (same AST shape), regardless of whether they are the same Python object.
    This catches duplicated subtrees built separately during transforms.

    To keep output compact, a structural class is only kept as a `let` binding
    if its bound name appears more than once in the rendered output.
    """
    # --- Pass 1: count references to each node by identity ---
    # This detects both DAG sharing (same object, multiple parents) and lets us
    # know the total weight of each structural equivalence class later.
    id_to_node: dict[int, RecLang] = {}
    ref_count: dict[int, int] = {}

    def _count(e: RecLang) -> None:
        eid = id(e)
        ref_count[eid] = ref_count.get(eid, 0) + 1
        if ref_count[eid] == 1:
            id_to_node[eid] = e
            for c in _all_children(e):
                _count(c)

    _count(expr)

    # --- Pass 2: group nodes by structural equality (post-order) ---
    # repr_of maps every node id to the id of the first structurally-equal node
    # encountered (its "representative").  struct_groups maps each representative
    # to the list of all ids in its class.  Insertion order of struct_groups is
    # post-order, so children always appear before their parents.
    repr_of: dict[int, int] = {}
    struct_groups: dict[int, list[int]] = {}
    seen: set[int] = set()

    def _group(e: RecLang) -> None:
        eid = id(e)
        if eid in seen:
            return
        seen.add(eid)
        for c in _all_children(e):
            _group(c)
        for repr_id in struct_groups:
            if id_to_node[repr_id] == e:
                struct_groups[repr_id].append(eid)
                repr_of[eid] = repr_id
                return
        struct_groups[eid] = [eid]
        repr_of[eid] = eid

    _group(expr)

    # --- Assign let-binding names to shared non-trivial structural classes ---
    # A class is "shared" when the sum of its members' ref-counts exceeds 1,
    # covering both duplicate objects (len > 1) and DAG-shared single objects.
    name_map: dict[int, str] = {}  # repr_id -> name
    binding_order: list[int] = []  # repr_ids in definition (post-)order
    counter = [0]

    for repr_id, members in struct_groups.items():
        total_refs = sum(ref_count[eid] for eid in members)
        if total_refs > 1 and not id_to_node[repr_id].is_leaf():
            name_map[repr_id] = f"t{counter[0]}"
            counter[0] += 1
            binding_order.append(repr_id)

    def _compute_name_uses(bound_repr_ids: set[int]) -> dict[int, int]:
        """Count occurrences of bound names in the rendered output.

        Traversal mirrors formatting: once a bound child is reached we count one
        variable occurrence and do not descend into that child.
        """
        uses = {repr_id: 0 for repr_id in bound_repr_ids}

        def _visit(e: RecLang, *, root: bool = False) -> None:
            rid = repr_of[id(e)]
            if rid in bound_repr_ids and not root:
                uses[rid] += 1
                return
            for child in _all_children(e):
                _visit(child)

        _visit(expr, root=True)
        for repr_id in bound_repr_ids:
            _visit(id_to_node[repr_id], root=True)
        return uses

    # Prune bindings that are referenced only once in the rendered program.
    # Recompute to a fixed point because removing one binding can change usage
    # counts for others.
    active_repr_ids = set(name_map)
    while active_repr_ids:
        uses = _compute_name_uses(active_repr_ids)
        drop = {repr_id for repr_id in active_repr_ids if uses[repr_id] <= 1}
        if not drop:
            break
        active_repr_ids -= drop

    name_map = {repr_id: name for repr_id, name in name_map.items() if repr_id in active_repr_ids}
    binding_order = [repr_id for repr_id in binding_order if repr_id in active_repr_ids]

    # bound: all node ids whose structural class has a let-binding name.
    # Used by _fmt_one to assign _PREC_ATOM to such children.
    bound = {eid for repr_id in name_map for eid in struct_groups[repr_id]}

    def fmt(e: RecLang) -> str:
        rid = repr_of[id(e)]
        if rid in name_map:
            return name_map[rid]
        return _fmt_one(e, fmt, bound)

    if not name_map:
        return _fmt_one(expr, fmt, bound)

    lines = [f"let {name_map[repr_id]} = {_fmt_one(id_to_node[repr_id], fmt, bound)}" for repr_id in binding_order]
    return "\n".join(lines) + "\nin " + _fmt_one(expr, fmt, bound)
