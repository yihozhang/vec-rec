"""Pretty printer for the VecRec IR."""

from __future__ import annotations
from typing import Sequence

from vecrec.expr.base import (
    RecLang, KernelExpr, KernelExpr2D, SignalExpr, SignalExpr2D,
)
from vecrec.expr.signal import Num, Var, RVar2D
from vecrec.expr.signal_ops import (
    SAdd, SSub, PointwiseMul, PointwiseDiv, SNeg,
    Convolve, Convolve2D, Recurse, Recurse2D,
    ConvertLanes, Repeater, Ith,
)
from vecrec.expr.kernel import TIKernel, TVKernel, TIKernel2D, TVKernel2D
from vecrec.expr.kernel_ops import KAdd, KSub, KConvolve, KNeg, KConvertLanes

# Precedence levels (higher binds tighter)
_PREC_BINDER = 10     # μ (Repeater)
_PREC_RECURSE = 20    # rec(...)
_PREC_ADD = 30        # +, -
_PREC_MUL = 40        # ×, ·, /
_PREC_UNARY = 50      # unary -
_PREC_INDEX = 60      # a[i]
_PREC_ATOM = 100      # literals, variables, kernels


def _prec(expr: RecLang) -> int:
    match expr:
        case Num(value=v) if v < 0:
            return _PREC_UNARY
        case Num() | Var() | RVar2D():
            return _PREC_ATOM
        case TIKernel() | TVKernel() | TIKernel2D() | TVKernel2D():
            return _PREC_ATOM
        case Ith():
            return _PREC_INDEX
        case SNeg() | KNeg():
            return _PREC_UNARY
        case Convolve() | Convolve2D() | PointwiseMul() | PointwiseDiv() | KConvolve():
            return _PREC_MUL
        case SAdd() | SSub() | KAdd() | KSub():
            return _PREC_ADD
        case Recurse() | Recurse2D():
            return _PREC_RECURSE
        case Repeater():
            return _PREC_BINDER
        case ConvertLanes():
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


def _fmt_kernel_row(row: list[float]) -> str:
    return ", ".join(_fmt_num(v) for v in row)

def pps(exprs: Sequence[RecLang]) -> list[str]:
    """Pretty-print a list of IR expressions."""
    return [pp(expr) for expr in exprs]

def pp(expr: RecLang) -> str:
    """Pretty-print an IR expression."""
    match expr:
        # --- Leaves ---
        case Num(value=v):
            return _fmt_num(v)

        case Var(name=n):
            return n

        case RVar2D(name=n):
            return n.lstrip("$")

        # --- Kernels ---
        case TIKernel():
            return f"[{_fmt_kernel_row(expr.data)}]"

        case TVKernel():
            vals = ", ".join(pp(v) for v in expr.data)
            return f"[{vals}]"

        case TIKernel2D():
            rows = "; ".join(
                _fmt_kernel_row(row) for row in expr.data
            )
            return f"[{rows}]"

        case TVKernel2D():
            rows = "; ".join(
                ", ".join(pp(v) for v in row) for row in expr.data
            )
            return f"[{rows}]"

        # --- Signal binary ops ---
        case SAdd(a, b):
            p = _PREC_ADD
            return f"{_wrap(pp(a), _prec(a), p)} + {_wrap(pp(b), _prec(b), p)}"

        case SSub(a, b):
            p = _PREC_ADD
            return f"{_wrap(pp(a), _prec(a), p)} - {_wrap(pp(b), _prec(b), p, right=True)}"

        case PointwiseMul(a, b):
            p = _PREC_MUL
            return f"{_wrap(pp(a), _prec(a), p)} · {_wrap(pp(b), _prec(b), p)}"

        case PointwiseDiv(a, b):
            p = _PREC_MUL
            return f"{_wrap(pp(a), _prec(a), p)} / {_wrap(pp(b), _prec(b), p, right=True)}"

        case SNeg(a):
            return f"-{_wrap(pp(a), _prec(a), _PREC_UNARY)}"

        # --- Convolutions ---
        case Convolve(a, b):
            p = _PREC_MUL
            return f"{_wrap(pp(a), _prec(a), p)} × {_wrap(pp(b), _prec(b), p)}"

        case Convolve2D(a, b):
            p = _PREC_MUL
            return f"{_wrap(pp(a), _prec(a), p)} × {_wrap(pp(b), _prec(b), p)}"

        # --- Recurrences ---
        case Recurse(a, g):
            return f"rec({pp(a)}, {pp(g)})"

        case Recurse2D(a, g):
            return f"rec\u2082({pp(a)}, {pp(g)})"

        # --- Kernel ops ---
        case KAdd(a, b):
            p = _PREC_ADD
            return f"{_wrap(pp(a), _prec(a), p)} + {_wrap(pp(b), _prec(b), p)}"

        case KSub(a, b):
            p = _PREC_ADD
            return f"{_wrap(pp(a), _prec(a), p)} - {_wrap(pp(b), _prec(b), p, right=True)}"

        case KConvolve(a, b):
            p = _PREC_MUL
            return f"{_wrap(pp(a), _prec(a), p)} × {_wrap(pp(b), _prec(b), p)}"

        case KNeg(a):
            return f"-{_wrap(pp(a), _prec(a), _PREC_UNARY)}"

        # --- Indexing ---
        case Ith(a, i):
            return f"{_wrap(pp(a), _prec(a), _PREC_INDEX)}[{i}]"

        # --- Repeater (μ-binder) ---
        case Repeater():
            name = expr.prev_rows_var.name.lstrip("$")
            return f"\u03bc{name}. {pp(expr.a)}"

        # --- Lane conversions (transparent) ---
        case ConvertLanes(a):
            return pp(a)

        case KConvertLanes(a):
            return pp(a)

    return repr(expr)
