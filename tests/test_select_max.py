"""Tests for the SelectMax (argmax) operator and the Smith-Waterman traceback
pointer matrix built as a mutual TropMax recurrence."""

from __future__ import annotations

import numpy as np

from vecrec import CodeGen
from vecrec.codegen import generate_kernel_executable
from vecrec.expr import (
    Convolve,
    Ith,
    Num,
    PointwiseMul,
    Recurse,
    SAdd,
    SignalExpr,
    SignalVector,
    TIKernel,
    Type,
    Var,
    Var2D,
    with_types,
)
from vecrec.expr.base import SignalVectorExpr
from vecrec.expr.signal import RVec2D
from vecrec.expr.signal_ops import RepeaterVec, SelectMax
from vecrec.optimize import get_candidate_schedules
from vecrec.util import ElementType

GAP = -1
_BOUND = np.int64(np.iinfo(np.int64).min // 8)
_RESTART, _DIAG, _UP, _LEFT = 0, 1, 2, 3


def _select_reference(arrays: list[np.ndarray], labels: list[int]) -> np.ndarray:
    """Strict-greater argmax with first-wins tie-break (matches SelectMax)."""
    n = len(arrays[0])
    out = np.empty(n, dtype=np.int64)
    for t in range(n):
        best = int(arrays[0][t])
        idx = labels[0]
        for k in range(1, len(arrays)):
            v = int(arrays[k][t])
            if v > best:
                best = v
                idx = labels[k]
        out[t] = idx
    return out


def test_select_max_tie_break_first_wins() -> None:
    """SelectMax picks the earliest candidate on ties (strict-greater update)."""
    with with_types(Type.TropMax, ElementType.I64):
        expr: SignalExpr = SelectMax([Var("a"), Var("b"), Var("c")], [0, 1, 2])

    scheds = get_candidate_schedules(expr, max_bits=256)
    assert len(scheds) > 0

    # pos0: (5,5,2) -> tie between a and b, a (label 0) wins
    # pos1: (3,7,7) -> b strictly beats a; c ties b but b already leads -> label 1
    # pos2: (9,9,1) -> tie a/b, a wins -> label 0
    a = np.array([5, 3, 9, 0, 4, 4, 1, 8], dtype=np.int64)
    b = np.array([5, 7, 9, 0, 4, 5, 1, 2], dtype=np.int64)
    c = np.array([2, 7, 1, 0, 9, 5, 1, 2], dtype=np.int64)
    expected = _select_reference([a, b, c], [0, 1, 2])

    for idx, sched in enumerate(scheds):
        runner = generate_kernel_executable(CodeGen(), sched)
        out, _, _ = runner.run({"a": a, "b": b, "c": c})
        assert out.dtype == np.int64
        m = min(len(out), len(expected))
        assert np.array_equal(out[:m], expected[:m]), (
            f"Schedule {idx}: tie-break mismatch\nGot:      {out[:m].tolist()}\n"
            f"Expected: {expected[:m].tolist()}"
        )


def _build_sw_traceback() -> SignalExpr:
    with with_types(Type.TropMax, ElementType.I64):
        s = Var2D("s")
        neginf = Type.TropMax.zero()
        z = TIKernel.z()

        def body(prev: RVec2D) -> SignalVectorExpr:
            prev_m = prev.at(0).at(0)
            restart = Num(0)
            diag = PointwiseMul(Convolve(z, prev_m), Ith(s, 0))
            up = PointwiseMul(prev_m, Num(GAP))
            base = SAdd(SAdd(restart, diag), up)
            M = Recurse(TIKernel([neginf, GAP]), base)
            left = PointwiseMul(Convolve(z, M), Num(GAP))
            P = SelectMax([restart, diag, up, left], [_RESTART, _DIAG, _UP, _LEFT])
            return SignalVector([M, P])

        return Ith(RepeaterVec(2, body).at(1), 0)


def _sw_pointer_reference(s: np.ndarray) -> np.ndarray:
    rows, cols = s.shape
    M = np.zeros((rows, cols), dtype=np.int64)
    P = np.zeros((rows, cols), dtype=np.int64)
    for i in range(rows):
        prev_row = M[i - 1] if i > 0 else np.full(cols, _BOUND, dtype=np.int64)
        for j in range(cols):
            up_val = GAP + int(prev_row[j])
            diag_input = int(prev_row[j - 1]) if j > 0 else int(_BOUND)
            diag_val = int(s[i, j]) + diag_input
            left_val = GAP + (int(M[i, j - 1]) if j > 0 else int(_BOUND))
            candidates = [0, diag_val, up_val, left_val]
            best = candidates[0]
            idx = _RESTART
            for k in range(1, 4):
                if candidates[k] > best:
                    best = candidates[k]
                    idx = k
            M[i, j] = best
            P[i, j] = idx
    return P.flatten()


def test_smith_waterman_traceback_pointers_correct() -> None:
    """Every schedule of the SW traceback expression matches the pointer reference."""
    expr = _build_sw_traceback()
    scheds = get_candidate_schedules(expr, max_bits=256)
    assert len(scheds) > 0

    rng = np.random.default_rng(3)
    rows, cols = 8, 64
    s = rng.integers(0, 1000, (rows, cols), dtype=np.int64)
    expected = _sw_pointer_reference(s)

    for idx, sched in enumerate(scheds):
        runner = generate_kernel_executable(CodeGen(), sched, row_size=cols)
        out, _, _ = runner.run({"s": s})
        m = min(len(out), len(expected))
        assert np.array_equal(out[:m], expected[:m]), (
            f"Schedule {idx}: pointer mismatch at "
            f"{int(np.where(out[:m] != expected[:m])[0][0])}"
        )
    # The traceback should genuinely exercise more than just the restart label.
    runner0 = generate_kernel_executable(CodeGen(), scheds[0], row_size=cols)
    out0, _, _ = runner0.run({"s": s})
    assert set(np.unique(out0[: len(expected)]).tolist()) - {_RESTART}, (
        "Expected non-restart traceback pointers to appear"
    )
