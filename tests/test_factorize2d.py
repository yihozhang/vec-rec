"""Tests for 2D kernel factorization."""

import numpy as np
import pytest

from vecrec.expr import *
from vecrec.expr.base import SignalExpr
from vecrec.expr.kernel import TIKernel, TIKernel2D
from vecrec.expr.signal import Var, Var2D
from vecrec.expr.signal_ops import Convolve, Convolve2D, Ith, Repeater, SAdd
from vecrec.factorize import factorize_kernel2d
from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.transform import (
    AnnotateLanes,
    ConstantFold,
    Eliminate2DKernels,
    Factorize2D,
    Preorder,
    SaturatePreorder,
    Try,
)
from vecrec.util import ElementType


# ---------------------------------------------------------------------------
# factorize_kernel2d unit tests
# ---------------------------------------------------------------------------


def test_separable_ones():
    """All-ones 3x3 is rank-1: each entry = 1*1."""
    with with_types(Type.Arith, ElementType.Float):
        A = TIKernel2D([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        result = factorize_kernel2d(A)
    assert result is not None
    row_k, col_k2d = result
    # Verify reconstruction: outer(col_vec, row_vec) ≈ A
    col_vec = [col_k2d.data[r][0] if col_k2d.data[r] else 0.0 for r in range(3)]
    row_vec = row_k.data
    reconstructed = [[col_vec[r] * row_vec[c] for c in range(len(row_vec))] for r in range(3)]
    for r in range(3):
        for c in range(3):
            assert abs(reconstructed[r][c] - 1.0) < 1e-8


def test_separable_diff_kernel():
    """4x4 difference kernel: A[r][c] = col[r]*row[c] for col=[1,0,0,-1], row=[1,0,0,-1]."""
    with with_types(Type.Arith, ElementType.Float):
        data = [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0,  0.0],
            [0.0, 0.0, 0.0,  0.0],
            [-1.0, 0.0, 0.0, 1.0],
        ]
        A = TIKernel2D(data)
        result = factorize_kernel2d(A)
    assert result is not None
    row_k, col_k2d = result
    # Check the 4 corner values reconstruct correctly
    col_vec = [col_k2d.data[r][0] if col_k2d.data[r] else 0.0 for r in range(4)]
    row_vec = list(row_k.data) + [0.0] * max(0, 4 - len(row_k.data))
    assert abs(col_vec[0] * row_vec[0] - 1.0) < 1e-8
    assert abs(col_vec[0] * row_vec[3] - (-1.0)) < 1e-8
    assert abs(col_vec[3] * row_vec[0] - (-1.0)) < 1e-8
    assert abs(col_vec[3] * row_vec[3] - 1.0) < 1e-8


def test_non_separable_identity():
    """2x2 identity matrix has rank 2 — not separable."""
    with with_types(Type.Arith, ElementType.Float):
        A = TIKernel2D([[1.0, 0.0], [0.0, 1.0]])
        result = factorize_kernel2d(A)
    assert result is None


def test_non_separable_antidiag():
    """Anti-diagonal [[0,1],[1,0]] is rank 2 — not separable."""
    with with_types(Type.Arith, ElementType.Float):
        A = TIKernel2D([[0.0, 1.0], [1.0, 0.0]])
        result = factorize_kernel2d(A)
    assert result is None


def test_single_row_always_separable():
    """A single-row kernel is trivially rank-1."""
    with with_types(Type.Arith, ElementType.Float):
        A = TIKernel2D([[3.0, 1.0, 4.0]])
        result = factorize_kernel2d(A)
    assert result is not None


def test_single_col_always_separable():
    """A single-column kernel is trivially rank-1."""
    with with_types(Type.Arith, ElementType.Float):
        A = TIKernel2D([[2.0], [5.0], [1.0]])
        result = factorize_kernel2d(A)
    assert result is not None


# ---------------------------------------------------------------------------
# Factorize2D transform tests
# ---------------------------------------------------------------------------


def test_factorize2d_transform_fires():
    """Factorize2D should rewrite Convolve2D with a separable kernel."""
    with with_types(Type.Arith, ElementType.Float):
        kernel = TIKernel2D([[1.0, 2.0], [2.0, 4.0]])  # rank-1: [1,2]^T * [1,2]
        sig2d = Var2D("x", 2)
        expr = Convolve2D(kernel, sig2d)
        results = Factorize2D().apply_signal(expr)
    assert len(results) == 1
    assert isinstance(results[0], Convolve)
    inner = results[0].b
    assert isinstance(inner, Convolve2D)
    # Inner kernel should be a single-column 2D kernel
    assert isinstance(inner.a, TIKernel2D)
    assert all(len(row) <= 1 for row in inner.a.data)


def test_factorize2d_transform_does_not_fire_non_separable():
    """Factorize2D should not fire for non-separable kernels."""
    with with_types(Type.Arith, ElementType.Float):
        kernel = TIKernel2D([[1.0, 0.0], [0.0, 1.0]])  # rank-2
        sig2d = Var2D("x", 2)
        expr = Convolve2D(kernel, sig2d)
        results = Factorize2D().apply_signal(expr)
    assert len(results) == 0


def test_factorize2d_codegen():
    """Factorize2D produces code that compiles and runs."""
    with with_types(Type.Arith, ElementType.Float):
        # separable 3×3 box kernel
        kernel = TIKernel2D([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        g = Var("g")

        def repeater_func(prev_rows):
            return SAdd(Convolve2D(kernel, prev_rows), g)

        expr: SignalExpr = Ith(Repeater(repeater_func, n_rows=4), 0)

    # Apply Factorize2D then eliminate 2D kernels, then codegen
    expr = Preorder(Try(Factorize2D())).apply_signal(expr)[0]
    expr = SaturatePreorder(ConstantFold).apply_signal(expr)[0]
    expr = Preorder(Try(Eliminate2DKernels())).apply_signal(expr)[0]
    expr = AnnotateLanes(512).apply_signal(expr)[0]

    result = generate_and_run_benchmark(CodeGen(), [expr])
    assert result["return_code"] == 0
