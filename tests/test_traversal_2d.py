"""Tests for Preorder and Postorder traversal with 2D kernel expressions."""

from __future__ import annotations
from typing import Sequence

from vecrec.expr import *
from vecrec.expr.base import (
    KernelExpr,
    KernelExpr2D,
    KernelMatrixExpr,
    SignalExpr2D,
    SignalVectorExpr,
)
from vecrec.expr.kernel import TIKernel, TVKernel, TIKernel2D, TVKernel2D
from vecrec.expr.signal import Var, Var2D
from vecrec.expr.signal_ops import Convolve, Convolve2D, Recurse2D, Ith, Repeater, SAdd
from vecrec.transform import Preorder, Postorder
from vecrec.transform.combinators import Noop
from vecrec.transform import Transform
from vecrec.util import ElementType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OrderRecorder(Transform):
    """Records visit order by type label; useful for pre/postorder sequencing tests."""

    def __init__(self, visit_order: list[tuple[str, object]]) -> None:
        self.visit_order = visit_order

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        self.visit_order.append(("kernel2d", expr))
        return [expr]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        self.visit_order.append(("signal", expr))
        return [expr]

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        self.visit_order.append(("signal2d", expr))
        return [expr]

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        self.visit_order.append(("kernel", expr))
        return [expr]

    def apply_kernel_matrix(self, expr: KernelMatrixExpr) -> Sequence[KernelMatrixExpr]:
        self.visit_order.append(("kernel_matrix", expr))
        return [expr]

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        self.visit_order.append(("signal_vector", expr))
        return [expr]


class RecordingTransform(Transform):
    """Records which expressions it is applied to and returns them unchanged."""

    def __init__(self) -> None:
        self.kernel_calls: list[KernelExpr] = []
        self.kernel2d_calls: list[KernelExpr2D] = []
        self.kernel_matrix_calls: list[KernelMatrixExpr] = []
        self.signal_calls: list[SignalExpr] = []
        self.signal2d_calls: list[SignalExpr2D] = []
        self.signal_vector_calls: list[SignalVectorExpr] = []

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        self.kernel_calls.append(expr)
        return [expr]

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        self.kernel2d_calls.append(expr)
        return [expr]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        self.signal_calls.append(expr)
        return [expr]

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        self.signal2d_calls.append(expr)
        return [expr]

    def apply_kernel_matrix(self, expr: KernelMatrixExpr) -> Sequence[KernelMatrixExpr]:
        self.kernel_matrix_calls.append(expr)
        return [expr]

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        self.signal_vector_calls.append(expr)
        return [expr]


# ---------------------------------------------------------------------------
# Preorder with TIKernel2D
# ---------------------------------------------------------------------------


def test_preorder_tikernel2d_leaf_unchanged():
    """Preorder with Noop leaves a TIKernel2D leaf unchanged."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[1.0, 2.0], [3.0, 4.0]])
        results = Preorder(Noop()).apply_kernel2d(k)
    assert len(results) == 1
    assert results[0] == k


def test_preorder_tikernel2d_transform_fires_at_root():
    """Preorder fires the transform at the root TIKernel2D before recursing."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[1.0, 0.0], [0.0, 1.0]])
        recorder = RecordingTransform()
        results = Preorder(recorder).apply_kernel2d(k)

    # The root kernel should have been passed to the transform
    assert k in recorder.kernel2d_calls
    assert len(results) == 1
    assert results[0] == k


# ---------------------------------------------------------------------------
# Postorder with TIKernel2D
# ---------------------------------------------------------------------------


def test_postorder_tikernel2d_leaf_unchanged():
    """Postorder with Noop leaves a TIKernel2D leaf unchanged."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[2.0, 3.0]])
        results = Postorder(Noop()).apply_kernel2d(k)
    assert len(results) == 1
    assert results[0] == k


def test_postorder_tikernel2d_transform_fires_at_root():
    """Postorder fires the transform at the root TIKernel2D (which is also the leaf)."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[5.0, 6.0], [7.0, 8.0]])
        recorder = RecordingTransform()
        results = Postorder(recorder).apply_kernel2d(k)

    assert k in recorder.kernel2d_calls
    assert len(results) == 1
    assert results[0] == k


# ---------------------------------------------------------------------------
# Preorder with TVKernel2D — signals are traversed
# ---------------------------------------------------------------------------


def test_preorder_tvkernel2d_visits_signal_children():
    """Preorder traverses into signal entries of TVKernel2D."""
    with with_types(Type.Arith, ElementType.Float):
        v = Var("x")
        num = Num(3.0)
        k = TVKernel2D([[v, num]])
        recorder = RecordingTransform()
        results = Preorder(recorder).apply_kernel2d(k)

    # The signal children of the TVKernel2D should have been visited
    assert v in recorder.signal_calls and num in recorder.signal_calls
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Postorder with TVKernel2D — signals are traversed
# ---------------------------------------------------------------------------


def test_postorder_tvkernel2d_visits_signal_children():
    """Postorder traverses into signal entries of TVKernel2D."""
    with with_types(Type.Arith, ElementType.Float):
        v = Var("y")
        num = Num(1.0)
        k = TVKernel2D([[v, num]])
        recorder = RecordingTransform()
        results = Postorder(recorder).apply_kernel2d(k)

    assert v in recorder.signal_calls and num in recorder.signal_calls
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Preorder with Convolve2D (SignalExpr containing KernelExpr2D)
# ---------------------------------------------------------------------------


def test_preorder_convolve2d_visits_kernel2d_and_signal2d():
    """Preorder traversal of Convolve2D visits the 2D kernel and 2D signal."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        sig2d = Var2D("img")
        expr = Convolve2D(k, sig2d)
        recorder = RecordingTransform()
        results = Preorder(recorder).apply_signal(expr)

    # The 2D kernel and the 2D signal should have been visited
    assert k in recorder.kernel2d_calls
    assert sig2d in recorder.signal2d_calls
    assert len(results) == 1


def test_preorder_convolve2d_preserves_structure():
    """Preorder with Noop preserves the Convolve2D structure."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[1.0, 0.0], [0.0, 1.0]])
        sig2d = Var2D("img")
        expr = Convolve2D(k, sig2d)
        results = Preorder(Noop()).apply_signal(expr)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, Convolve2D)
    assert result.a == k
    assert result.b == sig2d


# ---------------------------------------------------------------------------
# Postorder with Convolve2D
# ---------------------------------------------------------------------------


def test_postorder_convolve2d_visits_kernel2d_and_signal2d():
    """Postorder traversal of Convolve2D visits the 2D kernel and 2D signal."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[1.0, 2.0]])
        sig2d = Var2D("src")
        expr = Convolve2D(k, sig2d)
        recorder = RecordingTransform()
        results = Postorder(recorder).apply_signal(expr)

    assert k in recorder.kernel2d_calls
    assert sig2d in recorder.signal2d_calls
    assert len(results) == 1


def test_postorder_convolve2d_preserves_structure():
    """Postorder with Noop preserves the Convolve2D structure."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[3.0, 1.0]])
        sig2d = Var2D("src")
        expr = Convolve2D(k, sig2d)
        results = Postorder(Noop()).apply_signal(expr)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, Convolve2D)
    assert result.a == k
    assert result.b == sig2d


# ---------------------------------------------------------------------------
# Preorder vs Postorder ordering with Recurse2D
# ---------------------------------------------------------------------------


def test_preorder_recurse2d_visits_kernel_before_children():
    """Preorder fires the transform at Recurse2D root before its kernel child."""
    visit_order: list[tuple[str, object]] = []

    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[1.0, -1.0]])
        g = Var("input")
        expr2d = Recurse2D(k, g)

        recorder = OrderRecorder(visit_order)
        Preorder(recorder).apply_signal2d(expr2d)

    # In preorder the root (Recurse2D → signal2d) is visited first,
    # then children: the kernel2d and the signal.
    kinds = [kind for kind, _ in visit_order]
    assert "signal2d" in kinds
    assert "kernel2d" in kinds
    # Root signal2d visit should appear before the kernel2d child visit
    assert kinds.index("signal2d") < kinds.index("kernel2d")


def test_postorder_recurse2d_visits_children_before_root():
    """Postorder fires the transform on Recurse2D children before the root."""
    visit_order: list[tuple[str, object]] = []

    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[1.0, -1.0]])
        g = Var("input")
        expr2d = Recurse2D(k, g)

        recorder = OrderRecorder(visit_order)
        results = Postorder(recorder).apply_signal2d(expr2d)

    kinds = [kind for kind, _ in visit_order]
    # Children (kernel2d and signal) are visited before the root signal2d
    assert "kernel2d" in kinds
    assert "signal" in kinds
    assert "signal2d" in kinds
    # The kernel2d and signal children should appear before the root Recurse2D
    assert kinds.index("kernel2d") < kinds.index("signal2d")
    assert kinds.index("signal") < kinds.index("signal2d")
    # Result is still a Recurse2D
    assert len(results) == 1
    assert isinstance(results[0], Recurse2D)


# ---------------------------------------------------------------------------
# Preorder / Postorder with Repeater (SignalExpr2D)
# ---------------------------------------------------------------------------


def test_preorder_repeater_preserves_structure():
    """Preorder with Noop preserves a Repeater expression."""
    with with_types(Type.Arith, ElementType.Float):
        g = Var("g")

        def repeater_func(prev_rows):
            return SAdd(Convolve2D(TIKernel2D([[1.0, 1.0]]), prev_rows), g)

        expr: SignalExpr = Ith(Repeater(repeater_func), 0)
        results = Preorder(Noop()).apply_signal(expr)

    assert len(results) == 1
    assert isinstance(results[0], Ith)


def test_postorder_repeater_preserves_structure():
    """Postorder with Noop preserves a Repeater expression."""
    with with_types(Type.Arith, ElementType.Float):
        g = Var("g")

        def repeater_func(prev_rows):
            return SAdd(Convolve2D(TIKernel2D([[1.0, 1.0]]), prev_rows), g)

        expr: SignalExpr = Ith(Repeater(repeater_func), 0)
        results = Postorder(Noop()).apply_signal(expr)

    assert len(results) == 1
    assert isinstance(results[0], Ith)


# ---------------------------------------------------------------------------
# Cartesian product when a transform branches into multiple expressions
# ---------------------------------------------------------------------------


class BranchingTransform(Transform):
    """Branches leaf kernels and Vars into [original, negated/alt].

    - TIKernel2D  → [original, negated]
    - TIKernel    → [original, negated]
    - Var         → [original, Var(name + "_alt")]
    """

    def apply_kernel2d(self, expr: KernelExpr2D) -> Sequence[KernelExpr2D]:
        if isinstance(expr, TIKernel2D):
            negated = TIKernel2D([[-v for v in row] for row in expr.data], expr.ty, expr.element_type)
            return [expr, negated]
        return [expr]

    def apply_kernel(self, expr: KernelExpr) -> Sequence[KernelExpr]:
        if isinstance(expr, TIKernel):
            negated = TIKernel([-v for v in expr.data], expr.ty, expr.element_type)
            return [expr, negated]
        return [expr]

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        if isinstance(expr, Var):
            return [expr, Var(expr.name + "_alt")]
        return [expr]

    def apply_signal2d(self, expr: SignalExpr2D) -> Sequence[SignalExpr2D]:
        return [expr]

    def apply_kernel_matrix(self, expr: KernelMatrixExpr) -> Sequence[KernelMatrixExpr]:
        return [expr]

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        return [expr]


def test_preorder_recursev_visits_matrix_vector_and_kernel_children():
    """Preorder traversal of RecurseV visits matrix/vector nodes and matrix entries."""
    with with_types(Type.Arith, ElementType.Float):
        feedback = TIKernel([0.0, 0.5])
        km = KernelMatrix([[feedback]])
        g = SignalVector([Var("x")])
        expr = RecurseV(km, g)
        recorder = RecordingTransform()
        results = Preorder(recorder).apply_signal_vector(expr)

    assert len(results) == 1
    assert km in recorder.kernel_matrix_calls
    assert g in recorder.signal_vector_calls
    assert feedback in recorder.kernel_calls
    assert any(isinstance(s, Var) and s.name == "x" for s in recorder.signal_calls)


def test_postorder_convolvev_visits_matrix_vector_and_children():
    """Postorder traversal of ConvolveV visits matrix/vector nodes and matrix entries."""
    with with_types(Type.Arith, ElementType.Float):
        k_id = TIKernel([1.0])
        km = KernelMatrix([[k_id]])
        g = SignalVector([Var("u")])
        expr = ConvolveV(km, g)
        recorder = RecordingTransform()
        results = Postorder(recorder).apply_signal_vector(expr)

    assert len(results) == 1
    assert km in recorder.kernel_matrix_calls
    assert g in recorder.signal_vector_calls
    assert k_id in recorder.kernel_calls
    assert any(isinstance(s, Var) and s.name == "u" for s in recorder.signal_calls)


def test_preorder_cartesian_product_convolvev():
    """Preorder computes Cartesian products across KernelMatrix and SignalVector children."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel([1.0, -1.0])
        x = Var("x")
        expr = ConvolveV(KernelMatrix([[k]]), SignalVector([x]))
        results = Preorder(BranchingTransform()).apply_signal_vector(expr)

    assert len(results) == 4
    assert all(isinstance(r, ConvolveV) for r in results)
    convolve_results = [r for r in results if isinstance(r, ConvolveV)]

    kernel_datas = {
        tuple(r.a.data[0][0].data)
        for r in convolve_results
        if isinstance(r.a, KernelMatrix)
        if isinstance(r.a.data[0][0], TIKernel)
    }
    var_names = {
        r.g.data[0].name for r in convolve_results if isinstance(r.g, SignalVector) and isinstance(r.g.data[0], Var)
    }
    assert kernel_datas == {(1.0, -1.0), (-1.0, 1.0)}
    assert var_names == {"x", "x_alt"}


def test_postorder_cartesian_product_convolvev():
    """Postorder computes Cartesian products across KernelMatrix and SignalVector children."""
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel([1.0, -1.0])
        x = Var("x")
        expr = ConvolveV(KernelMatrix([[k]]), SignalVector([x]))
        results = Postorder(BranchingTransform()).apply_signal_vector(expr)

    assert len(results) == 4
    assert all(isinstance(r, ConvolveV) for r in results)
    convolve_results = [r for r in results if isinstance(r, ConvolveV)]

    kernel_datas = {
        tuple(r.a.data[0][0].data)
        for r in convolve_results
        if isinstance(r.a, KernelMatrix)
        if isinstance(r.a.data[0][0], TIKernel)
    }
    var_names = {
        r.g.data[0].name for r in convolve_results if isinstance(r.g, SignalVector) and isinstance(r.g.data[0], Var)
    }
    assert kernel_datas == {(1.0, -1.0), (-1.0, 1.0)}
    assert var_names == {"x", "x_alt"}


def test_postorder_cartesian_product_recurse2d():
    """Postorder computes the Cartesian product when a transform branches on each child.

    Recurse2D(k, g) with a transform that produces 2 kernel variants and 2 signal
    variants should yield 2 * 2 = 4 Recurse2D results.
    """
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[1.0, -1.0]])
        g = Var("input")
        expr2d = Recurse2D(k, g)
        results = Postorder(BranchingTransform()).apply_signal2d(expr2d)

    assert len(results) == 4
    assert all(isinstance(r, Recurse2D) for r in results)
    recurse_results = [r for r in results if isinstance(r, Recurse2D)]

    # Both kernel variants appear
    assert any(isinstance(r.a, TIKernel2D) and r.a.data == [[1.0, -1.0]] for r in recurse_results)
    assert any(isinstance(r.a, TIKernel2D) and r.a.data == [[-1.0, 1.0]] for r in recurse_results)
    # Both signal variants appear
    assert any(isinstance(r.g, Var) and r.g.name == "input" for r in recurse_results)
    assert any(isinstance(r.g, Var) and r.g.name == "input_alt" for r in recurse_results)


def test_preorder_cartesian_product_recurse2d():
    """Preorder also computes the Cartesian product when a transform branches on each child.

    Same setup as the postorder test: 2 kernel variants × 2 signal variants = 4 results.
    """
    with with_types(Type.Arith, ElementType.Float):
        k = TIKernel2D([[1.0, -1.0]])
        g = Var("input")
        expr2d = Recurse2D(k, g)
        results = Preorder(BranchingTransform()).apply_signal2d(expr2d)

    assert len(results) == 4
    assert all(isinstance(r, Recurse2D) for r in results)
    recurse_results = [r for r in results if isinstance(r, Recurse2D)]
    assert any(isinstance(r.a, TIKernel2D) and r.a.data == [[1.0, -1.0]] for r in recurse_results)
    assert any(isinstance(r.a, TIKernel2D) and r.a.data == [[-1.0, 1.0]] for r in recurse_results)
    assert any(isinstance(r.g, Var) and r.g.name == "input" for r in recurse_results)
    assert any(isinstance(r.g, Var) and r.g.name == "input_alt" for r in recurse_results)


# ---------------------------------------------------------------------------
# Deep traversal into TVKernel2D / TVKernel entries
# ---------------------------------------------------------------------------


def test_traversal_deep_into_tvkernel2d():
    """Traversal descends into Convolve entries nested inside a TVKernel2D.

    Structure:
        Convolve2D(TVKernel2D([[Convolve(TIKernel([1,-1]), Var("x"))]]), Var2D("img"))

    BranchingTransform branches TIKernel → 2 variants and Var → 2 variants.
    The inner Convolve therefore has 2×2 = 4 variants, giving 4 top-level results.
    Each result must be a Convolve2D whose kernel is a TVKernel2D containing a Convolve
    with the expected TIKernel and Var variants.
    """
    with with_types(Type.Arith, ElementType.Float):
        inner_k = TIKernel([1.0, -1.0])
        v = Var("x")
        inner = Convolve(inner_k, v)
        tvk2d = TVKernel2D([[inner]])
        expr = Convolve2D(tvk2d, Var2D("img"))

        for traversal in [Preorder, Postorder]:
            results = traversal(BranchingTransform()).apply_signal(expr)

            assert len(results) == 4, f"{traversal.__name__}: expected 4, got {len(results)}"
            assert all(isinstance(r, Convolve2D) for r in results)

            # Extract the TIKernel data from the single entry of each TVKernel2D
            kernel_datas = set()
            var_names = set()
            for r in results:
                assert isinstance(r, Convolve2D)
                assert isinstance(r.a, TVKernel2D)
                entry = r.a.data[0][0]
                assert isinstance(entry, Convolve)
                assert isinstance(entry.a, TIKernel)
                kernel_datas.add(tuple(entry.a.data))
                assert isinstance(entry.b, Var)
                var_names.add(entry.b.name)

            assert kernel_datas == {(1.0, -1.0), (-1.0, 1.0)}
            assert var_names == {"x", "x_alt"}


def test_traversal_deep_into_tvkernel():
    """Traversal descends into Convolve entries nested inside a 1D TVKernel.

    Structure:
        Convolve(TVKernel([Convolve(TIKernel([1,-1]), Var("x"))]), Impulse())

    BranchingTransform branches TIKernel → 2 variants and Var → 2 variants.
    Impulse does not branch. The inner Convolve gets 2×2 = 4 variants, giving 4 results.
    """
    with with_types(Type.Arith, ElementType.Float):
        inner_k = TIKernel([1.0, -1.0])
        v = Var("x")
        inner = Convolve(inner_k, v)
        tvk = TVKernel([inner])
        expr = Convolve(tvk, Impulse(1.0))

        for traversal in [Preorder, Postorder]:
            results = traversal(BranchingTransform()).apply_signal(expr)

            assert len(results) == 4, f"{traversal.__name__}: expected 4, got {len(results)}"
            assert all(isinstance(r, Convolve) for r in results)

            kernel_datas = set()
            var_names = set()
            for r in results:
                assert isinstance(r, Convolve)
                assert isinstance(r.a, TVKernel)
                entry = r.a.data[0]
                assert isinstance(entry, Convolve)
                assert isinstance(entry.a, TIKernel)
                kernel_datas.add(tuple(entry.a.data))
                assert isinstance(entry.b, Var)
                var_names.add(entry.b.name)

            assert kernel_datas == {(1.0, -1.0), (-1.0, 1.0)}
            assert var_names == {"x", "x_alt"}
