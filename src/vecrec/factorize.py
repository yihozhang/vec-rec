import numpy as np
from vecrec.expr import TIKernel, Type
from vecrec.expr.kernel import TIKernel2D
from vecrec.util import ElementType
from typing import List, Optional, Tuple
import numpy.typing as npt

__all__ = ["factorize_polynomial", "factorize_kernel2d"]


def factorize_polynomial(
    coefficients: List[float] | npt.NDArray[np.float64],
    element_type: ElementType,
    tolerance: float = 1e-10,
) -> List[TIKernel]:
    """
    Factorize a polynomial into products of first-order and second-order factors.

    Args:
        coefficients: List of coefficients [a_n, a_(n-1), ..., a_1, a_0]
                     representing a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0
        tolerance: Tolerance for determining if a root is real or if imaginary part is negligible

    Returns:
        Tuple of (first_order_factors, second_order_factors)
        - first_order_factors: List of [a, b] representing (ax + b)
        - second_order_factors: List of [a, b, c] representing (ax^2 + bx + c)
    """

    # Handle edge cases
    if len(coefficients) < 2:
        raise ValueError("Polynomial must have at least degree 1")

    # Remove leading zeros
    while len(coefficients) > 1 and abs(coefficients[0]) < tolerance:
        coefficients = coefficients[1:]

    # Find roots using numpy
    roots = np.roots(coefficients)

    # Separate real and complex roots
    real_roots = []
    complex_roots = []

    for root in roots:
        if abs(root.imag) < tolerance:
            real_roots.append(root.real)
        else:
            complex_roots.append(root)

    # Pair complex conjugates
    complex_pairs = []
    used = set()

    for i, root in enumerate(complex_roots):
        if i in used:
            continue

        # Find conjugate pair
        for j, other_root in enumerate(complex_roots[i + 1 :], start=i + 1):
            if j in used:
                continue

            # Check if they are conjugates
            if (
                abs(root.real - other_root.real) < tolerance
                and abs(root.imag + other_root.imag) < tolerance
            ):
                complex_pairs.append((root, other_root))
                used.add(i)
                used.add(j)
                break

    # Build first-order factors: (x - r) = (1*x + (-r))
    first_order = []
    leading_coef = coefficients[0]

    for root in real_roots:
        first_order.append([1.0, -root])

    # Build second-order factors from complex conjugate pairs
    # If roots are (a + bi) and (a - bi), then:
    # (x - (a+bi))(x - (a-bi)) = x^2 - 2ax + (a^2 + b^2)
    second_order = []

    for root1, _root2 in complex_pairs:
        a = root1.real
        b = root1.imag
        # Coefficients: [1, -2a, a^2 + b^2]
        second_order.append([1.0, -2 * a, a * a + b * b])

    first_order = sorted(first_order)
    second_order = sorted(second_order)

    # Adjust for leading coefficient by scaling all coefficients of one factor.
    # Only multiplying factor[0] is wrong: [1,1]*c → [c,1] but we need [c,c].
    if len(first_order) > 0:
        first_order[-1] = [c * leading_coef for c in first_order[-1]]
    elif len(second_order) > 0:
        second_order[-1] = [c * leading_coef for c in second_order[-1]]

    return [TIKernel(f, Type.Arith, element_type) for f in first_order] + [TIKernel(f, Type.Arith, element_type) for f in second_order]


def factorize_kernel2d(
    kernel: TIKernel2D,
    tolerance: float = 1e-10,
) -> Optional[Tuple[TIKernel, TIKernel2D]]:
    """
    Try to factorize a 2D TI kernel into separable 1D kernels.

    A 2D kernel A is separable (rank-1) when its coefficient matrix can be written
    as an outer product: A[r][c] = col_vec[r] * row_vec[c]. This is detected via SVD.

    If separable, returns (row_kernel, col_kernel_2d) where:
    - row_kernel = TIKernel(row_vec)  -- applies along the column (horizontal) axis
    - col_kernel_2d = TIKernel2D([[col_vec[0]], [col_vec[1]], ...])  -- single-column 2D kernel

    So Convolve2D(A, S) == Convolve(row_kernel, Convolve2D(col_kernel_2d, S)).

    Returns None if the kernel is not separable within tolerance.
    """
    n_rows = kernel.n_rows()
    if n_rows == 0:
        return None

    n_cols = max((len(row) for row in kernel.data), default=0)
    if n_cols == 0:
        return None

    # Build the dense coefficient matrix
    A = np.zeros((n_rows, n_cols))
    for r, row in enumerate(kernel.data):
        for c, v in enumerate(row):
            A[r, c] = v

    # SVD-based rank-1 check
    U, s_vals, Vt = np.linalg.svd(A, full_matrices=False)

    if len(s_vals) == 0 or abs(s_vals[0]) < tolerance:
        return None

    # Rank-1: all singular values except the first must be negligible
    if len(s_vals) >= 2 and abs(s_vals[1]) > tolerance * abs(s_vals[0]):
        return None

    col_vec = (U[:, 0] * s_vals[0]).tolist()
    row_vec = Vt[0, :].tolist()

    # Verify reconstruction error
    reconstructed = np.outer(col_vec, row_vec)
    scale = max(float(np.max(np.abs(A))), 1.0)
    if float(np.max(np.abs(A - reconstructed))) > tolerance * scale:
        return None

    row_kernel = TIKernel(row_vec, kernel.ty, kernel.element_type)
    col_kernel_2d = TIKernel2D([[v] for v in col_vec], kernel.ty, kernel.element_type)
    return row_kernel, col_kernel_2d
