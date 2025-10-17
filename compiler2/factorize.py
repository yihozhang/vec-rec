import numpy as np
from compiler2.expr import TIKernel
import pytest
from typing import List, Tuple

__all__ = ["factorize_polynomial"]

def factorize_polynomial(coefficients: List[float] | np.ndarray, tolerance: float = 1e-10) -> List[TIKernel]:
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
        for j, other_root in enumerate(complex_roots[i+1:], start=i+1):
            if j in used:
                continue
            
            # Check if they are conjugates
            if abs(root.real - other_root.real) < tolerance and \
               abs(root.imag + other_root.imag) < tolerance:
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
        second_order.append([1.0, -2*a, a*a + b*b])
    
    first_order = sorted(first_order)
    second_order = sorted(second_order)
    
    # Adjust for leading coefficient
    # We need to distribute the leading coefficient among the factors
    if len(first_order) > 0:
        first_order[-1][0] *= leading_coef
    elif len(second_order) > 0:
        second_order[-1] *= leading_coef
    
    first_order = [TIKernel(f) for f in first_order]
    second_order = [TIKernel(f) for f in second_order]
    
    return first_order + second_order


def test_example_1():
    # x^2 - 5x + 6 = (x - 2)(x - 3)
    res = factorize_polynomial([1, -5, 6])
    assert res == [TIKernel([1.0, -3.0]), TIKernel([1.0, -2.0])]


def test_example_2():
    # x^3 - 6x^2 + 11x - 6 = (x - 1)(x - 2)(x - 3)
    res = factorize_polynomial([1, -6, 11, -6])
    print(res)
    assert res == [TIKernel([1.0, -3.0]), TIKernel([1.0, -2.0]), TIKernel([1.0, -1.0])]
    


def test_example_3():
    # x^2 + 1 => complex roots i and -i => second-order factor x^2 + 1
    res = factorize_polynomial([1, 0, 1])
    assert res == [TIKernel([1.0, 0.0, 1.0])]


def test_example_4():
    # x^3 - 1 = (x - 1)(x^2 + x + 1)
    res = factorize_polynomial([1, 0, 0, -1])
    assert res == [TIKernel([1.0, -1.0]), TIKernel([1.0, 1.0, 1.0])]


def test_example_5():
    # 2x^2 - 8 => 2*(x-2)(x+2) => leading coef distributed
    res = factorize_polynomial([2, 0, -8])
    # one of the first-order factors should have a leading coefficient 2
    assert any(abs(f[0] - 2.0) < 1e-8 for f in res if len(f) == 2) or any(abs(f[0] - 2.0) < 1e-8 for f in res if len(f) == 3)
