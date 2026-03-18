from functools import reduce

from vecrec.expr import TIKernel, Type, with_types
from vecrec.util import ElementType
from vecrec.factorize import factorize_polynomial


def test_example_1():
    # x^2 - 5x + 6 = (x - 2)(x - 3)
    with with_types(Type.Arith, ElementType.Float):
        res = factorize_polynomial([1, -5, 6], ElementType.Float)
        assert res == [TIKernel([1.0, -3.0]), TIKernel([1.0, -2.0])]


def test_example_2():
    # x^3 - 6x^2 + 11x - 6 = (x - 1)(x - 2)(x - 3)
    with with_types(Type.Arith, ElementType.Float):
        res = factorize_polynomial([1, -6, 11, -6], ElementType.Float)
        print(res)
        assert res == [TIKernel([1.0, -3.0]), TIKernel([1.0, -2.0]), TIKernel([1.0, -1.0])]


def test_example_3():
    # x^2 + 1 => complex roots i and -i => second-order factor x^2 + 1
    with with_types(Type.Arith, ElementType.Float):
        res = factorize_polynomial([1, 0, 1], ElementType.Float)
        assert res == [TIKernel([1.0, 0.0, 1.0])]


def test_example_4():
    # x^3 - 1 = (x - 1)(x^2 + x + 1)
    with with_types(Type.Arith, ElementType.Float):
        res = factorize_polynomial([1, 0, 0, -1], ElementType.Float)
        assert res == [TIKernel([1.0, -1.0]), TIKernel([1.0, 1.0, 1.0])]


def test_example_5():
    # 2x^2 - 8 => 2*(x-2)(x+2) => leading coef distributed
    res = factorize_polynomial([2, 0, -8], ElementType.Float)
    # one of the first-order factors should have a leading coefficient 2
    assert any(abs(f[0] - 2.0) < 1e-8 for f in res if len(f) == 2) or any(
        abs(f[0] - 2.0) < 1e-8 for f in res if len(f) == 3
    )


def test_factors_reconstruct_original():
    """Product of factors must equal the original kernel (leading-coeff bug regression test).

    Previously, the leading coefficient was only applied to factor[0] instead of
    all coefficients, so e.g. 0.707*(x+1) became [0.707, 1] instead of [0.707, 0.707],
    and the product of factors would not equal the original polynomial.
    """
    with with_types(Type.Arith, ElementType.Float):
        cases = [
            [2., 0, -8.],                    # 2*(x-2)(x+2)
            [0.707, 0., 0., 0., -0.707],      # 0.707*(x^4-1), arises from Factorize2D + SVD scaling
            [3., 0., 3.],                     # 3*(x^2+1), non-unit leading coeff on second-order factor
        ]
        for coeffs in cases:
            factors = factorize_polynomial(coeffs, ElementType.Float)
            product = reduce(lambda a, b: a * b, factors)
            original = TIKernel(coeffs)
            for i, (got, expected) in enumerate(zip(product.data, original.data)):
                assert abs(got - expected) < 1e-6, (
                    f"coeffs={coeffs}: factor product[{i}]={got} != original[{i}]={expected}"
                )
