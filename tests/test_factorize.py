from vecrec.expr import TIKernel, Type
from vecrec.factorize import factorize_polynomial, ElementType
from vecrec.util import ElementType


def test_example_1():
    # x^2 - 5x + 6 = (x - 2)(x - 3)
    res = factorize_polynomial([1, -5, 6], ElementType.Float)
    assert res == [TIKernel([1.0, -3.0], Type.Arith, ElementType.Float), TIKernel([1.0, -2.0], Type.Arith, ElementType.Float)]


def test_example_2():
    # x^3 - 6x^2 + 11x - 6 = (x - 1)(x - 2)(x - 3)
    res = factorize_polynomial([1, -6, 11, -6], ElementType.Float)
    print(res)
    assert res == [TIKernel([1.0, -3.0], Type.Arith, ElementType.Float), TIKernel([1.0, -2.0], Type.Arith, ElementType.Float), TIKernel([1.0, -1.0], Type.Arith, ElementType.Float)]


def test_example_3():
    # x^2 + 1 => complex roots i and -i => second-order factor x^2 + 1
    res = factorize_polynomial([1, 0, 1], ElementType.Float)
    assert res == [TIKernel([1.0, 0.0, 1.0], Type.Arith, ElementType.Float)]


def test_example_4():
    # x^3 - 1 = (x - 1)(x^2 + x + 1)
    res = factorize_polynomial([1, 0, 0, -1], ElementType.Float)
    assert res == [TIKernel([1.0, -1.0], Type.Arith, ElementType.Float), TIKernel([1.0, 1.0, 1.0], Type.Arith, ElementType.Float)]


def test_example_5():
    # 2x^2 - 8 => 2*(x-2)(x+2) => leading coef distributed
    res = factorize_polynomial([2, 0, -8], ElementType.Float)
    # one of the first-order factors should have a leading coefficient 2
    assert any(abs(f[0] - 2.0) < 1e-8 for f in res if len(f) == 2) or any(
        abs(f[0] - 2.0) < 1e-8 for f in res if len(f) == 3
    )
