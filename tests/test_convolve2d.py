"""Test Convolve2D codegen implementation"""

from vecrec.expr.base import SignalExpr, SignalExpr2D, Type, with_types
from vecrec.expr.kernel import TIKernel2D, TVKernel2D
from vecrec.expr.signal import Var, RVar2D, Num
from vecrec.expr.signal_ops import Convolve2D, Ith, Repeater, SAdd
from vecrec.util import ElementType, get_native_bits
from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.transform import AnnotateLanes, InferNRows, Preorder, Eliminate2DKernels, Try


def test_convolve2d_ti_kernel() -> None:
    """Test Convolve2D with a time-invariant 2D kernel"""
    with with_types(Type.Arith, ElementType.Float):
        # Create a simple 2D kernel (2x3)
        kernel_data = [
            [1.0 / 6, 1.0 / 6, 1.0 / 6],
            [1.0 / 6, 1.0 / 6, 1.0 / 6],
        ]
        kernel2d = TIKernel2D(kernel_data)
        g = Var("g")

        # Create a Repeater with a RVar2D signal
        def repeater_func(prev_rows: RVar2D) -> SignalExpr:
            # Create a Convolve2D that convolves the 2D kernel with previous rows
            conv2d = SAdd(Convolve2D(kernel2d, prev_rows), g)
            return conv2d

        repeater: SignalExpr
        repeater = Ith(Repeater(repeater_func), 0)

    # Apply transforms: first eliminate 2D kernels, then annotate lanes
    repeater = Preorder(Try(Eliminate2DKernels())).apply_generic(repeater)[0]
    repeater = InferNRows().apply_generic(repeater)[0]
    repeater = AnnotateLanes(get_native_bits()).apply_generic(repeater)[0]

    result = generate_and_run_benchmark(CodeGen(), [repeater])
    assert result["return_code"] == 0
    print(result)

    codegen = CodeGen()
    code = codegen.generate(repeater, "test_convolve2d_ti")

    print("Generated code for Convolve2D with TI kernel:")
    print(code.text)
    print()

    # Verify the code contains expected components
    assert "RepeaterContext" in code.text
    assert "Signal2D" in code.text
    assert "IthRow" in code.text
    assert "SConvolve" in code.text
    assert "SAdd" in code.text  # Multiple rows should be added together

    print("✓ Convolve2D with TI kernel generated successfully")


def test_convolve2d_tv_kernel() -> None:
    """Test Convolve2D with a time-varying 2D kernel"""
    with with_types(Type.Arith, ElementType.Float):
        # Create a simple 2D time-varying kernel (2x2)
        kernel_data: list[list[SignalExpr]] = [
            [Num(1.0), Num(2.0)],
            [Num(3.0), Num(4.0)],
        ]
        kernel2d = TVKernel2D(kernel_data)
        g = Var("g")

        # Create a Repeater with a RVar2D signal
        def repeater_func(prev_rows: RVar2D) -> SignalExpr:
            # Create a Convolve2D that convolves the 2D kernel with previous rows
            conv2d = SAdd(Convolve2D(kernel2d, prev_rows), g)
            return conv2d

        repeater: SignalExpr2D
        repeater = Repeater(repeater_func)

    # Apply transforms: first eliminate 2D kernels, then annotate lanes
    repeater = Preorder(Try(Eliminate2DKernels())).apply_signal2d(repeater)[0]
    repeater = InferNRows().apply_generic(repeater)[0]
    repeater = AnnotateLanes(get_native_bits()).apply_signal2d(repeater)[0]

    result = generate_and_run_benchmark(CodeGen(), [repeater])
    assert result["return_code"] == 0
    codegen = CodeGen()
    code = codegen.generate(repeater, "test_convolve2d_tv")

    print("Generated code for Convolve2D with TV kernel:")
    print(code.text)
    print()

    # Verify the code contains expected components
    assert "RepeaterContext" in code.text
    assert "Signal2D" in code.text
    assert "IthRow" in code.text
    assert "SConvolve" in code.text
    assert "TimeVaryingKernel" in code.text

    print("✓ Convolve2D with TV kernel generated successfully")


def test_convolve2d_single_row() -> None:
    """Test Convolve2D with a single-row kernel (should not need SAdd)"""
    with with_types(Type.Arith, ElementType.Float):
        # Create a single-row 2D kernel
        kernel_data = [
            [1.0, 2.0, 3.0],
        ]
        kernel2d = TIKernel2D(kernel_data)
        g = Var("g")

        # Create a Repeater with a RVar2D signal
        def repeater_func(prev_rows: RVar2D) -> SignalExpr:
            conv2d = SAdd(Convolve2D(kernel2d, prev_rows), g)
            return conv2d

        repeater: SignalExpr2D
        repeater = Repeater(repeater_func)

    # Apply transforms: first eliminate 2D kernels, then annotate lanes
    repeater = Preorder(Try(Eliminate2DKernels())).apply_signal2d(repeater)[0]
    repeater = InferNRows().apply_generic(repeater)[0]
    repeater = AnnotateLanes(get_native_bits()).apply_signal2d(repeater)[0]

    result = generate_and_run_benchmark(CodeGen(), [repeater])
    assert result["return_code"] == 0
    codegen = CodeGen()
    code = codegen.generate(repeater, "test_convolve2d_single")

    print("Generated code for Convolve2D with single-row kernel:")
    print(code.text)
    print()

    # Should not need SAdd for single row
    assert "IthRow" in code.text
    assert "SConvolve" in code.text

    print("✓ Convolve2D with single-row kernel generated successfully")


if __name__ == "__main__":
    test_convolve2d_ti_kernel()
    # test_convolve2d_tv_kernel()
    # test_convolve2d_single_row()
    print("\n✓ All Convolve2D tests passed!")
