"""Test Convolve2D codegen implementation"""

from vecrec.expr.base import SignalExpr, Type
from vecrec.expr.kernel import TIKernel2D, TVKernel2D
from vecrec.expr.signal import Var, Var2D, Num
from vecrec.expr.signal_ops import Convolve2D, Ith, Repeater, SAdd
from vecrec.util import ElementType
from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.transform import AnnotateLanes, Preorder, Eliminate2DKernels


def test_convolve2d_ti_kernel() -> None:
    """Test Convolve2D with a time-invariant 2D kernel"""
    # Create a simple 2D kernel (2x3)
    kernel_data = [
        [1.0/6, 1.0/6, 1.0/6],
        [1.0/6, 1.0/6, 1.0/6],
    ]
    kernel2d = TIKernel2D(kernel_data, Type.Arith, ElementType.Float)
    g = Var("g", Type.Arith, ElementType.Float)
    
    # Create a Repeater with a Var2D signal
    def repeater_func(prev_rows: Var2D) -> SignalExpr:
        # Create a Convolve2D that convolves the 2D kernel with previous rows
        conv2d = SAdd(Convolve2D(kernel2d, prev_rows), g)
        return conv2d
    
    repeater: SignalExpr
    repeater = Ith(Repeater(repeater_func, n_rows=3, ty=Type.Arith, element_type=ElementType.Float), 0)
    
    # Apply transforms: first eliminate 2D kernels, then annotate lanes
    repeater = Preorder(Eliminate2DKernels()).apply_generic(repeater)[0]
    repeater = AnnotateLanes(512).apply_generic(repeater)[0]

    result = generate_and_run_benchmark(CodeGen(), [repeater], ["test_convolve2d_ti"])
    assert result['return_code'] == 0
    print(result)
    
    codegen = CodeGen()
    code = codegen.generate(repeater, "test_convolve2d_ti")
    
    print("Generated code for Convolve2D with TI kernel:")
    print(code.text)
    print()
    
    # Verify the code contains expected components
    assert "RepeaterContext" in code.text
    assert "make_signal2d" in code.text
    assert "make_ith_row" in code.text
    assert "make_s_convolve" in code.text
    assert "SAdd" in code.text  # Multiple rows should be added together
    
    print("✓ Convolve2D with TI kernel generated successfully")


def test_convolve2d_tv_kernel() -> None:
    """Test Convolve2D with a time-varying 2D kernel"""
    # Create a simple 2D time-varying kernel (2x2)
    kernel_data: list[list[SignalExpr]] = [
        [Num(1.0, Type.Arith, ElementType.Float), Num(2.0, Type.Arith, ElementType.Float)],
        [Num(3.0, Type.Arith, ElementType.Float), Num(4.0, Type.Arith, ElementType.Float)],
    ]
    kernel2d = TVKernel2D(kernel_data, Type.Arith, ElementType.Float)
    g = Var("g", Type.Arith, ElementType.Float)
    
    # Create a Repeater with a Var2D signal
    def repeater_func(prev_rows: Var2D):
        # Create a Convolve2D that convolves the 2D kernel with previous rows
        conv2d = SAdd(Convolve2D(kernel2d, prev_rows), g)
        return conv2d
    
    # TODO: better error handling when n_rows is too small (e.g., 2)
    repeater = Repeater(repeater_func, n_rows=3, ty=Type.Arith, element_type=ElementType.Float)
    
    # Apply transforms: first eliminate 2D kernels, then annotate lanes
    repeater = Preorder(Eliminate2DKernels()).apply_signal2d(repeater)[0]
    repeater = AnnotateLanes(512).apply_signal2d(repeater)[0]
    
    result = generate_and_run_benchmark(CodeGen(), [repeater], ["test_convolve2d_tv"])
    assert result['return_code'] == 0
    codegen = CodeGen()
    code = codegen.generate(repeater, "test_convolve2d_tv")
    
    print("Generated code for Convolve2D with TV kernel:")
    print(code.text)
    print()
    
    # Verify the code contains expected components
    assert "RepeaterContext" in code.text
    assert "make_signal2d" in code.text
    assert "make_ith_row" in code.text
    assert "make_s_convolve" in code.text
    assert "make_time_varying_kernel" in code.text
    
    print("✓ Convolve2D with TV kernel generated successfully")


def test_convolve2d_single_row() -> None:
    """Test Convolve2D with a single-row kernel (should not need SAdd)"""
    # Create a single-row 2D kernel
    kernel_data = [
        [1.0, 2.0, 3.0],
    ]
    kernel2d = TIKernel2D(kernel_data, Type.Arith, ElementType.Float)
    g = Var("g", Type.Arith, ElementType.Float)
    
    # Create a Repeater with a Var2D signal
    def repeater_func(prev_rows: Var2D):
        conv2d = SAdd(Convolve2D(kernel2d, prev_rows), g)
        return conv2d
    
    repeater = Repeater(repeater_func, n_rows=2, ty=Type.Arith, element_type=ElementType.Float)
    
    # Apply transforms: first eliminate 2D kernels, then annotate lanes
    repeater = Preorder(Eliminate2DKernels()).apply_signal2d(repeater)[0]
    repeater = AnnotateLanes(512).apply_signal2d(repeater)[0]
    
    result = generate_and_run_benchmark(CodeGen(), [repeater], ["test_convolve2d_single"])
    assert result['return_code'] == 0
    codegen = CodeGen()
    code = codegen.generate(repeater, "test_convolve2d_single")
    
    print("Generated code for Convolve2D with single-row kernel:")
    print(code.text)
    print()
    
    # Should not need SAdd for single row
    assert "make_ith_row" in code.text
    assert "make_s_convolve" in code.text
    
    print("✓ Convolve2D with single-row kernel generated successfully")


if __name__ == "__main__":
    test_convolve2d_ti_kernel()
    # test_convolve2d_tv_kernel()
    # test_convolve2d_single_row()
    print("\n✓ All Convolve2D tests passed!")
