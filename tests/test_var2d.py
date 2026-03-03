"""Tests for Var2D - user-provided 2D signal input."""

import numpy as np

from vecrec.expr.base import SignalExpr, Type, with_types
from vecrec.expr.kernel import TIKernel2D
from vecrec.expr.signal import Var, Var2D
from vecrec.expr.signal_ops import Convolve2D, Ith, SAdd
from vecrec.util import ElementType
from vecrec.codegen import (
    CodeGen,
    generate_and_run_benchmark,
    generate_kernel_executable,
)
from vecrec.transform import AnnotateLanes, Preorder, Eliminate2DKernels, Try


# expr = Convolve2D(TIKernel2D([[1.], [1.]]), Var2D("img", 2))
def test_var2d_ith_compiles_and_runs() -> None:
    """Var2D used via Ith should compile and run without errors."""
    with with_types(Type.Arith, ElementType.Float):
        var2d = Var2D("img", 2)
        expr: SignalExpr = Ith(var2d, 0)
        expr = AnnotateLanes(512).apply_generic(expr)[0]

    result = generate_and_run_benchmark(CodeGen(), [expr])
    assert result["return_code"] == 0, result.get("error")

    runner = generate_kernel_executable(CodeGen(), expr)
    input = np.random.rand(4, 1024).astype(np.float32)
    output, _ = runner.run(input)
    assert np.allclose(output, input.flatten(), atol=1e-6)


def test_var2d_generates_input_signal2d() -> None:
    """Codegen for Var2D should emit InputSignal2D and the stride parameter."""
    with with_types(Type.Arith, ElementType.Float):
        var2d = Var2D("img", 2)
        expr: SignalExpr = Ith(var2d, 1)
        expr = AnnotateLanes(512).apply_generic(expr)[0]

    runner = generate_kernel_executable(CodeGen(), expr)
    input = np.random.rand(4, 1024).astype(np.float32)
    output, _ = runner.run(input)
    assert np.allclose(output[1024:], input[:-1].flatten(), atol=1e-6)


def test_var2d_with_convolve2d_compiles_and_runs() -> None:
    """Var2D used as input to Convolve2D should compile and run end-to-end.

    Convolve2D(kernel2d, signal2d) is already a SignalExpr (1D output), so no
    Ith wrapper is needed. Eliminate2DKernels expands it into 1D convolutions,
    one per kernel row, each reading from Ith(var2d, row_index).
    """
    kernel_data = [
        [0.5, 0.0, -0.5],
        [0.5, 0.0, -0.5],
    ]

    with with_types(Type.Arith, ElementType.Float):
        kernel2d = TIKernel2D(kernel_data)
        var2d = Var2D("img", 2)

        expr: SignalExpr = Convolve2D(kernel2d, var2d)
        expr = Preorder(Try(Eliminate2DKernels())).apply_generic(expr)[0]
        expr = AnnotateLanes(512).apply_generic(expr)[0]

    result = generate_and_run_benchmark(CodeGen(), [expr])
    assert result["return_code"] == 0, result.get("error")

    runner = generate_kernel_executable(CodeGen(), expr)

    input = np.random.rand(4, 1024).astype(np.float32)
    output, _ = runner.run(input)

    def causal_conv_1d(row: np.ndarray, taps: list[float]) -> np.ndarray:
        result = np.zeros_like(row)
        for delay, weight in enumerate(taps):
            if weight == 0.0:
                continue
            result[delay:] += np.float32(weight) * row[: row.shape[0] - delay]
        return result

    expected_rows = []
    zero_row = np.zeros_like(input[0])
    for row_idx in range(input.shape[0]):
        current_row = input[row_idx]
        prev_row = input[row_idx - 1] if row_idx > 0 else zero_row
        expected_row = causal_conv_1d(current_row, kernel_data[0]) + causal_conv_1d(
            prev_row, kernel_data[1]
        )
        expected_rows.append(expected_row)

    expected_output = np.concatenate(expected_rows)
    assert np.allclose(output, expected_output, atol=1e-6)


def test_var2d_alongside_var1d() -> None:
    """Var2D and Var can coexist in the same expression."""
    with with_types(Type.Arith, ElementType.Float):
        var1d = Var("bias")
        var2d = Var2D("img", 2)
        expr: SignalExpr = SAdd(Ith(var2d, 0), var1d)
        expr = AnnotateLanes(512).apply_generic(expr)[0]

    codegen = CodeGen()
    code = codegen.generate(expr, "test_var2d_var1d")

    assert "const float *bias" in code.text
    assert "const float *img" in code.text
    assert "int img_stride" in code.text

    result = generate_and_run_benchmark(CodeGen(), [expr])
    assert result["return_code"] == 0, result.get("error")

    runner = generate_kernel_executable(CodeGen(), expr)

    H, W = 4, 1024
    img = np.random.rand(H, W).astype(np.float32)
    bias = np.random.rand(H * W).astype(np.float32)

    output, _ = runner.run({"img": img, "bias": bias})
    expected = img.flatten() + bias
    assert np.allclose(output, expected, atol=1e-6)


def test_var2d_kernel_executable_ith_row() -> None:
    """Verify Ith(Var2D, 0) produces the expected row-copy output.

    With causal (look-behind) semantics, all H rows produce output.
    d[0] is the current row, so output[k*1024 : (k+1)*1024] == img_in[k]
    for k = 0 .. H-1.
    """
    n_rows = 2
    H = 5  # rows in input
    W = 1024  # matches the default row_size of generate_kernel_executable

    with with_types(Type.Arith, ElementType.Float):
        var2d = Var2D("img", n_rows)
        expr: SignalExpr = Ith(var2d, 0)
        expr = AnnotateLanes(512).apply_generic(expr)[0]

    runner = generate_kernel_executable(CodeGen(), expr)

    rng = np.random.default_rng(42)
    img_in = rng.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32)

    output, elapsed_us = runner.run({"img": img_in})

    # All H rows produce output
    N = H * W
    assert output.shape == (N,), f"Expected shape ({N},), got {output.shape}"
    assert output.dtype == np.float32
    assert elapsed_us >= 0

    # Each 1024-element chunk of output corresponds to one row of img_in
    expected = img_in.flatten()
    assert np.allclose(
        output, expected, atol=1e-6
    ), f"max abs error: {np.abs(output - expected).max()}"
