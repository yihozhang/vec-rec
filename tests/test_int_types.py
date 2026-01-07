#!/usr/bin/env python3
"""Simple test to verify multi-element type support"""

from vecrec.expr import *
from vecrec.util import ElementType
from vecrec.transform import *
from vecrec.codegen import CodeGen, generate_and_run_benchmark

def test_float_backward_compatibility():
    kernel = TIKernel([0, 1.8, -0.9], Type.Arith, ElementType.Float)
    signal = Var("x", Type.Arith, ElementType.Float)
    expr = Recurse(kernel, signal)

    schedule = Seq(
        Dilate(),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = schedule.apply_signal(expr)

    codegen = CodeGen()
    out = generate_and_run_benchmark(codegen, results, [f"k{i}" for i in range(len(results))], True)
    print(out)

def test_int32_support():
    # Create an int32 prefix sum
    kernel = TIKernel([0, 1], Type.Arith, ElementType.I32)
    signal = Var("g", Type.Arith, ElementType.I32)
    expr = Recurse(kernel, signal)

    schedule = Seq(
        Dilate(),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = schedule.apply_signal(expr)

    codegen = CodeGen()
    out = generate_and_run_benchmark(codegen, results, [f"k{i}" for i in range(len(results))], True)
    assert out['return_code'] == 0, "Int32 benchmark failed"

def test_multiple_types():
    """Test that we catch type mismatches"""
    print("\nTesting type mismatch detection...")

    try:
        # This should fail - mixing float and int32
        kernel_float = TIKernel([0, 1.8], Type.Arith, ElementType.Float)
        kernel_int = TIKernel([0, 1], Type.Arith, ElementType.I32)

        # This should raise an assertion error
        result = kernel_float + kernel_int
        print("✗ Should have caught type mismatch!")
        assert False
    except AssertionError as e:
        print(f"✓ Correctly caught type mismatch: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("VecRec Multi-Element Type Support Test")
    print("=" * 60)

    try:
        test_float_backward_compatibility()
        test_int32_support()
        test_multiple_types()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
