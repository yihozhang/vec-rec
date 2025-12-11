#!/usr/bin/env python3
"""
Example demonstrating the automatic benchmark generation feature.

This script shows how to:
1. Generate kernel code for signal processing expressions
2. Automatically generate a C++ benchmark program
3. Optionally include correctness checking
"""

from vecrec import CodeGen, instantiate_kernels, generate_benchmark
from vecrec.codegen import generate_and_run_benchmark
from vecrec.transform import ApplyParallel, ConstantFold, Delay, Dilate, ApplySequence, Preorder, Try
from vecrec.expr import Recurse, TIKernel, Var


def main():
    # Define a recursive filter kernel
    kernel = TIKernel([0, 1.8, -0.9])
    signal = Var("x")
    expr = Recurse(kernel, signal)
    
    # Apply transformations to optimize the kernel
    transforms = [
        Dilate(),
        Dilate(),
        ApplyParallel([Dilate(), Delay()]),
        Preorder(Try(ConstantFold)),
    ]
    results = ApplySequence(transforms).apply_signal(expr)
    
    # Create code generator
    codegen = CodeGen(256)  # 256-bit SIMD
    
    # Generate code for original and transformed kernels
    original = codegen.generate(expr, "original")
    dilated = codegen.generate(results[0], "dilated")
    dilate_and_delayed = codegen.generate(results[1], "dilate_and_delayed")
    
    codes = [original, dilated, dilate_and_delayed]
    exprs_list = [expr, results[0], results[1]]
    kernel_names = ["original", "dilated", "dilate_and_delayed"]
    
    # Generate kernel header file
    instantiate_kernels("output.h", codes)
    
    # Generate benchmark program with correctness checking
    result = generate_and_run_benchmark(
        codegen=codegen,
        exprs=exprs_list,
        kernel_names=kernel_names,
        header_path="output.h",
        include_correctness_check=True
    )
    print(result['output'])

if __name__ == "__main__":
    main()
