
from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.transform import ApplyParallel, ConstantFold, Delay, Dilate, ApplySequence, Preorder, Try
from vecrec.expr import Recurse, TIKernel, Type, Var


def test_generate_and_run():
    # Define a recursive filter kernel
    kernel = TIKernel([0, 1.8, -0.9], Type.Arith)
    signal = Var("x", Type.Arith)
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
    
    # Generate, compile, and run benchmark
    print("Generating, compiling, and running benchmark...")
    result = generate_and_run_benchmark(
        codegen=codegen,
        exprs=[expr, results[0], results[1]],
        kernel_names=["original", "dilated", "dilate_and_delayed"],
        header_path="output.h",
        include_correctness_check=True,
        input_size=1 << 16,  # Smaller size for quick testing
        warmup_iterations=5,
        benchmark_iterations=10,
    )
    
    if result['success']:
        print("\n✓ Benchmark completed successfully!\n")
        print("Output:")
        print(result['output'])
    else:
        print("\n✗ Benchmark failed!\n")
        print("Error:", result['error'])
        if result['output']:
            print("Output:", result['output'])
        assert False, "Benchmark failed"

