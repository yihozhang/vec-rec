
from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.transform import AnnotateLanes, Any, ConstantFold, Delay, Dilate, Optional, PushDownConvertLanes, Seq, Preorder, Try
from vecrec.expr import Recurse, TIKernel, Type, Var
from vecrec.util import ElementType


def test_generate_and_run():
    # Define a recursive filter kernel
    kernel = TIKernel([0, 1.8, -0.9], Type.Arith, ElementType.Float)
    signal = Var("x", Type.Arith, ElementType.Float)
    expr = Recurse(kernel, signal)
    
    # Apply transformations to optimize the kernel
    schedule = Seq(
        Optional(
            Seq(
                Dilate(),
                Dilate(),
                Any(Dilate(), Delay()),
                Preorder(Try(ConstantFold)),
            )
        ),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )
    results = schedule.apply_signal(expr)
    
    # Create code generator
    codegen = CodeGen()
    
    # Generate, compile, and run benchmark
    print("Generating, compiling, and running benchmark...")
    result = generate_and_run_benchmark(
        codegen=codegen,
        exprs=results[:3],
        kernel_names=["original", "dilated", "dilate_and_delayed"],
        include_correctness_check=True,
        input_size=1 << 16,  # Smaller size for quick testing
        warmup_iterations=5,
        benchmark_iterations=10,
    )
    
    if result['return_code'] == 0:
        print("\n✓ Benchmark completed successfully!\n")
        print("Output:")
        print(result['output'])
    else:
        print("\n✗ Benchmark failed!\n")
        print("Error:", result['error'])
        if result['output']:
            print("Output:", result['output'])
        assert False, "Benchmark failed"

