import pytest
from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.transform import AnnotateLanes, PushDownConvertLanes, Seq, Preorder, Try, ConstantFold
from vecrec.expr import Recurse, TIKernel, Type, Var, Convolve
from vecrec.util import ElementType

@pytest.mark.parametrize("kernel_type, element_type", [
    (Type.TropMax, ElementType.Float),
    (Type.TropMin, ElementType.Float),
    (Type.TropMax, ElementType.I32),
    (Type.TropMax, ElementType.I64),
])
def test_tropical(kernel_type: Type, element_type: ElementType) -> None:
    kernel = TIKernel([1.0, 1.0], kernel_type, element_type)
    kernel2 = TIKernel([kernel_type.zero(), -1.0], kernel_type, element_type)
    signal = Var("x", kernel_type, element_type)
    expr = Recurse(kernel2, Convolve(kernel, signal))

    schedule = Seq(
        Preorder(Try(ConstantFold)),
        AnnotateLanes(256),
        PushDownConvertLanes(),
    )
    results = schedule.apply_signal(expr)

    codegen = CodeGen()
    result = generate_and_run_benchmark(
        codegen=codegen,
        exprs=results,
        include_correctness_check=True,
        input_size=1 << 16,
        warmup_iterations=5,
        benchmark_iterations=10,
    )

    if result['return_code'] == 0:
        print(f"✓ {kernel_type.name} {element_type.name} test passed")
        print(f"Output: {result['output']}")
    else:
        print(f"✗ {kernel_type.name} {element_type.name} test failed: {result['error']}")
        if result['output']:
            print(f"Output: {result['output']}")
        assert False, f"{kernel_type.name} {element_type.name} test failed: {result['error']}"

# TODO: does not check the correctness of the output. Just that it compiles.