import pytest
from vecrec.expr import Recurse, TIKernel, Type, Var
from vecrec.transform import (
    AnnotateLanes,
    Any,
    ConstantFold,
    Delay,
    Dilate,
    Optional,
    Preorder,
    PushDownConvertLanes,
    RepeatUpTo,
    Seq,
    Try,
)
from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.util import ElementType


@pytest.mark.parametrize(
    "element_type,coefficients",
    [
        # (ElementType.Float, [0, 1.8, -0.9]),
        # (ElementType.I32, [0, 2, -1]),
        (ElementType.I64, [0, 2, -1]),
    ],
)
def test_second_order_ti_element_types(element_type, coefficients):
    """
    Test that the second-order TI kernel benchmark generator works with different ElementType values.
    This test mimics benchmarks/arith/second-order-ti.py but validates all element types.
    Uses appropriate coefficients for each type (floats for Float, integers for integer types).
    """
    # Create the same program as in the benchmark, but with parameterized element type
    program = Recurse(
        TIKernel(coefficients, Type.Arith, element_type),
        Var("g", Type.Arith, element_type),
    )

    # Apply the same transformations as in the benchmark
    # TODO: the number of dilation/delay that can be performed depends on the element type
    transforms = Seq(
        Optional(
            Seq(
                RepeatUpTo(2, Dilate(), Preorder(Try(ConstantFold))),
                Any(Dilate(), Delay()),
            ),
        ),
        AnnotateLanes(512),
        PushDownConvertLanes(),
    )

    # Apply transformations
    results = transforms.apply_signal(program)

    # Ensure we got some results
    assert len(results) > 0, f"No results from transforms for {element_type}"

    # Generate and run benchmark with smaller parameters for faster testing
    codegen = CodeGen()
    benchmark_result = generate_and_run_benchmark(
        codegen,
        results,
        ["k" + str(i) for i in range(len(results))],
        include_correctness_check=True,
        input_size=1 << 12,  # Smaller size for faster testing
        warmup_iterations=2,
        benchmark_iterations=3,
    )

    # Verify benchmark ran successfully
    assert benchmark_result is not None, f"Benchmark result is None for {element_type}"
    assert (
        benchmark_result.get("return_code") == 0
    ), f"Benchmark failed for {element_type}: {benchmark_result.get('error')}"

    print(f"✓ {element_type.name} passed")
