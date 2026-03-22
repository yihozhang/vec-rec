from __future__ import annotations

from vecrec.codegen import CodeGen, generate_and_run_benchmark
from vecrec.transform import *
from vecrec.expr import Recurse, TIKernel, Type, Var, with_types
from vecrec.util import ElementType


def main():
    with with_types(Type.Arith, ElementType.Float):
        kernel = TIKernel([0, 1.8, -0.9])
        signal = Var("x")
    expr = Recurse(kernel, signal)
    
    constant_fold = SaturatePreorder(ConstantFold)
    schedule = Seq(
        Any(
            Noop(),
            Seq(
                Dilate(),
                constant_fold,
                Any(
                    Noop(), 
                    Seq(Delay(), constant_fold),
                    Seq(*[Delay(), constant_fold] * 3), 
                    Seq(*[Delay(), constant_fold] * 9),
                    Dilate(),
                )
            )
        ),
        
        AnnotateLanes(512),
        PushDownConvertLanes(),
        # UnrollToMaxLanes(512),
    )
    results = schedule.apply_signal(expr)
    codegen = CodeGen()
    print(len(results))
    result = generate_and_run_benchmark(codegen, results, True)
    print(result)
