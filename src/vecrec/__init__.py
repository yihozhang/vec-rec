from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from vecrec.codegen import CodeGen, instantiate_kernels, generate_benchmark, generate_and_run_benchmark
from vecrec.transform import *
from vecrec.expr import Convolve, Recurse, TIKernel, Type, Var


def main():
    kernel = TIKernel([0, 1.8, -0.9], Type.Arith)
    signal = Var("x", Type.Arith)
    expr = Recurse(kernel, signal)
    
    constant_fold = Preorder(Try(ConstantFold))
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
    result = generate_and_run_benchmark(codegen, results, [f"k{i+1}" for i in range(len(results))], True)
    print(result)