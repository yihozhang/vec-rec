from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from vecrec.codegen import CodeGen, instantiate_kernels, generate_benchmark, generate_and_run_benchmark
from vecrec.transform import *
from vecrec.expr import Convolve, Recurse, TIKernel, Type, Var
from vecrec.util import ElementType


def main():
    kernel = TIKernel([0, 1.8, -0.9], Type.Arith, ElementType.Float)
    signal = Var("x", Type.Arith, ElementType.Float)
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
    result = generate_and_run_benchmark(codegen, results, True)
    print(result)
