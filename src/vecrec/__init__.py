from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from vecrec.codegen import CodeGen, instantiate_kernels, generate_benchmark, generate_and_run_benchmark
from vecrec.transform import ApplyParallel, ConstantFold, Delay, Dilate, ApplySequence, Preorder, Try
from vecrec.expr import Convolve, Recurse, TIKernel, Var


def main():
    kernel = TIKernel([0, 1.8, -0.9])
    signal = Var("x")
    expr = Recurse(kernel, signal)
    transforms = [
        Dilate(),
        Dilate(),
        ApplyParallel([Dilate(), Delay()]),
        Preorder(Try(ConstantFold)),
    ]
    results = ApplySequence(transforms).apply_signal(expr)
    print(results[0])
    print(results[1])
    codegen = CodeGen(256)
    original = codegen.generate(expr, "original")
    dilated = codegen.generate(results[0], "dilated")
    dilate_and_delayed = codegen.generate(results[1], "dilate_and_delayed")
    instantiate_kernels("output.h", [original, dilated, dilate_and_delayed])
