from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from vecrec.codegen import CodeGen
from vecrec.transform import ConstantFold, Delay, Dilate, ApplySequence, Preorder, Try
from vecrec.expr import Convolve, Recurse, TIKernel, Var


def main():
    kernel = TIKernel([0, 0.9, -1.8])
    signal = Var("x")
    expr = Recurse(kernel, signal)
    transforms = [
        Dilate(),
        Dilate(),
        Dilate(),
        Delay(),
        Preorder(Try(ConstantFold)),
    ]
    results = ApplySequence(transforms).apply_signal(expr)
    codegen = CodeGen(512)
    code = codegen.generate(results[0], "test")
    code.to_file("output.cpp")
