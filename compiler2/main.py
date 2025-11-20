from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from compiler2.transform import ConstantFold, Delay, Dilate, ApplySequence, Preorder, Try
from compiler2.expr import Convolve, Recurse, TIKernel, Var

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
    results = ApplySequence(transforms).apply(expr)
    print(results)
if __name__ == "__main__":
    main()