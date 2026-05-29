"""Drop-in replacements for scipy.signal.sosfilt and scipy.signal.sosfiltfilt
using VecRec for time-invariant second-order section cascades.

The C++ kernel is generated and compiled on the first call for a given set of
SOS coefficients.  Subsequent calls with the same coefficients reuse the cached
executable, amortizing compilation cost across many calls.

Usage::

    from integrations.vecrec_sosfilt import vecrec_sosfilt, vecrec_sosfiltfilt
    y = vecrec_sosfilt(sos, x)      # drop-in for scipy.signal.sosfilt
    y = vecrec_sosfiltfilt(sos, x)  # drop-in for scipy.signal.sosfiltfilt
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from vecrec.codegen import CodeGen, generate_kernel_executable
from vecrec.codegen.generate_kernel import KernelExecutable
from vecrec.expr import Convolve, Recurse, TIKernel, Type, Var, with_types
from vecrec.util import ElementType

# Module-level cache: (sos_key,) -> KernelExecutable
_KERNEL_CACHE: Dict[Tuple[float, ...], KernelExecutable] = {}


def _sos_cache_key(sos: np.ndarray) -> Tuple[float, ...]:
    return tuple(float(v) for v in sos.flatten())


def _build_sos_expr(sos: np.ndarray) -> Recurse:
    """Build a nested VecRec Recurse expression from an SOS coefficient array.

    sos shape: (N_sections, 6), each row: [b0, b1, b2, 1.0, a1, a2].
    Sections applied in row order: section 0 is innermost (applied first).
    """
    with with_types(Type.Arith, ElementType.Float):
        from vecrec.expr.base import SignalExpr
        sig: SignalExpr = Var("g")
        for row in sos:
            b0, b1, b2 = float(row[0]), float(row[1]), float(row[2])
            a1, a2 = float(row[4]), float(row[5])
            sig = Recurse(
                TIKernel([0.0, -a1, -a2]),
                Convolve(TIKernel([b0, b1, b2]), sig),
            )
    return sig  # type: ignore[return-value]


def _get_or_build_runner(sos: np.ndarray) -> KernelExecutable:
    """Return a cached KernelExecutable for these SOS coefficients, building it on first use."""
    sos_f32 = np.asarray(sos, dtype=np.float32)
    key = _sos_cache_key(sos_f32)
    if key not in _KERNEL_CACHE:
        expr = _build_sos_expr(sos_f32)
        codegen = CodeGen()
        _KERNEL_CACHE[key] = generate_kernel_executable(codegen, expr)
    return _KERNEL_CACHE[key]


def vecrec_sosfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Drop-in for scipy.signal.sosfilt for time-invariant cascades.

    Generates and caches the C++ kernel on first call for these sos coefficients.

    Args:
        sos: Array of shape (N_sections, 6) with rows [b0, b1, b2, 1.0, a1, a2].
        x:   1-D input signal (will be cast to float32).

    Returns:
        Filtered output as a float32 numpy array of the same length as x.
    """
    runner = _get_or_build_runner(sos)
    output, _, _ = runner.run({"g": np.asarray(x, dtype=np.float32)})
    return output


def vecrec_sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Drop-in for scipy.signal.sosfiltfilt (zero-phase filtering via two passes + flip).

    Applies the forward filter, flips the result, applies the same filter again,
    and flips back — producing zero-phase (non-causal) output.

    Args:
        sos: Array of shape (N_sections, 6) with rows [b0, b1, b2, 1.0, a1, a2].
        x:   1-D input signal (will be cast to float32).

    Returns:
        Zero-phase filtered output as a float32 numpy array of the same length as x.
    """
    fwd = vecrec_sosfilt(sos, x)
    bwd = vecrec_sosfilt(sos, np.flip(fwd))
    return np.flip(bwd).copy()
