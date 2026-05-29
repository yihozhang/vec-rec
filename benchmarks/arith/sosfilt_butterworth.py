# 4th-order Butterworth bandpass filter represented as a cascade of biquad sections
# (second-order sections, SOS), matching scipy.signal.sosfilt.
#
# butter(2, [0.1, 0.4], btype='bandpass') produces a 4th-order bandpass (order 2
# Butterworth prototype doubled by the bandpass transformation) with 2 SOS sections.
#
# Each biquad: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
# In VecRec: Recurse(TIKernel([0, -a1, -a2]), Convolve(TIKernel([b0, b1, b2]), x))
# Sections are applied in row order by nesting: section 0 is innermost (applied first).

from __future__ import annotations

import numpy as np
from scipy.signal import butter

from vecrec.codegen import CodeGen, generate_kernel_executable
from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark

# Generate coefficients at module level so they are fixed constants.
# butter(2, [0.1, 0.4], btype='bandpass') yields 2 SOS sections (4th-order bandpass).
# Each row: [b0, b1, b2, 1.0, a1, a2].
_SOS_F64 = butter(2, [0.1, 0.4], btype="bandpass", output="sos").astype(np.float64)
# float32 version used by VecRec (ElementType.Float = float32)
_SOS_F32 = _SOS_F64.astype(np.float32)


def _build_sos_expr(sos: np.ndarray) -> Recurse:
    """Build a nested Recurse expression from an SOS array (float32).

    sos shape: (N, 6), rows [b0, b1, b2, 1, a1, a2].
    Sections applied in row order: section 0 is innermost (closest to the input).
    """
    with with_types(Type.Arith, ElementType.Float):
        sig: SignalExpr = Var("g")
        for row in sos:
            b0, b1, b2 = float(row[0]), float(row[1]), float(row[2])
            a1, a2 = float(row[4]), float(row[5])
            sig = Recurse(
                TIKernel([0.0, -a1, -a2]),
                Convolve(TIKernel([b0, b1, b2]), sig),
            )
    return sig  # type: ignore[return-value]


def _python_sosfilt(sos_f32: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Pure-Python reference matching VecRec float32 semantics."""
    x = g.astype(np.float32)
    for row in sos_f32:
        b0 = np.float32(row[0])
        b1 = np.float32(row[1])
        b2 = np.float32(row[2])
        a1 = np.float32(row[4])
        a2 = np.float32(row[5])
        y = np.zeros(len(x), dtype=np.float32)
        for i in range(len(x)):
            x0 = x[i]
            x1 = x[i - 1] if i >= 1 else np.float32(0.0)
            x2 = x[i - 2] if i >= 2 else np.float32(0.0)
            y1 = y[i - 1] if i >= 1 else np.float32(0.0)
            y2 = y[i - 2] if i >= 2 else np.float32(0.0)
            y[i] = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x = y
    return x


class SosfiltButterworthBenchmark(Benchmark):

    @property
    def name(self) -> str:
        return "sosfilt-butterworth"

    @property
    def description(self) -> str:
        return "4th-order Butterworth bandpass (cascade of biquad sections), seismic preprocessing"

    def original_expr(self) -> Recurse:
        return _build_sos_expr(_SOS_F32)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        return _python_sosfilt(_SOS_F32, inputs["g"])


class SosfiltfiltButterworthBenchmark(Benchmark):

    @property
    def name(self) -> str:
        return "sosfiltfilt-butterworth"

    @property
    def description(self) -> str:
        return "Zero-phase 4th-order Butterworth bandpass via forward+backward pass"

    def original_expr(self) -> Recurse:
        # The underlying forward-pass expression is the same as sosfilt.
        return _build_sos_expr(_SOS_F32)

    def verify_schedule(
        self,
        expr: SignalExpr | SignalExpr2D,
        codegen: CodeGen | None = None,
    ) -> tuple[bool, str]:
        """Override to apply forward+backward passes for zero-phase filtering."""
        try:
            if codegen is None:
                codegen = self.build_codegen()
            runner = generate_kernel_executable(codegen, expr, row_size=256)

            n = runner.row_size
            g = np.random.uniform(-1.0, 1.0, (n,)).astype(np.float32)
            inputs = {"g": g}

            # Forward pass
            fwd_out, _, _ = runner.run(inputs)
            # Backward pass (flip, filter, flip)
            bwd_in = np.flip(fwd_out).copy()
            bwd_out, _, _ = runner.run({"g": bwd_in})
            kernel_out = np.flip(bwd_out).copy()

            expected = self._python_reference(inputs, n)

            min_len = min(len(kernel_out), len(expected))
            kernel_out = kernel_out[:min_len]
            expected = expected[:min_len]

            if not np.allclose(kernel_out, expected, atol=self.verify_atol, rtol=self.verify_rtol, equal_nan=False):
                bad = np.where(~np.isclose(kernel_out, expected, atol=self.verify_atol, rtol=self.verify_rtol))[0]
                first_diff = int(bad[0])
                return False, (
                    f"Mismatch at index {first_diff}: "
                    f"got {float(kernel_out[first_diff]):.6f}, "
                    f"expected {float(expected[first_diff]):.6f}"
                )
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        """Zero-phase reference: forward pass + flip + filter + flip."""
        g = inputs["g"].astype(np.float32)
        fwd = _python_sosfilt(_SOS_F32, g)
        bwd = _python_sosfilt(_SOS_F32, np.flip(fwd).copy())
        return np.flip(bwd).copy()
