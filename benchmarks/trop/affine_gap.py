# M(i,j) = max(M(i-1,j-1), I(i-1,j-1), D(i-1,j-1)) + s(i,j);
# I(i,j) = max(M(i-1,j) + g_o + g_e, I(i-1,j) + g_e);
# D(i,j) = max(M(i,j-1) + g_o + g_e, D(i,j-1) + g_e)

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.util import ElementType

from benchmark import Benchmark

GO = -11   # gap open penalty
GE = -1    # gap extend penalty
GO_GE = GO + GE  # combined: -12

_NEGINF = np.int64(np.iinfo(np.int64).min // 8)


class AffineGapBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "affine-gap"

    @property
    def description(self) -> str:
        return "Affine-gap global alignment in TropMax (max-plus) semiring"

    @property
    def verify_n_rows_2d(self) -> int:
        return 8

    def original_expr(self) -> SignalExpr:
        with with_types(Type.TropMax, ElementType.I64):
            s = Var2D("s")
            neginf = Type.TropMax.zero()
            one = Type.TropMax.one()
            z = TIKernel.z()

            def body(prev: RVec2D) -> SignalVectorExpr:
                prev_m = prev.at(0).at(0)  # M(i-1, j)
                prev_i = prev.at(1).at(0)  # I(i-1, j)
                prev_d = prev.at(2).at(0)  # D(i-1, j)

                # Diagonal predecessors: M/I/D(i-1, j-1)
                prev_m_diag = Convolve(z, prev_m)
                prev_i_diag = Convolve(z, prev_i)
                prev_d_diag = Convolve(z, prev_d)

                # M(i,j) = max(M,I,D diagonals) + s(i,j)
                # SAdd = pointwise max, PointwiseMul = pointwise + in TropMax
                max_diag = SAdd(SAdd(prev_m_diag, prev_i_diag), prev_d_diag)
                s_row = Ith(s, 0)
                # Seed M[0][0] = s[0][0] via a one-shot Impulse(one=0) at (0,0)
                M = SAdd(
                    PointwiseMul(max_diag, s_row),
                    PointwiseMul(Impulse(one), s_row),
                )

                # I(i,j) = max(M(i-1,j) + go+ge, I(i-1,j) + ge)
                I = SAdd(
                    PointwiseMul(prev_m, Num(GO_GE)),
                    PointwiseMul(prev_i, Num(GE)),
                )

                # D(i,j) = max(M(i,j-1) + go+ge, D(i,j-1) + ge)
                # Convolve(TIKernel([neginf, c]), x)[j] = c + x[j-1] in TropMax
                ext_D = Convolve(TIKernel([neginf, GO_GE]), M)
                D = Recurse(TIKernel([neginf, GE]), ext_D)

                return SignalVector([M, I, D])

            rv = RepeaterVec(3, body)
            return Ith(rv.at(0), 0)  # current-row M

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        s = inputs["s"].astype(np.int64)
        rows, cols = s.shape

        M = np.full((rows, cols), _NEGINF, dtype=np.int64)
        I = np.full((rows, cols), _NEGINF, dtype=np.int64)
        D = np.full((rows, cols), _NEGINF, dtype=np.int64)

        for i in range(rows):
            for j in range(cols):
                # Diagonal predecessor (i-1, j-1)
                m_diag = int(M[i - 1, j - 1]) if (i > 0 and j > 0) else int(_NEGINF)
                i_diag = int(I[i - 1, j - 1]) if (i > 0 and j > 0) else int(_NEGINF)
                d_diag = int(D[i - 1, j - 1]) if (i > 0 and j > 0) else int(_NEGINF)

                best_diag = max(m_diag, i_diag, d_diag)
                if i == 0 and j == 0:
                    # Seed: Impulse(0) fires once at (0,0); max(-inf+s, 0+s) = s[0][0]
                    M[i, j] = np.int64(int(s[i, j]))
                elif best_diag > _NEGINF:
                    M[i, j] = np.int64(best_diag + int(s[i, j]))

                # I(i,j) = max(M(i-1,j) + go+ge, I(i-1,j) + ge)
                m_up = int(M[i - 1, j]) if i > 0 else int(_NEGINF)
                i_up = int(I[i - 1, j]) if i > 0 else int(_NEGINF)
                if m_up > _NEGINF:
                    I[i, j] = max(int(I[i, j]), m_up + GO_GE)
                if i_up > _NEGINF:
                    I[i, j] = max(int(I[i, j]), i_up + GE)

                # D(i,j) = max(M(i,j-1) + go+ge, D(i,j-1) + ge)
                m_left = int(M[i, j - 1]) if j > 0 else int(_NEGINF)
                d_left = int(D[i, j - 1]) if j > 0 else int(_NEGINF)
                if m_left > _NEGINF:
                    D[i, j] = max(int(D[i, j]), m_left + GO_GE)
                if d_left > _NEGINF:
                    D[i, j] = max(int(D[i, j]), d_left + GE)

        return M.flatten()
