# M(i,j) = max(M(i-1,j-1), I(i-1,j-1), D(i-1,j-1)) + s(i,j);
# I(i,j) = max(M(i-1,j) + g_o + g_e, I(i-1,j) + g_e);
# D(i,j) = max(M(i,j-1) + g_o + g_e, D(i,j-1) + g_e)
#
# Affine-gap global alignment in TropMax (max-plus). Like edit_distance.py, the
# time-varying diagonal substitution score s(i,j) is NOT a materialized 2D input;
# it is computed on the fly by a user-provided C++ struct (SubstSignal2D) from two
# 1D sequences s1, s2. The sequences are streams of int8 characters -- the same
# representation parasail uses (its s1/s2 are `const char *`, i.e. 8-bit bytes,
# mapped through a substitution matrix). Here the on-the-fly score is the simple
# match/mismatch scheme s(i,j) = (s1[i] == s2[j]) ? MATCH_SCORE : MISMATCH_SCORE.

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.expr.base import SignalExpr, SignalExpr2D
from vecrec.util import ElementType
from vecrec.codegen import CodeGen, generate_kernel_executable

from benchmark import Benchmark

GO = -11   # gap open penalty
GE = -1    # gap extend penalty
GO_GE = GO + GE  # combined: -12

MATCH_SCORE = 2     # substitution score for s1[i] == s2[j]
MISMATCH_SCORE = -1  # substitution score for s1[i] != s2[j]
ALPHA = 4            # alphabet size for the deterministic test/benchmark strings

# TropMax "zero" (the -inf identity for max) is represented in I32 by INT32_MIN/8,
# matching the sentinel used by the SConvolve/SRecurse TropMax templates in common.h.
_NEGINF = int(np.iinfo(np.int32).min // 8)

# The on-the-fly substitution-score signal. g(row, col) is MATCH_SCORE when
# s1[row] == s2[col] and MISMATCH_SCORE otherwise. Row offset i exposes row
# (current_row - i); rows < 0 never match (so they score MISMATCH), which is
# harmless because the recurrence's "row -1" predecessor is killed by the
# TropMax sentinel anyway. Affine gap only reads .at(0) (n_rows=1), but the
# struct is written general so it composes with deeper windows.
SUBST_SIGNAL_PRELUDE = r"""
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct SubstSignal2D {
    using elt_type = typename ElementType<vec_type>::type;   // score type (e.g. int32)
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);

    const int8_t *__restrict__ s1;   // row-indexed characters
    const int8_t *__restrict__ s2;   // column-indexed characters, length >= stride
    int stride;
    elt_type match_score;
    elt_type mismatch_score;
    int current_row;
    int my_col;
    vec_type s1v[n_rows];            // broadcast current-row s1 chars (refreshed per row)

    SubstSignal2D() = delete;
    SubstSignal2D(const int8_t *__restrict__ s1, const int8_t *__restrict__ s2, int stride,
                  elt_type match_score, elt_type mismatch_score)
        : s1(s1), s2(s2), stride(stride),
          match_score(match_score), mismatch_score(mismatch_score),
          current_row(block_row), my_col(0) { load_s1_row(); }

    // s1[current_row - i] is constant across a row (current_row only moves in
    // reset_and_next_row), so broadcast it into s1v once per row instead of
    // reloading + broadcasting (and re-evaluating the r>=0 branch) on every run().
    inline __attribute__((always_inline)) void load_s1_row() {
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int r = current_row - i;
            elt_type a = (r >= 0) ? (elt_type)s1[r] : (elt_type)(-1);
            s1v[i] = a;                                 // scalar broadcast to all lanes
        }
    }

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        // Widen the int8 characters of s2[my_col .. my_col+lanes) into the score-typed
        // lane vector so the comparison/blend happens at the score element width.
        vec_type s2v;
        if constexpr (lanes == 1) {
            s2v = (elt_type)s2[my_col];
        } else {
            _Pragma("unroll") for (int k = 0; k < lanes; k++) s2v[k] = (elt_type)s2[my_col + k];
        }
        const vec_type mv = match_score;                // scalar broadcast to all lanes
        const vec_type mmv = mismatch_score;
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            reg[base + i] = (s1v[i] == s2v) ? mv : mmv;
        }
        my_col += lanes;
    }

    void reset_and_next_row() {
        current_row++;
        my_col = 0;
        load_s1_row();
    }
};
"""

# Allocate + fill the two 1D strings from a deterministic formula so the Python
# reference can reproduce them exactly. Emitted once in driver scope; `N` and
# `row_size` are in scope.
# s1 gets a small halo: load_s1_row() runs on every reset_and_next_row(), including the
# final one that bumps current_row one row past the last (its result is never read, but
# the read must stay in bounds). The halo covers that plus any block_row offset.
SUBST_SIGNAL_SETUP = (
    f"std::vector<int8_t> s1(N / row_size + 64), s2(row_size);\n"
    f"for (int i = 0; i < N / row_size; i++) s1[i] = (int8_t)(i % {ALPHA});\n"
    f"for (int j = 0; j < row_size; j++) s2[j] = (int8_t)(j % {ALPHA});\n"
)


class AffineGapBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "affine-gap"

    @property
    def description(self) -> str:
        return "Affine-gap global alignment in TropMax (max-plus) with an on-the-fly computed substitution signal"

    @property
    def verify_n_rows_2d(self) -> int:
        return 8

    def _subst_signal(self) -> ComputedSignal2D:
        return ComputedSignal2D(
            "s",
            cpp_struct="SubstSignal2D",
            prelude=SUBST_SIGNAL_PRELUDE,
            setup_code=SUBST_SIGNAL_SETUP,
            make_args=f"s1.data(), s2.data(), row_size, {MATCH_SCORE}, {MISMATCH_SCORE}",
        )

    def original_expr(self) -> SignalExpr:
        with with_types(Type.TropMax, ElementType.I32):
            s = self._subst_signal()
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
        s1 = inputs["s1"].astype(np.int32)
        s2 = inputs["s2"].astype(np.int32)
        rows, cols = len(s1), len(s2)

        def sub(i: int, j: int) -> int:
            return MATCH_SCORE if int(s1[i]) == int(s2[j]) else MISMATCH_SCORE

        M = np.full((rows, cols), _NEGINF, dtype=np.int64)
        I = np.full((rows, cols), _NEGINF, dtype=np.int64)
        D = np.full((rows, cols), _NEGINF, dtype=np.int64)

        for i in range(rows):
            for j in range(cols):
                # Diagonal predecessor (i-1, j-1)
                m_diag = int(M[i - 1, j - 1]) if (i > 0 and j > 0) else _NEGINF
                i_diag = int(I[i - 1, j - 1]) if (i > 0 and j > 0) else _NEGINF
                d_diag = int(D[i - 1, j - 1]) if (i > 0 and j > 0) else _NEGINF

                best_diag = max(m_diag, i_diag, d_diag)
                if i == 0 and j == 0:
                    # Seed: Impulse(0) fires once at (0,0); max(-inf+s, 0+s) = s(0,0)
                    M[i, j] = np.int64(sub(i, j))
                elif best_diag > _NEGINF:
                    M[i, j] = np.int64(best_diag + sub(i, j))

                # I(i,j) = max(M(i-1,j) + go+ge, I(i-1,j) + ge)
                m_up = int(M[i - 1, j]) if i > 0 else _NEGINF
                i_up = int(I[i - 1, j]) if i > 0 else _NEGINF
                if m_up > _NEGINF:
                    I[i, j] = max(int(I[i, j]), m_up + GO_GE)
                if i_up > _NEGINF:
                    I[i, j] = max(int(I[i, j]), i_up + GE)

                # D(i,j) = max(M(i,j-1) + go+ge, D(i,j-1) + ge)
                m_left = int(M[i, j - 1]) if j > 0 else _NEGINF
                d_left = int(D[i, j - 1]) if j > 0 else _NEGINF
                if m_left > _NEGINF:
                    D[i, j] = max(int(D[i, j]), m_left + GO_GE)
                if d_left > _NEGINF:
                    D[i, j] = max(int(D[i, j]), d_left + GE)

        return M.flatten()

    def verify_schedule(
        self, expr: SignalExpr | SignalExpr2D, codegen: CodeGen | None = None
    ) -> tuple[bool, str]:
        """The substitution signal owns its buffer allocation in C++, so there are no
        fed inputs. Run the kernel with an explicit output size and compare to the
        Python reference computed from the same deterministic string formula used by
        the C++ setup code."""
        try:
            if codegen is None:
                codegen = self.build_codegen()
            row_size = 256
            runner = generate_kernel_executable(codegen, expr, row_size=row_size)

            n_rows = self.verify_n_rows_2d
            n = n_rows * row_size

            kernel_out, _, _ = runner.run({}, n=n)

            # Reproduce the deterministic strings the C++ setup code generated.
            s1 = np.array([i % ALPHA for i in range(n_rows)], dtype=np.int32)
            s2 = np.array([j % ALPHA for j in range(row_size)], dtype=np.int32)
            expected = self._python_reference({"s1": s1, "s2": s2}, n)

            min_len = min(len(kernel_out), len(expected))
            kernel_out = np.asarray(kernel_out[:min_len], dtype=np.int64)
            expected = np.asarray(expected[:min_len], dtype=np.int64)

            # Unreachable M cells (first row/col except the (0,0) seed) hold the
            # TropMax -inf sentinel -- in the kernel that is the sentinel plus a
            # small substitution score, in the reference exactly the sentinel.
            # Both are far below any real alignment score, so clamp to a common
            # floor; the comparison then only tests reachable cells exactly.
            floor = _NEGINF // 2
            kernel_out = np.maximum(kernel_out, floor)
            expected = np.maximum(expected, floor)

            if not np.array_equal(kernel_out, expected):
                bad = np.where(kernel_out != expected)[0]
                first_diff = int(bad[0])
                return False, (
                    f"Mismatch at index {first_diff}: "
                    f"got {int(kernel_out[first_diff])}, expected {int(expected[first_diff])}"
                )
            return True, ""
        except Exception as exc:
            return False, str(exc)
