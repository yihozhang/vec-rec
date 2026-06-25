# Levenshtein edit distance D(i,j) = min(
#     D(i-1,j-1) + (s1[i] != s2[j] ? 1 : 0),   # substitution / match
#     D(i-1,j)   + 1,                           # deletion
#     D(i,j-1)   + 1,                           # insertion
# )
#
# This is structurally identical to Needleman-Wunsch (benchmarks/trop/needleman_wunsch.py),
# but in the TropMin (min-plus) semiring instead of TropMax, and — crucially — the
# time-varying diagonal substitution cost is NOT a materialized 2D input. It is computed
# on the fly by a user-provided C++ struct (MatchSignal2D) from two 1D strings s1, s2.
# This benchmark demonstrates VecRec's generalized, pluggable input signal: a signal can
# be any C++ struct honoring the run()/reset_and_next_row() interface, shipped as a
# `prelude` and constructed by benchmark-owned `setup_code`, with the object passed into
# the kernel constructor.

from __future__ import annotations

import numpy as np

from vecrec.expr import *
from vecrec.expr.base import KernelExpr2D, SignalExpr2D
from vecrec.util import ElementType
from vecrec.codegen import CodeGen, generate_kernel_executable

from benchmark import Benchmark

GAP = 1  # insertion / deletion cost
MATCH_COST = 0
MISMATCH_COST = 1
ALPHA = 4  # alphabet size for the deterministic test/benchmark strings

# TropMin "zero" (the +inf identity for min) is represented in I32 by INT32_MAX/8,
# matching the sentinel used by the SConvolve/SRecurse TropMin templates in common.h.
_BOUND = int(np.iinfo(np.int32).max // 8)

# The on-the-fly substitution-cost signal. g(row, col) = (s1[row] == s2[col]) ? 0 : 1.
# Row offset i exposes row (current_row - i); rows < 0 never match (so cost = mismatch),
# which is harmless because the recurrence's "row -1" predecessor is killed by the
# SConvolve/SRecurse TropMin sentinel anyway. Edit distance only reads .at(0) (n_rows=1),
# but the struct is written general like InputSignal2D so it composes with deeper windows.
MATCH_SIGNAL_PRELUDE = r"""
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct MatchSignal2D {
    using elt_type = typename ElementType<vec_type>::type;   // cost type (e.g. int32)
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);

    const int8_t *s1;   // row-indexed characters
    const int8_t *s2;   // column-indexed characters, length >= stride
    int stride;
    elt_type match_cost;
    elt_type mismatch_cost;
    int current_row;
    int my_col;

    MatchSignal2D() = delete;
    MatchSignal2D(const int8_t *s1, const int8_t *s2, int stride,
                  elt_type match_cost, elt_type mismatch_cost)
        : s1(s1), s2(s2), stride(stride),
          match_cost(match_cost), mismatch_cost(mismatch_cost),
          current_row(block_row), my_col(0) {}

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        // Widen the int8 characters of s2[my_col .. my_col+lanes) into the cost-typed
        // lane vector so the comparison/blend happens at the cost element width.
        vec_type s2v;
        if constexpr (lanes == 1) {
            s2v = (elt_type)s2[my_col];
        } else {
            _Pragma("unroll") for (int k = 0; k < lanes; k++) s2v[k] = (elt_type)s2[my_col + k];
        }
        const vec_type mv = match_cost;                // scalar broadcast to all lanes
        const vec_type mmv = mismatch_cost;
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int r = current_row - i;
            elt_type a = (r >= 0) ? (elt_type)s1[r] : (elt_type)(-1);
            vec_type av = a;                           // scalar broadcast to all lanes
            reg[base + i] = (av == s2v) ? mv : mmv;
        }
        my_col += lanes;
    }

    void reset_and_next_row() {
        current_row++;
        my_col = 0;
    }
};
"""

# Allocate + fill the two 1D strings from a deterministic formula so the Python reference
# can reproduce them exactly. Emitted once in driver scope; `N` and `row_size` are in scope.
MATCH_SIGNAL_SETUP = (
    f"std::vector<int8_t> s1(N / row_size), s2(row_size);\n"
    f"for (int i = 0; i < N / row_size; i++) s1[i] = (int8_t)(i % {ALPHA});\n"
    f"for (int j = 0; j < row_size; j++) s2[j] = (int8_t)(j % {ALPHA});\n"
)


class EditDistanceBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "edit-distance"

    @property
    def description(self) -> str:
        return "Levenshtein edit distance in TropMin (min-plus) with an on-the-fly computed match signal"

    @property
    def verify_n_rows_2d(self) -> int:
        # Divisible by the row-blocking factors the sweep tries (2, 4) so blocks tile evenly.
        return 8

    def _match_signal(self) -> ComputedSignal2D:
        return ComputedSignal2D(
            "g",
            cpp_struct="MatchSignal2D",
            prelude=MATCH_SIGNAL_PRELUDE,
            setup_code=MATCH_SIGNAL_SETUP,
            make_args=f"s1.data(), s2.data(), row_size, {MATCH_COST}, {MISMATCH_COST}",
        )

    def original_expr(self) -> SignalExpr:
        with with_types(Type.TropMin, ElementType.I32):
            g = self._match_signal()
            posinf = Num(Type.TropMin.zero())
            gap = Num(GAP)
            kernel: KernelExpr2D = TVKernel2D([
                [posinf, gap],     # same row: D(i, j-1) + gap
                [gap, g.at(0)],    # one row back: D(i-1, j) + gap  and  D(i-1, j-1) + g(i,j)
            ])
            return Recurse2D(kernel, Impulse(0)).at(0)

    def _python_reference(self, inputs: dict[str, np.ndarray], n: int) -> np.ndarray:
        s1 = inputs["s1"].astype(np.int32)
        s2 = inputs["s2"].astype(np.int32)
        rows, cols = len(s1), len(s2)
        M = np.zeros((rows, cols), dtype=np.int32)

        for i in range(rows):
            # Repeater "row -1" is the TropMin sentinel (+inf), so unreachable predecessors
            # never win the min — this yields the standard D(0,j)=j and D(i,0)=i boundaries.
            prev_row = M[i - 1] if i > 0 else np.full(cols, _BOUND, dtype=np.int64)

            for j in range(cols):
                if i == 0 and j == 0:
                    M[i, j] = 0  # Recurse2D Impulse(0) seeds D(0,0)=0.
                    continue
                cost = MATCH_COST if int(s1[i]) == int(s2[j]) else MISMATCH_COST
                up_val = GAP + int(prev_row[j])                                # D(i-1, j) + gap
                diag_input = int(prev_row[j - 1]) if j > 0 else _BOUND
                diag_val = cost + diag_input                                  # D(i-1, j-1) + g(i,j)
                left_val = GAP + (int(M[i, j - 1]) if j > 0 else _BOUND)      # D(i, j-1) + gap
                M[i, j] = min(left_val, diag_val, up_val)

        return M.flatten()

    def verify_schedule(
        self, expr: SignalExpr | SignalExpr2D, codegen: CodeGen | None = None
    ) -> tuple[bool, str]:
        """The match signal owns its buffer allocation in C++, so there are no fed inputs.
        Run the kernel with an explicit output size and compare to the Python reference
        computed from the same deterministic string formula used by the C++ setup code."""
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
            kernel_out = kernel_out[:min_len]
            expected = expected[:min_len]

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
