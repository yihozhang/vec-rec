"""Tests for ComputedSignal2D - an on-the-fly (closure) 2D input signal.

The signal struct, its buffer allocation, and its construction are entirely
benchmark/test-owned: the struct definition is shipped as a `prelude`, the raw
buffers are allocated/filled by `setup_code`, and the constructed object is passed
into the kernel constructor. Here we use a MatchSignal2D that computes
g(row, col) = (s1[row] == s2[col]) ? 0 : 1 from two 1D strings, and demonstrate
both the raw signal and a full Levenshtein edit-distance recurrence built on it.
"""

import numpy as np

from vecrec.expr.base import SignalExpr, Type, with_types
from vecrec.expr.kernel import TVKernel2D
from vecrec.expr.signal import ComputedSignal2D, Num, Impulse
from vecrec.expr.signal_ops import Ith, Recurse2D
from vecrec.util import ElementType, get_native_bits
from vecrec.codegen import CodeGen, generate_and_run_benchmark, generate_kernel_executable
from vecrec.transform import AnnotateLanes, InferNRows
from vecrec.optimize import get_candidate_schedules

ALPHA = 4
GAP = 1
_BOUND = int(np.iinfo(np.int32).max // 8)

PRELUDE = r"""
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct MatchSignal2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    const int8_t *s1; const int8_t *s2; int stride;
    elt_type match_cost, mismatch_cost; int current_row, my_col;
    MatchSignal2D() = delete;
    MatchSignal2D(const int8_t *s1, const int8_t *s2, int stride,
                  elt_type m, elt_type mm)
        : s1(s1), s2(s2), stride(stride), match_cost(m), mismatch_cost(mm),
          current_row(block_row), my_col(0) {}
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type s2v;
        if constexpr (lanes == 1) {
            s2v = (elt_type)s2[my_col];
        } else {
            _Pragma("unroll") for (int k = 0; k < lanes; k++) s2v[k] = (elt_type)s2[my_col + k];
        }
        const vec_type mv = match_cost;
        const vec_type mmv = mismatch_cost;
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int r = current_row - i;
            elt_type a = (r >= 0) ? (elt_type)s1[r] : (elt_type)(-1);
            vec_type av = a;
            reg[base + i] = (av == s2v) ? mv : mmv;
        }
        my_col += lanes;
    }
    void reset_and_next_row() { current_row++; my_col = 0; }
};
"""

SETUP = (
    f"std::vector<int8_t> s1(N / row_size), s2(row_size);\n"
    f"for (int i = 0; i < N / row_size; i++) s1[i] = (int8_t)(i % {ALPHA});\n"
    f"for (int j = 0; j < row_size; j++) s2[j] = (int8_t)(j % {ALPHA});\n"
)


def _match_signal() -> ComputedSignal2D:
    return ComputedSignal2D(
        "g",
        cpp_struct="MatchSignal2D",
        prelude=PRELUDE,
        setup_code=SETUP,
        make_args="s1.data(), s2.data(), row_size, 0, 1",
    )


def test_computed_signal2d_raw_output() -> None:
    """Ith(ComputedSignal2D, 0) should emit the computed match matrix, with the
    signal struct constructed in driver scope and passed into the kernel."""
    with with_types(Type.Arith, ElementType.I32):
        expr: SignalExpr = Ith(_match_signal(), 0)
        expr = InferNRows().apply_generic(expr)[0]
        expr = AnnotateLanes(get_native_bits()).apply_generic(expr)[0]

    row_size = 256
    n_rows = 8
    n = n_rows * row_size

    # The computed signal is surfaced as a descriptor (prelude + ctor) rather than a
    # raw input buffer; the kernel takes the constructed object as a ctor parameter.
    code = CodeGen().generate(expr, "k_match")
    assert len(code.computed_signals) == 1
    assert "struct MatchSignal2D" in code.computed_signals[0].prelude
    assert "MatchSignal2D<int32_vec16, 1, 1, 0>" in code.text  # resolved type as a ctor param

    runner = generate_kernel_executable(CodeGen(), expr, row_size=row_size)
    output, _, _ = runner.run({}, n=n)  # no fed inputs: setup_code owns the buffers

    expected = np.array(
        [[0 if (i % ALPHA) == (j % ALPHA) else 1 for j in range(row_size)] for i in range(n_rows)],
        dtype=np.int32,
    ).flatten()
    assert np.array_equal(output, expected)


def _levenshtein_reference(n_rows: int, cols: int) -> np.ndarray:
    s1 = np.array([i % ALPHA for i in range(n_rows)], dtype=np.int32)
    s2 = np.array([j % ALPHA for j in range(cols)], dtype=np.int32)
    M = np.zeros((n_rows, cols), dtype=np.int32)
    for i in range(n_rows):
        prev = M[i - 1] if i > 0 else np.full(cols, _BOUND, dtype=np.int64)
        for j in range(cols):
            if i == 0 and j == 0:
                M[i, j] = 0
                continue
            cost = 0 if int(s1[i]) == int(s2[j]) else 1
            up = GAP + int(prev[j])
            diag_in = int(prev[j - 1]) if j > 0 else _BOUND
            diag = cost + diag_in
            left = GAP + (int(M[i, j - 1]) if j > 0 else _BOUND)
            M[i, j] = min(up, diag, left)
    return M.flatten()


def _edit_distance_expr() -> SignalExpr:
    with with_types(Type.TropMin, ElementType.I32):
        g = _match_signal()
        kernel = TVKernel2D([
            [Num(Type.TropMin.zero()), Num(GAP)],
            [Num(GAP), g.at(0)],
        ])
        return Recurse2D(kernel, Impulse(0)).at(0)


def test_edit_distance_correctness() -> None:
    """Full Levenshtein recurrence over the computed match signal matches a Python
    reference computed from the same deterministic strings."""
    expr = _edit_distance_expr()
    schedule = get_candidate_schedules(expr, get_native_bits())[0]

    row_size = 256
    n_rows = 8
    n = n_rows * row_size

    runner = generate_kernel_executable(CodeGen(rows_per_iter=1), schedule, row_size=row_size)
    output, _, _ = runner.run({}, n=n)

    expected = _levenshtein_reference(n_rows, row_size)
    assert np.array_equal(output, expected)


def test_edit_distance_schedules_agree() -> None:
    """All candidate schedules compile, run, and agree with the reference schedule
    (cross-schedule validation), exercising the delay/dilate transforms on the
    computed-signal recurrence."""
    expr = _edit_distance_expr()
    schedules = get_candidate_schedules(expr, get_native_bits())

    result = generate_and_run_benchmark(CodeGen(rows_per_iter=1), schedules, include_correctness_check=True)
    assert result["return_code"] == 0, result.get("error")

    validation = result["output"].get("validation", {})
    bad = {k: v for k, v in validation.items() if not (v is True or (isinstance(v, int) and v >= 256))}
    assert not bad, f"schedules diverged from reference: {bad}"
