#pragma once
#include "common.h"
#include <limits>
#include <utility>

// ---------------------------------------------------------------------------
// Reg<vec_type, N>: stack-allocated register file for push-based drivers.
// Each node in a push driver reads/writes registers by index.
// ---------------------------------------------------------------------------
template <typename vec_type, int N>
struct Reg {
    vec_type data[N];
    inline __attribute__((always_inline)) vec_type &operator[](int i) { return data[i]; }
    inline __attribute__((always_inline)) const vec_type &operator[](int i) const { return data[i]; }
};

// ---------------------------------------------------------------------------
// Leaf input nodes
// ---------------------------------------------------------------------------

// PushSignal1D: streams a 1-D input array into reg[out_reg] each step.
template <typename vec_type, int out_reg>
struct PushSignal1D {
    using elt_type = typename ElementType<vec_type>::type;
    const vec_type *data;

    PushSignal1D(const elt_type *raw) : data((const vec_type *)raw - 1) {}

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg) {
        ++data;
        memcpy(&reg[out_reg], data, sizeof(vec_type));
    }

    void reset_and_next_row() {}
};

// PushSignal1DConstant: broadcasts a scalar constant into reg[out_reg].
template <typename vec_type, int out_reg>
struct PushSignal1DConstant {
    using elt_type = typename ElementType<vec_type>::type;
    vec_type val;

    PushSignal1DConstant(elt_type v) { val = v; }

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg) {
        reg[out_reg] = val;
    }

    void reset_and_next_row() {}
};

// PushSignal1DImpulse: emits `value` at step 0 (lane 0), then zero thereafter.
template <typename vec_type, int out_reg>
struct PushSignal1DImpulse {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int lanes = vec_lanes_of(vec_type{});

    elt_type impulse_value;
    elt_type zero_value;
    bool fired;

    PushSignal1DImpulse(elt_type value, elt_type zero)
        : impulse_value(value), zero_value(zero), fired(false) {}

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg) {
        if (!fired) {
            fired = true;
            if constexpr (lanes == 1) {
                reg[out_reg] = impulse_value;
            } else {
                vec_type result = zero_value;
                result[0] = impulse_value;
                reg[out_reg] = result;
            }
        } else {
            reg[out_reg] = zero_value;
        }
    }

    void reset_and_next_row() {}
};

// ---------------------------------------------------------------------------
// PushTIKernel: writes constant tap values to an arbitrary set of registers.
// `_taps` values from `vals` are written to the registers listed in `k_regs`.
// ---------------------------------------------------------------------------
template <typename vec_type, int _taps,
          const typename ElementType<vec_type>::type *vals,
          const int *k_regs>
struct PushTIKernel {
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg) {
#pragma unroll
        for (int i = 0; i < _taps; i++)
            reg[k_regs[i]] = vals[i];
    }

    void reset_and_next_row() {}
};

// ---------------------------------------------------------------------------
// PushSConvolveOp / PushSRecurseOp macros
//
// Template parameters:
//   out_reg     : output register index
//   in_reg      : input signal register index
//   _taps       : number of kernel taps (> 0)
//   _max_index  : maximum time index (= indices[_taps-1])
//   indices     : compile-time array of time-delay indices (size _taps)
//   k_regs      : compile-time array of register indices for kernel values
//
// Each iteration:
//   PushSConvolve: reads reg[in_reg] as current input, shifts input history,
//                  computes FIR, writes result to reg[out_reg].
//   PushSRecurse : reads reg[in_reg] as driving signal, computes IIR with
//                  output feedback, writes result to reg[out_reg].
// ---------------------------------------------------------------------------
#define PushSConvolveOp(suffix, zero_init, mul_op, add_op)                         \
    template <typename vec_type, int out_reg, int in_reg,                          \
              int _taps, int _max_index,                                            \
              const int *indices, const int *k_regs>                               \
    struct PushSConvolve##suffix {                                                  \
        using elt_type = typename ElementType<vec_type>::type;                     \
        constexpr static int vec_lanes = vec_lanes_of(vec_type{});                 \
        constexpr static int buffer_size =                                         \
            (_max_index + vec_lanes - 1) / vec_lanes + 1;                          \
        vec_type prev_input[buffer_size];                                          \
                                                                                    \
        PushSConvolve##suffix() {                                                  \
            for (int i = 0; i < buffer_size; i++)                                  \
                prev_input[i] = (zero_init);                                       \
        }                                                                          \
                                                                                    \
        template <typename Reg_t>                                                  \
        inline __attribute__((always_inline)) void run(Reg_t &reg) {              \
            vec_type curr_input = reg[in_reg];                                     \
            _Pragma("unroll") for (int i = 0; i + 1 < buffer_size; i++) {         \
                asm volatile("" :::);                                              \
                prev_input[i] = prev_input[i + 1];                                \
            }                                                                      \
            prev_input[buffer_size - 1] = curr_input;                             \
            vec_type result = (zero_init);                                         \
            _Pragma("unroll") for (int i = 0; i < _taps; i++) {                   \
                int idx = (buffer_size - 1) * vec_lanes - indices[i];             \
                vec_type contrib;                                                  \
                if (idx % vec_lanes == 0) {                                        \
                    contrib = mul_op(prev_input[idx / vec_lanes], reg[k_regs[i]]); \
                } else {                                                           \
                    contrib = mul_op(                                              \
                        extract_slice(prev_input[idx / vec_lanes],                \
                                      prev_input[idx / vec_lanes + 1],            \
                                      idx % vec_lanes),                            \
                        reg[k_regs[i]]);                                           \
                }                                                                  \
                result = add_op(result, contrib);                                  \
            }                                                                      \
            reg[out_reg] = result;                                                 \
        }                                                                          \
                                                                                    \
        void reset_and_next_row() {                                                \
            for (int i = 0; i < buffer_size; i++)                                  \
                prev_input[i] = (zero_init);                                       \
        }                                                                          \
    };

PushSConvolveOp(, 0,
    [](auto a, auto b) { return a * b; },
    [](auto acc, auto x) { return acc + x; });
PushSConvolveOp(TropMin, std::numeric_limits<elt_type>::max() / 8,
    [](auto a, auto b) { return a + b; },
    [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
PushSConvolveOp(TropMax, std::numeric_limits<elt_type>::min() / 8,
    [](auto a, auto b) { return a + b; },
    [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

#define PushSRecurseOp(suffix, zero_init, mul_op, add_op)                          \
    template <typename vec_type, int out_reg, int in_reg,                          \
              int _taps, int _max_index,                                            \
              const int *indices, const int *k_regs>                               \
    struct PushSRecurse##suffix {                                                   \
        using elt_type = typename ElementType<vec_type>::type;                     \
        constexpr static int vec_lanes = vec_lanes_of(vec_type{});                 \
        constexpr static int buffer_size =                                         \
            (_max_index + vec_lanes - 1) / vec_lanes + 1;                          \
        vec_type prev_output[buffer_size];                                         \
                                                                                    \
        PushSRecurse##suffix() {                                                   \
            for (int i = 0; i < buffer_size; i++)                                  \
                prev_output[i] = (zero_init);                                      \
        }                                                                          \
                                                                                    \
        template <typename Reg_t>                                                  \
        inline __attribute__((always_inline)) void run(Reg_t &reg) {              \
            vec_type curr_output = reg[in_reg];                                    \
            _Pragma("unroll") for (int i = 0; i < _taps; i++) {                   \
                int idx = (buffer_size - 1) * vec_lanes - indices[i];             \
                vec_type contrib;                                                  \
                if (idx % vec_lanes == 0) {                                        \
                    contrib = mul_op(prev_output[idx / vec_lanes], reg[k_regs[i]]); \
                } else {                                                           \
                    contrib = mul_op(                                              \
                        extract_slice(prev_output[idx / vec_lanes],               \
                                      prev_output[idx / vec_lanes + 1],           \
                                      idx % vec_lanes),                            \
                        reg[k_regs[i]]);                                           \
                }                                                                  \
                curr_output = add_op(curr_output, contrib);                        \
            }                                                                      \
            _Pragma("unroll") for (int i = 0; i + 1 < buffer_size; i++) {         \
                asm volatile("" :::);                                              \
                prev_output[i] = prev_output[i + 1];                              \
            }                                                                      \
            prev_output[buffer_size - 1] = curr_output;                           \
            reg[out_reg] = curr_output;                                            \
        }                                                                          \
                                                                                    \
        void reset_and_next_row() {                                                \
            for (int i = 0; i < buffer_size; i++)                                  \
                prev_output[i] = (zero_init);                                      \
        }                                                                          \
    };

PushSRecurseOp(, 0,
    [](auto a, auto b) { return a * b; },
    [](auto acc, auto x) { return acc + x; });
PushSRecurseOp(TropMin, std::numeric_limits<elt_type>::max() / 8,
    [](auto a, auto b) { return a + b; },
    [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
PushSRecurseOp(TropMax, std::numeric_limits<elt_type>::min() / 8,
    [](auto a, auto b) { return a + b; },
    [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

// ---------------------------------------------------------------------------
// Stateless binary / unary ops — each is its own struct so LLVM can see
// the full computation in one function body.
// ---------------------------------------------------------------------------
#define PushBinOp(name, operator)                                            \
    template <typename vec_type, int out_reg, int in_reg1, int in_reg2>      \
    struct name {                                                             \
        template <typename Reg_t>                                            \
        inline __attribute__((always_inline)) void run(Reg_t &reg) {        \
            reg[out_reg] = operator(reg[in_reg1], reg[in_reg2]);             \
        }                                                                    \
        void reset_and_next_row() {}                                         \
    };

PushBinOp(PushSAdd,         [](auto a, auto b) { return a + b; });
PushBinOp(PushSSub,         [](auto a, auto b) { return a - b; });
PushBinOp(PushPointwiseMul, [](auto a, auto b) { return a * b; });
PushBinOp(PushPointwiseDiv, [](auto a, auto b) { return a / b; });
PushBinOp(PushSAddTropMax,  [](auto a, auto b) { return __builtin_elementwise_max(a, b); });
PushBinOp(PushSAddTropMin,  [](auto a, auto b) { return __builtin_elementwise_min(a, b); });
PushBinOp(PushPointwiseMulTropMax, [](auto a, auto b) { return a + b; });
PushBinOp(PushPointwiseMulTropMin, [](auto a, auto b) { return a + b; });

template <typename vec_type, int out_reg, int in_reg>
struct PushSNeg {
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg) {
        reg[out_reg] = -reg[in_reg];
    }
    void reset_and_next_row() {}
};

// ---------------------------------------------------------------------------
// ConvertLanes: wrap a nested sub-driver to adapt lane widths.
//
// PushConvertN2One<vec_in, vec_out, out_reg, inner_out_reg, SubDriver>
//   Factor = vec_out_lanes / vec_in_lanes narrow calls → one wide output.
//   Calls SubDriver factor times, assembles wide result.
//
// PushConvertOne2N<vec_in, vec_out, out_reg, inner_out_reg, SubDriver>
//   Factor = vec_in_lanes / vec_out_lanes.  One wide SubDriver call is
//   spread over factor narrow outputs.
// ---------------------------------------------------------------------------
template <typename vec_in, typename vec_out, int out_reg, int inner_out_reg,
          typename SubDriver>
struct PushConvertN2One {
    constexpr static int factor = vec_lanes_of(vec_out{}) / vec_lanes_of(vec_in{});
    static_assert(vec_lanes_of(vec_out{}) % vec_lanes_of(vec_in{}) == 0,
                  "lane count must be divisible");

    SubDriver sub;

    explicit PushConvertN2One(SubDriver sub_arg) : sub(sub_arg) {}

    template <typename OuterReg_t>
    inline __attribute__((always_inline)) void run(OuterReg_t &outer_reg) {
        vec_out accumulated{};
        typename SubDriver::reg_type inner_reg;
#pragma unroll
        for (int i = 0; i < factor; i++) {
            sub.run(inner_reg);
            reinterpret_cast<vec_in *>(&accumulated)[i] = inner_reg[inner_out_reg];
        }
        outer_reg[out_reg] = accumulated;
    }

    void reset_and_next_row() { sub.reset_and_next_row(); }
};

template <typename vec_in, typename vec_out, int out_reg, int inner_out_reg,
          typename SubDriver>
struct PushConvertOne2N {
    constexpr static int factor = vec_lanes_of(vec_in{}) / vec_lanes_of(vec_out{});
    static_assert(vec_lanes_of(vec_in{}) % vec_lanes_of(vec_out{}) == 0,
                  "lane count must be divisible");

    SubDriver sub;
    vec_out out_buffer[factor > 0 ? factor : 1];
    int offset;

    explicit PushConvertOne2N(SubDriver sub_arg) : sub(sub_arg), offset(-1) {}

    template <typename OuterReg_t>
    inline __attribute__((always_inline)) void run(OuterReg_t &outer_reg) {
        offset = (offset + 1) % factor;
        if (offset == 0) {
            typename SubDriver::reg_type inner_reg;
            sub.run(inner_reg);
            vec_in buffer = inner_reg[inner_out_reg];
            for_<factor>([&](auto i) {
                out_buffer[i.value] =
                    ExtractSubVector<vec_in, vec_out, i.value>::extract_sub_vector(buffer);
            });
        }
        outer_reg[out_reg] = out_buffer[offset];
    }

    void reset_and_next_row() {
        offset = -1;
        sub.reset_and_next_row();
    }
};

// ---------------------------------------------------------------------------
// PushInputSignal2D: reads from a user-provided 2D array (causal, look-behind)
// and writes n_rows consecutive registers starting at out_base_reg.
// d[0] = current row, d[1] = previous row, …, d[n_rows-1] = (n_rows-1) rows back.
// ---------------------------------------------------------------------------
template <typename vec_type, int out_base_reg, int n_rows>
struct PushInputSignal2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);

    const elt_type *data;
    int stride;
    int current_row;
    elt_type *zero_row;
    const elt_type *row_ptrs[n_rows];

    PushInputSignal2D(const elt_type *data_, int stride_, elt_type zero)
        : data(data_), stride(stride_), current_row(0),
          zero_row(new elt_type[stride_]) {
        for (int i = 0; i < stride_; i++)
            zero_row[i] = zero;
        row_ptrs[0] = data - lanes;
        for (int i = 1; i < n_rows; i++)
            row_ptrs[i] = zero_row - lanes;
    }

    ~PushInputSignal2D() { delete[] zero_row; }

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg) {
#pragma unroll
        for (int i = 0; i < n_rows; i++) {
            row_ptrs[i] += lanes;
            memcpy(&reg[out_base_reg + i], row_ptrs[i], sizeof(vec_type));
        }
    }

    void reset_and_next_row() {
        current_row++;
        for (int i = 0; i < n_rows; i++) {
            int src_row = current_row - i;
            row_ptrs[i] = ((src_row >= 0) ? data + (size_t)src_row * stride : zero_row) - lanes;
        }
    }
};

// ---------------------------------------------------------------------------
// PushRepeater: wraps a nested InnerDriver whose RVar2D slots are pre-filled
// from a RepeaterContext ring buffer before each call.
//
//   n_rows         : total rows stored (current + previous)
//   out_base_reg   : outer register where n_rows rows are written
//   rv_base_reg    : first register in inner reg pre-filled with prev-row data
//   inner_out_reg  : register in inner reg that holds the computed current row
// ---------------------------------------------------------------------------
template <typename vec_type, int n_rows, int out_base_reg,
          int rv_base_reg, int inner_out_reg, typename InnerDriver>
struct PushRepeater {
    using elt_type = typename ElementType<vec_type>::type;

    RepeaterContext<vec_type, n_rows> context;
    InnerDriver inner;

    PushRepeater(int row_size, InnerDriver inner_arg)
        : context(row_size), inner(inner_arg) {}

    template <typename OuterReg_t>
    inline void run(OuterReg_t &outer_reg) {
        context.advance_col();
        typename InnerDriver::reg_type inner_reg;
        // Pre-fill previous-row register slots from the ring buffer
        for (int i = 0; i < n_rows - 1; i++)
            inner_reg[rv_base_reg + i] = context.get(i + 1);
        // Run the inner push driver to compute the current row
        inner.run(inner_reg);
        // Store result and expose all rows to the outer register file
        context.set(inner_reg[inner_out_reg]);
        for (int i = 0; i < n_rows; i++)
            outer_reg[out_base_reg + i] = context.get(i);
    }

    void reset_and_next_row() {
        context.next_row();
        inner.reset_and_next_row();
    }
};

// ---------------------------------------------------------------------------
// PushVectorAt: copies reg[in_reg] to reg[out_reg].
// Used to extract a single row from a multi-row output (Ith, VectorAt).
// ---------------------------------------------------------------------------
template <typename vec_type, int out_reg, int in_reg>
struct PushVectorAt {
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg) {
        reg[out_reg] = reg[in_reg];
    }
    void reset_and_next_row() {}
};

// ---------------------------------------------------------------------------
// PushSRecurseV2: mutually-recursive 2-vector IIR (push form).
//
// Computes F = A*F + G where F,G ∈ R^2 and A is a 2×2 kernel matrix:
//   F[0] = K00*F[0] + K01*F[1] + G[0]   (diagonal + cross)
//   F[1] = K10*F[0] + K11*F[1] + G[1]
//
// Register parameters:
//   f0_reg, f1_reg : output registers (written)
//   g0_reg, g1_reg : input registers (read before computation)
//   tapsXY, maxXY  : tap count and max time-index for kernel A[X][Y]
//   kbXY           : base register for A[X][Y]'s consecutive tap values
//   idxXY          : compile-time array of time-delay indices for A[X][Y]
// ---------------------------------------------------------------------------
template <typename vec_type,
          int f0_reg, int f1_reg, int g0_reg, int g1_reg,
          int taps00, int max00, int kb00, const int *idx00,
          int taps01, int max01, int kb01, const int *idx01,
          int taps10, int max10, int kb10, const int *idx10,
          int taps11, int max11, int kb11, const int *idx11>
struct PushSRecurseV2 {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int vec_lanes = vec_lanes_of(vec_type{});

    // Buffer sizes (at least 1 to avoid zero-size arrays)
    constexpr static int bs00 = (taps00 > 0) ? (max00 + vec_lanes - 1) / vec_lanes + 1 : 1;
    constexpr static int bs01 = (taps01 > 0) ? (max01 + vec_lanes - 1) / vec_lanes + 1 : 1;
    constexpr static int bs10 = (taps10 > 0) ? (max10 + vec_lanes - 1) / vec_lanes + 1 : 1;
    constexpr static int bs11 = (taps11 > 0) ? (max11 + vec_lanes - 1) / vec_lanes + 1 : 1;

    vec_type prev_f0[bs00]; // diagonal F[0] output history
    vec_type prev_f1[bs11]; // diagonal F[1] output history
    vec_type hist_f1[bs01]; // history of F[1] for cross-term K01*F[1]
    vec_type hist_f0[bs10]; // history of F[0] for cross-term K10*F[0]
    vec_type last_out0{0}, last_out1{0};

    PushSRecurseV2() {
        for (int i = 0; i < bs00; i++) prev_f0[i] = 0;
        for (int i = 0; i < bs11; i++) prev_f1[i] = 0;
        for (int i = 0; i < bs01; i++) hist_f1[i] = 0;
        for (int i = 0; i < bs10; i++) hist_f0[i] = 0;
    }

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg) {
        // --- Cross-term K01 * F[1] contributes to F[0] ---
        _Pragma("unroll") for (int i = 0; i + 1 < bs01; i++) {
            asm volatile("" :::);
            hist_f1[i] = hist_f1[i + 1];
        }
        hist_f1[bs01 - 1] = last_out1;
        vec_type cross0 = 0;
        if constexpr (taps01 > 0) {
            _Pragma("unroll") for (int i = 0; i < taps01; i++) {
                int idx = (bs01 - 1) * vec_lanes - idx01[i];
                vec_type slice;
                if (idx % vec_lanes == 0)
                    slice = hist_f1[idx / vec_lanes];
                else
                    slice = extract_slice(hist_f1[idx / vec_lanes], hist_f1[idx / vec_lanes + 1], idx % vec_lanes);
                cross0 += slice * reg[kb01 + i];
            }
        }

        // --- Cross-term K10 * F[0] contributes to F[1] ---
        _Pragma("unroll") for (int i = 0; i + 1 < bs10; i++) {
            asm volatile("" :::);
            hist_f0[i] = hist_f0[i + 1];
        }
        hist_f0[bs10 - 1] = last_out0;
        vec_type cross1 = 0;
        if constexpr (taps10 > 0) {
            _Pragma("unroll") for (int i = 0; i < taps10; i++) {
                int idx = (bs10 - 1) * vec_lanes - idx10[i];
                vec_type slice;
                if (idx % vec_lanes == 0)
                    slice = hist_f0[idx / vec_lanes];
                else
                    slice = extract_slice(hist_f0[idx / vec_lanes], hist_f0[idx / vec_lanes + 1], idx % vec_lanes);
                cross1 += slice * reg[kb10 + i];
            }
        }

        // --- Effective inputs ---
        vec_type eff_g0 = reg[g0_reg] + cross0;
        vec_type eff_g1 = reg[g1_reg] + cross1;

        // --- Diagonal IIR for F[0] (K00 * F[0]) ---
        vec_type f0 = eff_g0;
        if constexpr (taps00 > 0) {
            _Pragma("unroll") for (int i = 0; i < taps00; i++) {
                int idx = (bs00 - 1) * vec_lanes - idx00[i];
                vec_type slice;
                if (idx % vec_lanes == 0)
                    slice = prev_f0[idx / vec_lanes];
                else
                    slice = extract_slice(prev_f0[idx / vec_lanes], prev_f0[idx / vec_lanes + 1], idx % vec_lanes);
                f0 += slice * reg[kb00 + i];
            }
        }
        _Pragma("unroll") for (int i = 0; i + 1 < bs00; i++) {
            asm volatile("" :::);
            prev_f0[i] = prev_f0[i + 1];
        }
        prev_f0[bs00 - 1] = f0;

        // --- Diagonal IIR for F[1] (K11 * F[1]) ---
        vec_type f1 = eff_g1;
        if constexpr (taps11 > 0) {
            _Pragma("unroll") for (int i = 0; i < taps11; i++) {
                int idx = (bs11 - 1) * vec_lanes - idx11[i];
                vec_type slice;
                if (idx % vec_lanes == 0)
                    slice = prev_f1[idx / vec_lanes];
                else
                    slice = extract_slice(prev_f1[idx / vec_lanes], prev_f1[idx / vec_lanes + 1], idx % vec_lanes);
                f1 += slice * reg[kb11 + i];
            }
        }
        _Pragma("unroll") for (int i = 0; i + 1 < bs11; i++) {
            asm volatile("" :::);
            prev_f1[i] = prev_f1[i + 1];
        }
        prev_f1[bs11 - 1] = f1;

        // --- Write outputs ---
        reg[f0_reg] = f0;
        reg[f1_reg] = f1;
        last_out0 = f0;
        last_out1 = f1;
    }

    void reset_and_next_row() {
        last_out0 = 0;
        last_out1 = 0;
        for (int i = 0; i < bs00; i++) prev_f0[i] = 0;
        for (int i = 0; i < bs11; i++) prev_f1[i] = 0;
        for (int i = 0; i < bs01; i++) hist_f1[i] = 0;
        for (int i = 0; i < bs10; i++) hist_f0[i] = 0;
    }
};
