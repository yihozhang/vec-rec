#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>
typedef float float_vec1;
typedef float float_vec2 __attribute__((ext_vector_type(2)));
typedef float float_vec4 __attribute__((ext_vector_type(4)));
typedef float float_vec8 __attribute__((ext_vector_type(8)));
typedef float float_vec16 __attribute__((ext_vector_type(16)));

typedef int32_t int32_vec1;
typedef int32_t int32_vec2 __attribute__((ext_vector_type(2)));
typedef int32_t int32_vec4 __attribute__((ext_vector_type(4)));
typedef int32_t int32_vec8 __attribute__((ext_vector_type(8)));
typedef int32_t int32_vec16 __attribute__((ext_vector_type(16)));

typedef int64_t int64_vec1;
typedef int64_t int64_vec2 __attribute__((ext_vector_type(2)));
typedef int64_t int64_vec4 __attribute__((ext_vector_type(4)));
typedef int64_t int64_vec8 __attribute__((ext_vector_type(8)));

constexpr int vec_lanes_of(float) { return 1; }
constexpr int vec_lanes_of(float_vec2) { return 2; }
constexpr int vec_lanes_of(float_vec4) { return 4; }
constexpr int vec_lanes_of(float_vec8) { return 8; }
constexpr int vec_lanes_of(float_vec16) { return 16; }

constexpr int vec_lanes_of(int32_t) { return 1; }
constexpr int vec_lanes_of(int32_vec2) { return 2; }
constexpr int vec_lanes_of(int32_vec4) { return 4; }
constexpr int vec_lanes_of(int32_vec8) { return 8; }
constexpr int vec_lanes_of(int32_vec16) { return 16; }

constexpr int vec_lanes_of(int64_t) { return 1; }
constexpr int vec_lanes_of(int64_vec2) { return 2; }
constexpr int vec_lanes_of(int64_vec4) { return 4; }
constexpr int vec_lanes_of(int64_vec8) { return 8; }

#include <cmath>
template <typename Vec>
inline Vec safe_mul(Vec a, Vec b) {
    Vec prod = a * b;

    // Detect NaN: x != x
    auto isnan_a = (a != a);
    auto isnan_b = (b != b);

    // Zero checks
    using Scalar = std::remove_reference_t<decltype(a[0])>;
    auto a_zero = (a == (Scalar)0);
    auto b_zero = (b == (Scalar)0);

    // Mask: (isnan(a) && b == 0) || (isnan(b) && a == 0)
    auto mask = (isnan_a & b_zero) | (isnan_b & a_zero);

    // Blend: mask ? 0 : prod   (elementwise)
    return mask ? Vec{} : prod;
}

template <typename Vec>
inline Vec safe_div(Vec a, Vec b) {
    Vec quot = a / b;

    using Scalar = std::remove_reference_t<decltype(a[0])>;

    // Detect zeros (works for +0 and -0)
    auto a_zero = (a == (Scalar)0);
    auto b_zero = (b == (Scalar)0);

    // Mask for 0/0
    auto mask = a_zero & b_zero;

    // Blend: if (0/0) -> 1, else normal division
    return mask ? Vec{(Scalar)1} : quot;
}

template <typename T>
struct ElementType {};

template <>
struct ElementType<float> {
    using type = float;
};
template <>
struct ElementType<float_vec2> {
    using type = float;
};
template <>
struct ElementType<float_vec4> {
    using type = float;
};
template <>
struct ElementType<float_vec8> {
    using type = float;
};
template <>
struct ElementType<float_vec16> {
    using type = float;
};

template <>
struct ElementType<int32_t> {
    using type = int32_t;
};
template <>
struct ElementType<int32_vec2> {
    using type = int32_t;
};
template <>
struct ElementType<int32_vec4> {
    using type = int32_t;
};
template <>
struct ElementType<int32_vec8> {
    using type = int32_t;
};
template <>
struct ElementType<int32_vec16> {
    using type = int32_t;
};

template <>
struct ElementType<int64_t> {
    using type = int64_t;
};
template <>
struct ElementType<int64_vec2> {
    using type = int64_t;
};
template <>
struct ElementType<int64_vec4> {
    using type = int64_t;
};
template <>
struct ElementType<int64_vec8> {
    using type = int64_t;
};

inline std::string string_of(float_vec4 x) {
    return std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) + ", " + std::to_string(x[3]);
}

inline std::string string_of(float_vec8 x) {
    return std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) + ", " + std::to_string(x[3]) + ", " +
           std::to_string(x[5]) + ", " + std::to_string(x[6]) + ", " + std::to_string(x[7]) + ", " + std::to_string(x[8]);
}

#define EXTRACT_SLICE_1(type)                               \
    inline type extract_slice(type a, type b, int offset) { \
        switch (offset) {                                   \
        case 0:                                             \
            return a;                                       \
        case 1:                                             \
            return b;                                       \
        };                                                  \
        assert(false);                                      \
        return type{};                                      \
    }

#define EXTRACT_SLICE_2(type_vec2)                                         \
    inline type_vec2 extract_slice(type_vec2 a, type_vec2 b, int offset) { \
        switch (offset) {                                                  \
        case 0:                                                            \
            return a;                                                      \
        case 1:                                                            \
            return __builtin_shufflevector(a, b, 1, 2);                    \
        case 2:                                                            \
            return b;                                                      \
        };                                                                 \
        assert(false);                                                     \
        return type_vec2{};                                                \
    }

#define EXTRACT_SLICE_4(type_vec4)                                         \
    inline type_vec4 extract_slice(type_vec4 a, type_vec4 b, int offset) { \
        switch (offset) {                                                  \
        case 0:                                                            \
            return a;                                                      \
        case 1:                                                            \
            return __builtin_shufflevector(a, b, 1, 2, 3, 4);              \
        case 2:                                                            \
            return __builtin_shufflevector(a, b, 2, 3, 4, 5);              \
        case 3:                                                            \
            return __builtin_shufflevector(a, b, 3, 4, 5, 6);              \
        case 4:                                                            \
            return b;                                                      \
        };                                                                 \
        assert(false);                                                     \
        return type_vec4{};                                                \
    }

#define EXTRACT_SLICE_8(type_vec8)                                             \
    inline type_vec8 extract_slice(type_vec8 a, type_vec8 b, int offset) {     \
        switch (offset) {                                                      \
        case 0:                                                                \
            return a;                                                          \
        case 1:                                                                \
            return __builtin_shufflevector(a, b, 1, 2, 3, 4, 5, 6, 7, 8);      \
        case 2:                                                                \
            return __builtin_shufflevector(a, b, 2, 3, 4, 5, 6, 7, 8, 9);      \
        case 3:                                                                \
            return __builtin_shufflevector(a, b, 3, 4, 5, 6, 7, 8, 9, 10);     \
        case 4:                                                                \
            return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8, 9, 10, 11);    \
        case 5:                                                                \
            return __builtin_shufflevector(a, b, 5, 6, 7, 8, 9, 10, 11, 12);   \
        case 6:                                                                \
            return __builtin_shufflevector(a, b, 6, 7, 8, 9, 10, 11, 12, 13);  \
        case 7:                                                                \
            return __builtin_shufflevector(a, b, 7, 8, 9, 10, 11, 12, 13, 14); \
        case 8:                                                                \
            return b;                                                          \
        };                                                                     \
        assert(false);                                                         \
        return type_vec8{};                                                    \
    }

#define EXTRACT_SLICE_16(type_vec16)                                                                              \
    inline type_vec16 extract_slice(type_vec16 a, type_vec16 b, int offset) {                                     \
        switch (offset) {                                                                                         \
        case 0:                                                                                                   \
            return a;                                                                                             \
        case 1:                                                                                                   \
            return __builtin_shufflevector(a, b, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);          \
        case 2:                                                                                                   \
            return __builtin_shufflevector(a, b, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);         \
        case 3:                                                                                                   \
            return __builtin_shufflevector(a, b, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);        \
        case 4:                                                                                                   \
            return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19);       \
        case 5:                                                                                                   \
            return __builtin_shufflevector(a, b, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);      \
        case 6:                                                                                                   \
            return __builtin_shufflevector(a, b, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);     \
        case 7:                                                                                                   \
            return __builtin_shufflevector(a, b, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22);    \
        case 8:                                                                                                   \
            return __builtin_shufflevector(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);   \
        case 9:                                                                                                   \
            return __builtin_shufflevector(a, b, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24);  \
        case 10:                                                                                                  \
            return __builtin_shufflevector(a, b, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25); \
        case 11:                                                                                                  \
            return __builtin_shufflevector(a, b, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26); \
        case 12:                                                                                                  \
            return __builtin_shufflevector(a, b, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27); \
        case 13:                                                                                                  \
            return __builtin_shufflevector(a, b, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28); \
        case 14:                                                                                                  \
            return __builtin_shufflevector(a, b, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29); \
        case 15:                                                                                                  \
            return __builtin_shufflevector(a, b, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30); \
        case 16:                                                                                                  \
            return b;                                                                                             \
        };                                                                                                        \
        assert(false);                                                                                            \
        return type_vec16{};                                                                                      \
    }

EXTRACT_SLICE_1(float)
EXTRACT_SLICE_2(float_vec2)
EXTRACT_SLICE_4(float_vec4)
EXTRACT_SLICE_8(float_vec8)
EXTRACT_SLICE_16(float_vec16)
EXTRACT_SLICE_1(int32_vec1)
EXTRACT_SLICE_2(int32_vec2)
EXTRACT_SLICE_4(int32_vec4)
EXTRACT_SLICE_8(int32_vec8)
EXTRACT_SLICE_16(int32_vec16)
EXTRACT_SLICE_1(int64_vec1)
EXTRACT_SLICE_2(int64_vec2)
EXTRACT_SLICE_4(int64_vec4)
EXTRACT_SLICE_8(int64_vec8)

// from https://stackoverflow.com/questions/37602057/why-isnt-a-for-loop-a-compile-time-expression
template <std::size_t N>
struct num {
    static const constexpr auto value = N;
};

template <class F, std::size_t... Is>
void for_(F func, std::index_sequence<Is...>) {
    (func(num<Is>{}), ...);
}

template <std::size_t N, typename F>
void for_(F func) {
    for_(func, std::make_index_sequence<N>());
}

template <typename vec_type_in, typename vec_type_out, int n>
struct ExtractSubVector {};

#define EXTRACT_SUB_VECTOR1(type_vec1)                       \
    template <typename vec_type_in, int n>                   \
    struct ExtractSubVector<vec_type_in, type_vec1, n> {     \
        static type_vec1 extract_sub_vector(vec_type_in a) { \
            return a[n];                                     \
        }                                                    \
    };

#define EXTRACT_SUB_VECTOR2(type_vec2)                              \
    template <typename vec_type_in, int n>                          \
    struct ExtractSubVector<vec_type_in, type_vec2, n> {            \
        static type_vec2 extract_sub_vector(vec_type_in a) {        \
            return __builtin_shufflevector(a, a, 2 * n, 2 * n + 1); \
        }                                                           \
    };

#define EXTRACT_SUB_VECTOR4(type_vec4)                                                    \
    template <typename vec_type_in, int n>                                                \
    struct ExtractSubVector<vec_type_in, type_vec4, n> {                                  \
        static type_vec4 extract_sub_vector(vec_type_in a) {                              \
            return __builtin_shufflevector(a, a, 4 * n, 4 * n + 1, 4 * n + 2, 4 * n + 3); \
        }                                                                                 \
    };

#define EXTRACT_SUB_VECTOR8(type_vec8)                                                                                                \
    template <typename vec_type_in, int n>                                                                                            \
    struct ExtractSubVector<vec_type_in, type_vec8, n> {                                                                              \
        static type_vec8 extract_sub_vector(vec_type_in a) {                                                                          \
            return __builtin_shufflevector(a, a, 8 * n, 8 * n + 1, 8 * n + 2, 8 * n + 3, 8 * n + 4, 8 * n + 5, 8 * n + 6, 8 * n + 7); \
        }                                                                                                                             \
    };

EXTRACT_SUB_VECTOR1(float)
EXTRACT_SUB_VECTOR2(float_vec2)
EXTRACT_SUB_VECTOR4(float_vec4)
EXTRACT_SUB_VECTOR8(float_vec8)
EXTRACT_SUB_VECTOR1(int32_t)
EXTRACT_SUB_VECTOR2(int32_vec2)
EXTRACT_SUB_VECTOR4(int32_vec4)
EXTRACT_SUB_VECTOR8(int32_vec8)
EXTRACT_SUB_VECTOR1(int64_t)
EXTRACT_SUB_VECTOR2(int64_vec2)
EXTRACT_SUB_VECTOR4(int64_vec4)

// ZeroKernel: a kernel with no taps (contributes nothing to the output).
// Used when a position in a KernelMatrix has no feedback.
template <typename vec_type>
struct ZeroKernel {
    constexpr static int taps = 0;
    constexpr static const int *idxs = nullptr;
    void run(vec_type * /*out*/) {}
    void peek(vec_type * /*out*/) {}
    void reset_and_next_row() {}
};

template <int _taps, typename vec_type, const int indices[_taps], const typename ElementType<vec_type>::type vals[_taps]>
struct TimeInvariantKernel {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static const int *idxs = indices;
    constexpr static int taps = _taps;

    TimeInvariantKernel() {}

    inline __attribute__((always_inline)) void run(vec_type *out) {
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = vals[i];
        }
    }

    inline __attribute__((always_inline)) void peek(vec_type *out) {
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = vals[i];
        }
    }

    void reset_and_next_row() {}
};

template <int taps, typename vec_type, int curr, typename... Args>
void write_coeff(vec_type *coeff, std::tuple<Args *...> &args) {
    if constexpr (curr < taps) {
        vec_type result;
        std::get<curr>(args)->run(&result);
        coeff[curr] = result;
        write_coeff<taps, vec_type, curr + 1, Args...>(coeff, args);
    }
}

template <int _taps, typename vec_type, const int indices[_taps], typename... Args>
struct TimeVaryingKernel {
    constexpr static const int *idxs = indices;
    constexpr static int taps = _taps;
    std::tuple<Args *...> args;
    vec_type current_output[taps];

    TimeVaryingKernel(Args *...args) : args(args...) {}

    inline __attribute__((always_inline)) void run(vec_type *out) {
        write_coeff<taps, vec_type, 0, Args...>(current_output, args);
        peek(out);
    }

    inline __attribute__((always_inline)) void peek(vec_type *out) {
        _Pragma("unroll") for (int i = 0; i < taps; i++) {
            out[i] = current_output[i];
        }
    }

    void reset_and_next_row() {
        reset_tv_kernel<0>();
    }

private:
    template <int I>
    void reset_tv_kernel() {
        if constexpr (I < taps) {
            std::get<I>(args)->reset_and_next_row();
            reset_tv_kernel<I + 1>();
        }
    }

public:
};

template <typename vec_type>
struct Signal1DConstant {
    using elt_type = typename ElementType<vec_type>::type;

    vec_type data;
    Signal1DConstant(elt_type data) {
        this->data = data;
    }

    // The caller is responsible for ensuring the array access
    // does not go out of bound
    // const vec_type *run() {
    inline __attribute__((always_inline)) void run(vec_type *d) {
        *d = data;
    }

    inline __attribute__((always_inline)) void peek(vec_type *d) {
        *d = data;
    }

    void reset_and_next_row() {}
};

// Signal1DImpulse: produces `value` in lane 0 at time 0, then `zero` (broadcast) thereafter.
// Useful for injecting initial conditions into recurrences.
template <typename vec_type>
struct Signal1DImpulse {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int lanes = vec_lanes_of(vec_type{});

    elt_type impulse_value;
    elt_type zero_value;
    vec_type current_output;
    bool fired;

    Signal1DImpulse(elt_type value, elt_type zero) : impulse_value(value), zero_value(zero), fired(false) {}

    inline __attribute__((always_inline)) void run(vec_type *d) {
        if (!fired) {
            fired = true;
            if constexpr (lanes == 1) {
                current_output = impulse_value;
            } else {
                vec_type result = zero_value; // broadcast zero to all lanes
                result[0] = impulse_value;    // override lane 0 with the impulse
                current_output = result;
            }
        } else {
            current_output = zero_value; // broadcast zero to all lanes
        }
        *d = current_output;
    }

    inline __attribute__((always_inline)) void peek(vec_type *d) {
        *d = current_output;
    }

    void reset_and_next_row() {}
};

template <typename vec_type>
struct Signal1D {
    using elt_type = typename ElementType<vec_type>::type;
    // constexpr static int taps = 1;
    // constexpr static const int idxs[1] = {0};

    const vec_type *data;
    Signal1D(const elt_type *data) : data((const vec_type *)data - 1) {}

    // The caller is responsible for ensuring the array access
    // does not go out of bound
    // const vec_type *run() {
    inline __attribute__((always_inline)) void run(vec_type *d) {
        data++;
        memcpy((float *)d, (float *)data, sizeof(vec_type));
    }

    inline __attribute__((always_inline)) void peek(vec_type *d) {
        memcpy((float *)d, (float *)data, sizeof(vec_type));
    }

    // Signal1D streams through contiguous data. Between rows the pointer
    // is already at the start of the next row, so this is a no-op.
    void reset_and_next_row() {}
};

// Context struct to share state between Repeater and Signal2D
template <typename vec_type, int n_rows>
struct RepeaterContext {
    int width;
    vec_type *buffer;
    int current_row;
    int current_col;

    RepeaterContext(int row_size) : width(row_size), current_row(0), current_col(-1) {
        buffer = new vec_type[n_rows * width];
        memset(buffer, 0, sizeof(vec_type) * n_rows * width);
    }

    ~RepeaterContext() {
        delete[] buffer;
    }

    // Get the value at the given row offset and current column
    vec_type get(int row_offset) {
        int row_idx = (current_row - row_offset + n_rows) % n_rows;
        return buffer[row_idx * width + current_col];
    }

    // Store a value in the current row and column
    void set(vec_type value) {
        buffer[current_row * width + current_col] = value;
    }

    void advance_col() {
        current_col++;
        if (current_col >= width) {
            current_col = 0;
        }
    }

    void next_row() {
        current_row = (current_row + 1) % n_rows;
        current_col = -1;
        // memset(buffer[current_row], 0, sizeof(buffer[current_row]));
    }
};

template <typename vec_type, int n_rows>
struct Signal2D {
    using elt_type = typename ElementType<vec_type>::type;

    RepeaterContext<vec_type, n_rows + 1> *context;

    Signal2D(RepeaterContext<vec_type, n_rows + 1> *ctx) : context(ctx) {
    }

    inline __attribute__((always_inline)) void run(vec_type *d) {
        // Read from previous rows at the current position
#pragma unroll
        for (int i = 0; i < n_rows; i++) {
            d[i] = context->get(i + 1);
        }
    }

    inline __attribute__((always_inline)) void peek(vec_type *d) {
        run(d);
    }

    void reset_and_next_row() {}
};

// InputSignal2D: reads from a user-provided 2D array with causal (look-behind) semantics.
// d[0] is the current row, d[1] is the previous row, ..., d[n_rows-1] is n_rows-1 rows back.
// Rows before row 0 are treated as zero (zero-padding).
// Row pointers are kept as elt_type* to avoid alignment assumptions; memcpy
// is used for the actual load so that unaligned input data is handled safely.
template <typename vec_type, int n_rows>
struct InputSignal2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);

    const elt_type *data;
    int stride;
    int current_row;
    elt_type *zero_row;
    // points into data for valid rows, or zero_row.get() for zero-padding rows
    const elt_type *row_ptrs[n_rows];

    InputSignal2D(const elt_type *data, int stride, elt_type zero)
        : data(data),
          stride(stride),
          current_row(0),
          zero_row(new elt_type[stride]) {
        for (int i = 0; i < stride; i++) {
            zero_row[i] = zero;
        }
        // d[0] = row 0 (current); d[1..n_rows-1] = zero (no previous rows yet)
        row_ptrs[0] = data - lanes;
        for (int i = 1; i < n_rows; i++) {
            row_ptrs[i] = zero_row - lanes;
        }
    }

    inline __attribute__((always_inline)) void run(vec_type *d) {
#pragma unroll
        for (int i = 0; i < n_rows; i++) {
            row_ptrs[i] += lanes;
            memcpy(&d[i], row_ptrs[i], sizeof(vec_type));
        }
    }

    inline __attribute__((always_inline)) void peek(vec_type *d) {
#pragma unroll
        for (int i = 0; i < n_rows; i++) {
            memcpy(&d[i], row_ptrs[i], sizeof(vec_type));
        }
    }

    void reset_and_next_row() {
        current_row++;
        // d[i] = row (current_row - i), zero-padded for rows before 0
        for (int i = 0; i < n_rows; i++) {
            int src_row = current_row - i;
            row_ptrs[i] = ((src_row >= 0) ? data + (size_t)src_row * stride : zero_row) - lanes;
        }
    }
};

template <typename vec_type, int n_rows>
auto make_input_signal2d(const typename ElementType<vec_type>::type *data, int stride, typename ElementType<vec_type>::type zero) {
    return new InputSignal2D<vec_type, n_rows>(data, stride, zero);
}

template <typename vec_type, int n_rows, typename Inner>
struct Repeater {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int vec_lanes = vec_lanes_of(vec_type{});

    RepeaterContext<vec_type, n_rows> *context;
    Inner *inner;

    Repeater(RepeaterContext<vec_type, n_rows> *ctx, Inner *inner)
        : context(ctx), inner(inner) {
    }

    inline __attribute__((always_inline)) void run(vec_type *out) {
        // Advance column position
        context->advance_col();

        // Compute the current row output using the inner signal
        // (which may read from the buffered previous rows via Signal2D)
        vec_type curr_output;
        inner->run(&curr_output);

        // Store the current output in the buffer
        context->set(curr_output);

        // Return n_rows vectors: current row at out[0], previous rows at out[1..n-1]
        // This makes Repeater stream as a 2D signal
        for (int i = 0; i < n_rows; i++) {
            out[i] = context->get(i);
        }
    }

    inline __attribute__((always_inline)) void peek(vec_type *out) {
        // Return n_rows vectors: current row at out[0], previous rows at out[1..n-1]
        for (int i = 0; i < n_rows; i++) {
            out[i] = context->get(i);
        }
    }

    void reset_and_next_row() {
        // Move to the next row
        context->next_row();

        // Reset the inner signal
        inner->reset_and_next_row();
    }
};

#define SConvolveOp(suffix, zero_init, mul_op, add_op)                             \
    template <typename vec_type, typename K, typename S>                           \
    struct SConvolve##suffix {                                                     \
        using elt_type = typename ElementType<vec_type>::type;                     \
        K *kernel;                                                                 \
        S *signal;                                                                 \
        constexpr static int vec_lanes = vec_lanes_of(vec_type{});                 \
        constexpr static int buffer_size =                                         \
            (K::idxs[K::taps - 1] + vec_lanes - 1) / vec_lanes + 1;                \
        vec_type prev_input[buffer_size];                                          \
        vec_type curr_output;                                                      \
        SConvolve##suffix(K *kernel, S *signal) : kernel(kernel), signal(signal) { \
            for (int i = 0; i < buffer_size; i++)                                  \
                prev_input[i] = (zero_init);                                       \
        }                                                                          \
        inline __attribute__((always_inline)) void run(vec_type *d) {              \
            curr_output = (zero_init);                                             \
            vec_type curr_input;                                                   \
            signal->run(&curr_input);                                              \
            _Pragma("unroll") for (int i = 0; i + 1 < buffer_size; i++) {          \
                asm volatile("" :::);                                              \
                prev_input[i] = prev_input[i + 1];                                 \
            }                                                                      \
            prev_input[buffer_size - 1] = curr_input;                              \
            vec_type k[K::taps];                                                   \
            kernel->run(k);                                                        \
            _Pragma("unroll") for (int i = 0; i < K::taps; i++) {                  \
                int idx = (buffer_size - 1) * vec_lanes - K::idxs[i];              \
                vec_type contrib;                                                  \
                if (idx % vec_lanes == 0) {                                        \
                    contrib = mul_op(prev_input[idx / vec_lanes], k[i]);           \
                } else {                                                           \
                    vec_type va = prev_input[idx / vec_lanes];                     \
                    vec_type vb = prev_input[idx / vec_lanes + 1];                 \
                    contrib = mul_op(                                              \
                        extract_slice(va, vb, idx % vec_lanes), k[i]);             \
                }                                                                  \
                curr_output = add_op(curr_output, contrib);                        \
            }                                                                      \
            *d = curr_output;                                                      \
        }                                                                          \
                                                                                   \
        inline __attribute__((always_inline)) void peek(vec_type *d) {             \
            *d = curr_output;                                                      \
        }                                                                          \
                                                                                   \
        void reset_and_next_row() {                                                \
            for (int i = 0; i < buffer_size; i++)                                  \
                prev_input[i] = (zero_init);                                       \
            kernel->reset_and_next_row();                                          \
            signal->reset_and_next_row();                                          \
        }                                                                          \
    };                                                                             \
    template <typename vec_type, typename K, typename S>                           \
    auto make_s_convolve##suffix(K *kernel, S *signal) {                           \
        return new SConvolve##suffix<vec_type, K, S>(kernel, signal);              \
    }

SConvolveOp(, 0, [](auto a, auto b) { return a * b; }, [](auto acc, auto x) { return acc + x; });
// SConvolveOp(, 0, [](auto a, auto b) { return safe_mul(a, b); }, [](auto acc, auto x) { return acc + x; });
SConvolveOp(TropMin, std::numeric_limits<elt_type>::max() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SConvolveOp(TropMax, std::numeric_limits<elt_type>::min() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

#define SRecurseOp(suffix, zero_init, mul_op, add_op)                             \
    template <typename vec_type, typename K, typename S>                          \
    struct SRecurse##suffix {                                                     \
        using elt_type = typename ElementType<vec_type>::type;                    \
        K *kernel;                                                                \
        S *signal;                                                                \
        constexpr static int vec_lanes = vec_lanes_of(vec_type{});                \
        constexpr static int buffer_size =                                        \
            (K::idxs[K::taps - 1] + vec_lanes - 1) / vec_lanes + 1;               \
        vec_type prev_output[buffer_size];                                        \
        vec_type curr_output;                                                     \
        SRecurse##suffix(K *kernel, S *signal) : kernel(kernel), signal(signal) { \
            for (int i = 0; i < buffer_size; i++)                                 \
                prev_output[i] = (zero_init);                                     \
        }                                                                         \
        inline __attribute__((always_inline)) void run(vec_type *out) {           \
            vec_type curr_input;                                                  \
            vec_type k[K::taps];                                                  \
            kernel->run(k);                                                       \
            signal->run(&curr_input);                                             \
            curr_output = curr_input;                                             \
            _Pragma("unroll") for (int i = 0; i < K::taps; i++) {                 \
                int idx = (buffer_size - 1) * vec_lanes - K::idxs[i];             \
                vec_type contrib;                                                 \
                if (idx % vec_lanes == 0) {                                       \
                    contrib = mul_op(prev_output[idx / vec_lanes], k[i]);         \
                } else {                                                          \
                    auto va = prev_output[idx / vec_lanes];                       \
                    auto vb = prev_output[idx / vec_lanes + 1];                   \
                    contrib = mul_op(                                             \
                        extract_slice(va, vb, idx % vec_lanes), k[i]);            \
                }                                                                 \
                curr_output = add_op(curr_output, contrib);                       \
            }                                                                     \
            _Pragma("unroll") for (int i = 0; i + 1 < buffer_size; i++) {         \
                asm volatile("" :::);                                             \
                prev_output[i] = prev_output[i + 1];                              \
            }                                                                     \
            prev_output[buffer_size - 1] = curr_output;                           \
            *out = curr_output;                                                   \
        }                                                                         \
                                                                                  \
        inline __attribute__((always_inline)) void peek(vec_type *out) {          \
            *out = curr_output;                                                   \
        }                                                                         \
                                                                                  \
        void reset_and_next_row() {                                               \
            for (int i = 0; i < buffer_size; i++)                                 \
                prev_output[i] = (zero_init);                                     \
            kernel->reset_and_next_row();                                         \
            signal->reset_and_next_row();                                         \
        }                                                                         \
    };                                                                            \
    template <typename vec_type, typename K, typename S>                          \
    auto make_s_recurse##suffix(K *kernel, S *signal) {                           \
        return new SRecurse##suffix<vec_type, K, S>(kernel, signal);              \
    }

SRecurseOp(, 0, [](auto a, auto b) { return a * b; }, [](auto acc, auto x) { return acc + x; });
// SRecurseOp(, 0, [](auto a, auto b) { return safe_mul(a, b); }, [](auto acc, auto x) { return acc + x; });
SRecurseOp(TropMin, std::numeric_limits<elt_type>::max() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SRecurseOp(TropMax, std::numeric_limits<elt_type>::min() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

template <typename vec_type_in, typename vec_type_out, typename Inner>
struct KConvertN2One {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_out / lanes_in;
    static_assert(lanes_out % lanes_in == 0, "lanes_in must be multiple of lanes_out");
    constexpr static int taps = Inner::taps;
    constexpr static const int *idxs = Inner::idxs;

    Inner *inner;
    vec_type_out current_output[taps];

    KConvertN2One(Inner *inner) : inner(inner) {}
    inline __attribute__((always_inline)) void run(vec_type_out *out) {
        vec_type_in *p = (vec_type_in *)current_output;
#pragma unroll
        for (int i = 0; i < factor; i++) {
            vec_type_in in[taps];
            inner->run(in);
#pragma unroll
            for (int j = 0; j < taps; j++) {
                p[j * factor + i] = in[j];
            }
        }
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = current_output[i];
        }
    }

    inline __attribute__((always_inline)) void peek(vec_type_out *out) {
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = current_output[i];
        }
    }

    void reset_and_next_row() { inner->reset_and_next_row(); }
};

template <typename vec_type_in, typename vec_type_out, typename Inner>
struct ConvertN2One {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_out / lanes_in;
    static_assert(lanes_out % lanes_in == 0, "lanes_in must be multiple of lanes_out");

    Inner *inner;
    vec_type_out current_output;

    ConvertN2One(Inner *inner) : inner(inner) {}
    inline __attribute__((always_inline)) void run(vec_type_out *out) {

        vec_type_in *p = (vec_type_in *)&current_output;
#pragma unroll
        for (int i = 0; i < factor; i++) {
            vec_type_in in;
            inner->run(&in);
            *p++ = in;
        }
        *out = current_output;
    }

    inline __attribute__((always_inline)) void peek(vec_type_out *out) {
        *out = current_output;
    }

    void reset_and_next_row() { inner->reset_and_next_row(); }
};

template <typename vec_type_in, typename vec_type_out, typename Inner>
struct KConvertOne2N {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_in / lanes_out;
    static_assert(lanes_in % lanes_out == 0, "lanes_out must be multiple of lanes_in");
    constexpr static int taps = Inner::taps;
    constexpr static const int *idxs = Inner::idxs;

    Inner *inner;
    vec_type_in buffer[taps];
    vec_type_out out_buffer[taps][factor];
    int offset;

    KConvertOne2N(Inner *inner) : inner(inner), offset(-1) {}

    inline __attribute__((always_inline)) void run(vec_type_out *out) {
        offset = (offset + 1) % factor;
        if (offset == 0) {
            inner->run(buffer);
#pragma unroll
            for (int i = 0; i < taps; i++) {
                for_<factor>([&](auto j) {
                    out_buffer[i][j.value] = ExtractSubVector<vec_type_in, vec_type_out, j.value>::extract_sub_vector(buffer[i]);
                });
            }
        }
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = out_buffer[i][offset];
        }
    }

    inline __attribute__((always_inline)) void peek(vec_type_out *out) {
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = out_buffer[i][offset];
        }
    }

    void reset_and_next_row() {
        offset = -1;
        inner->reset_and_next_row();
    }
};

template <typename vec_type_in, typename vec_type_out, typename Inner>
struct ConvertOne2N {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_in / lanes_out;
    static_assert(lanes_in % lanes_out == 0, "lanes_out must be multiple of lanes_in");

    Inner *inner;
    vec_type_in buffer;
    vec_type_out out_buffer[factor];
    int offset;

    ConvertOne2N(Inner *inner) : inner(inner), offset(-1) {}

    inline __attribute__((always_inline)) void run(vec_type_out *out) {
        offset = (offset + 1) % factor;
        if (offset == 0) {
            inner->run(&buffer);

            for_<factor>([&](auto i) {
                out_buffer[i.value] = ExtractSubVector<vec_type_in, vec_type_out, i.value>::extract_sub_vector(buffer);
            });
        }
        *out = out_buffer[offset];
    }

    inline __attribute__((always_inline)) void peek(vec_type_out *out) {
        *out = out_buffer[offset];
    }

    void reset_and_next_row() {
        offset = -1;
        inner->reset_and_next_row();
    }
};

#define BinOp(name, operator)                                            \
    template <typename vec_type, typename S1, typename S2>               \
    struct name {                                                        \
        S1 *s1;                                                          \
        S2 *s2;                                                          \
        vec_type current_output;                                         \
        name(S1 *s1, S2 *s2) : s1(s1), s2(s2) {}                         \
                                                                         \
        inline __attribute__((always_inline)) void run(vec_type *out) {  \
            vec_type left, right;                                        \
            s1->run(&left);                                              \
            s2->run(&right);                                             \
            current_output = operator(left, right);                      \
            *out = current_output;                                       \
        }                                                                \
        inline __attribute__((always_inline)) void peek(vec_type *out) { \
            *out = current_output;                                       \
        }                                                                \
                                                                         \
        void reset_and_next_row() {                                      \
            s1->reset_and_next_row();                                    \
            s2->reset_and_next_row();                                    \
        }                                                                \
    };                                                                   \
    template <typename vec_type, typename S1, typename S2>               \
    auto make_##name(S1 *s1, S2 *s2) { return new name<vec_type, S1, S2>(s1, s2); }

BinOp(SAdd, [](auto left, auto right) { return left + right; });
BinOp(SSub, [](auto left, auto right) { return left - right; });
BinOp(PointwiseMul, [](auto left, auto right) { return left * right; });
// BinOp(PointwiseMul, [](auto left, auto right) { return safe_mul(left, right); });
BinOp(PointwiseDiv, [](auto left, auto right) { return left / right; });
// BinOp(PointwiseDiv, [](auto left, auto right) { return safe_div(left, right); });

// No Sub and Div
BinOp(SAddTropMax, [](auto left, auto right) { return __builtin_elementwise_max(left, right); });
BinOp(SAddTropMin, [](auto left, auto right) { return __builtin_elementwise_min(left, right); });
BinOp(PointwiseMulTropMax, [](auto left, auto right) { return left + right; });
BinOp(PointwiseMulTropMin, [](auto left, auto right) { return left + right; });

template <typename vec_type, typename S1>
struct SNeg {
    S1 *s1;
    vec_type current_output;
    SNeg(S1 *s1) : s1(s1) {}

    inline __attribute__((always_inline)) void run(vec_type *out) {
        vec_type data;
        s1->run(&data);
        current_output = -data;
        *out = current_output;
    }

    inline __attribute__((always_inline)) void peek(vec_type *out) {
        *out = current_output;
    }

    void reset_and_next_row() { s1->reset_and_next_row(); }
};

template <typename vec_type, typename S>
struct Secondary {
    S *s;
    Secondary(S *s) : s(s) {}

    inline __attribute__((always_inline)) void run(vec_type *out) {
        s->peek(out);
    }

    inline __attribute__((always_inline)) void peek(vec_type *out) {
        s->peek(out);
    }

    void reset_and_next_row() {}
};

template <typename vec_type, typename S1>
auto make_SNeg(S1 *s1) {
    return new SNeg<vec_type, S1>(s1);
}

template <typename vec_type, typename S>
auto make_Secondary(S *s) {
    return new Secondary<vec_type, S>(s);
}

template <typename vec_type_in, typename vec_type_out, typename Inner>
auto make_k_convert_n2one(Inner *inner) {
    return new KConvertN2One<vec_type_in, vec_type_out, Inner>(inner);
}

template <typename vec_type_in, typename vec_type_out, typename Inner>
auto make_convert_n2one(Inner *inner) {
    return new ConvertN2One<vec_type_in, vec_type_out, Inner>(inner);
}

template <typename vec_type_in, typename vec_type_out, typename Inner>
auto make_k_convert_one2n(Inner *inner) {
    return new KConvertOne2N<vec_type_in, vec_type_out, Inner>(inner);
}

template <typename vec_type_in, typename vec_type_out, typename Inner>
auto make_convert_one2n(Inner *inner) {
    return new ConvertOne2N<vec_type_in, vec_type_out, Inner>(inner);
}

// IthRow: Extract the ith row from a 2D signal (Signal2D) as a 1D signal
template <typename vec_type, int n_rows, int row_index, typename Signal2DType>
struct IthRow {
    using elt_type = typename ElementType<vec_type>::type;

    Signal2DType *signal2d;
    vec_type current_output;

    IthRow(Signal2DType *sig) : signal2d(sig) {
        static_assert(row_index >= 0 && row_index < n_rows, "row_index out of bounds");
    }

    inline __attribute__((always_inline)) void run(vec_type *d) {
        // Read all n_rows from the 2D signal
        vec_type rows[n_rows];
        signal2d->run(rows);

        // Return only the row at row_index
        current_output = rows[row_index];
        *d = current_output;
    }

    inline __attribute__((always_inline)) void peek(vec_type *out) {
        *out = current_output;
    }

    void reset_and_next_row() {
        signal2d->reset_and_next_row();
    }
};

template <typename vec_type, int n_rows, typename Inner>
auto make_repeater(RepeaterContext<vec_type, n_rows> *ctx, Inner *inner) {
    return new Repeater<vec_type, n_rows, Inner>(ctx, inner);
}

template <typename vec_type, int n_rows>
auto make_signal2d(RepeaterContext<vec_type, n_rows + 1> *ctx) {
    return new Signal2D<vec_type, n_rows>(ctx);
}

template <typename vec_type, int n_rows, int row_index, typename Signal2DType>
auto make_ith_row(Signal2DType *sig) {
    return new IthRow<vec_type, n_rows, row_index, Signal2DType>(sig);
}

template <int _taps, typename vec_type, const int indices[_taps], typename... Args>
auto make_time_varying_kernel(Args *...args) {
    return new TimeVaryingKernel<_taps, vec_type, indices, Args...>(args...);
}

// ---------------------------------------------------------------------------
// Helpers for SRecurseV2
// ---------------------------------------------------------------------------

// SLatched: a signal whose value is set externally each step.
template <typename vec_type>
struct SLatched {
    vec_type value{0};
    void run(vec_type *out) { *out = value; }
    void peek(vec_type *out) { *out = value; }
    void reset_and_next_row() { value = 0; }
};

// MaybeSConvolve<vec_type, K>: SConvolve<K, SLatched> when K has taps > 0,
// else a no-op that always returns 0.  Used for cross terms in SRecurseV2.
template <typename vec_type, typename K, bool Active = (K::taps > 0)>
struct MaybeSConvolve {
    SConvolve<vec_type, K, SLatched<vec_type>> impl;
    explicit MaybeSConvolve(K *k) : impl(k, new SLatched<vec_type>()) {}
    void set_input(vec_type v) { impl.signal->value = v; }
    void run(vec_type *out) { impl.run(out); }
    void peek(vec_type *out) { impl.peek(out); }
    void reset_and_next_row() { impl.reset_and_next_row(); }
};
template <typename vec_type, typename K>
struct MaybeSConvolve<vec_type, K, false> {
    explicit MaybeSConvolve(K *) {}
    void set_input(vec_type) {}
    void run(vec_type *out) { *out = 0; }
    void peek(vec_type *out) { *out = 0; }
    void reset_and_next_row() {}
};

// MaybeSRecurse<vec_type, K>: SRecurse<K, SLatched> when K has taps > 0,
// else a pass-through (no feedback).  Used for diagonal terms in SRecurseV2.
template <typename vec_type, typename K, bool Active = (K::taps > 0)>
struct MaybeSRecurse {
    SRecurse<vec_type, K, SLatched<vec_type>> impl;
    explicit MaybeSRecurse(K *k) : impl(k, new SLatched<vec_type>()) {}
    void set_input(vec_type v) { impl.signal->value = v; }
    void run(vec_type *out) { impl.run(out); }
    void peek(vec_type *out) { impl.peek(out); }
    void reset_and_next_row() { impl.reset_and_next_row(); }
};
template <typename vec_type, typename K>
struct MaybeSRecurse<vec_type, K, false> {
    SLatched<vec_type> latched;
    explicit MaybeSRecurse(K *) {}
    void set_input(vec_type v) { latched->value = v; }
    void run(vec_type *out) { latched.run(out); }
    void peek(vec_type *out) { latched.peek(out); }
    void reset_and_next_row() { latched.reset_and_next_row(); }
};

// ---------------------------------------------------------------------------
// Mutually-recursive (vector) recurrence: SRecurseV2
//
// Computes F = A*F + G where F,G are 2-vectors and A is a 2x2 kernel matrix:
//   F[0] = A[0][0]*F[0] + A[0][1]*F[1] + G[0]
//   F[1] = A[1][0]*F[0] + A[1][1]*F[1] + G[1]
//
// Implemented using MaybeSRecurse for diagonal terms (K00, K11) and
// MaybeSConvolve for cross terms (K01, K10).
// ---------------------------------------------------------------------------
template <typename vec_type,
          typename K00, typename K01,
          typename K10, typename K11,
          typename S0, typename S1>
struct SRecurseV2 {
    MaybeSConvolve<vec_type, K01> c01; // K01 * history(F[1])
    MaybeSConvolve<vec_type, K10> c10; // K10 * history(F[0])
    MaybeSRecurse<vec_type, K00> r0;   // K00 * F[0] + eff_g0
    MaybeSRecurse<vec_type, K11> r1;   // K11 * F[1] + eff_g1
    S0 *s0;
    S1 *s1;
    vec_type last_out0{0}, last_out1{0};

    SRecurseV2(K00 *k00, K01 *k01, K10 *k10, K11 *k11, S0 *s0, S1 *s1)
        : c01(k01), c10(k10), r0(k00), r1(k11), s0(s0), s1(s1) {}

    inline __attribute__((always_inline)) void run(vec_type *out) {
        // Inject previous outputs into cross-term sliding windows
        c01.set_input(last_out1);
        c10.set_input(last_out0);

        // Compute cross-term contributions from previous step's outputs
        vec_type cross0, cross1;
        c01.run(&cross0);
        c10.run(&cross1);

        // Read actual inputs and form effective inputs (G + cross)
        vec_type g0, g1;
        s0->run(&g0);
        s1->run(&g1);
        r0.set_input(g0 + cross0);
        r1.set_input(g1 + cross1);

        // Compute diagonal recurrences
        r0.run(&out[0]);
        r1.run(&out[1]);

        // Remember outputs for next step's cross terms
        last_out0 = out[0];
        last_out1 = out[1];
    }

    inline __attribute__((always_inline)) void peek(vec_type *out) {
        out[0] = last_out0;
        out[1] = last_out1;
    }

    void reset_and_next_row() {
        last_out0 = 0;
        last_out1 = 0;
        c01.reset_and_next_row();
        c10.reset_and_next_row();
        r0.reset_and_next_row();
        r1.reset_and_next_row();
        s0->reset_and_next_row();
        s1->reset_and_next_row();
    }
};

template <typename vec_type,
          typename K00, typename K01, typename K10, typename K11,
          typename S0, typename S1>
auto make_s_recurse_v2(K00 *k00, K01 *k01, K10 *k10, K11 *k11, S0 *s0, S1 *s1) {
    return new SRecurseV2<vec_type, K00, K01, K10, K11, S0, S1>(k00, k01, k10, k11, s0, s1);
}

// ---------------------------------------------------------------------------
// VectorAtN: extract component `index` from an n-vector signal.
// ---------------------------------------------------------------------------

template <typename vec_type, int n, int index, typename SVType>
struct VectorAtN {
    SVType *signal_vec;
    vec_type all[n];

    VectorAtN(SVType *sv) : signal_vec(sv) {
        static_assert(index >= 0 && index < n, "VectorAtN: index out of range");
    }

    inline __attribute__((always_inline)) void run(vec_type *out) {
        signal_vec->run(all);
        *out = all[index];
    }

    inline __attribute__((always_inline)) void peek(vec_type *out) {
        *out = all[index];
    }

    void reset_and_next_row() { signal_vec->reset_and_next_row(); }
};

template <int n, int index, typename vec_type, typename SVType>
auto make_vector_at(SVType *sv) {
    return new VectorAtN<vec_type, n, index, SVType>(sv);
}
