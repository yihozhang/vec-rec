#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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

inline std::string string_of(int64_vec4 x) {
    return std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) + ", " + std::to_string(x[3]);
}

inline std::string string_of(int64_vec8 x) {
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

template <typename vec_type, int N>
struct Reg {
    vec_type data[N];
    Reg() {
        _Pragma("unroll") for (int i = 0; i < N; i++) {
            data[i] = vec_type{}; // zero-initialize
        }
    }
    inline __attribute__((always_inline)) vec_type &operator[](int i) { return data[i]; }
    inline __attribute__((always_inline)) const vec_type &operator[](int i) const { return data[i]; }
};

template <int _taps, typename vec_type, const int indices[_taps], const typename ElementType<vec_type>::type vals[_taps]>
struct TimeInvariantKernel {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static const int *idxs = indices;
    constexpr static int taps = _taps;

    TimeInvariantKernel() = default;

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        _Pragma("unroll") for (int i = 0; i < taps; i++) {
            asm volatile("" :::);
            reg[base + i] = vals[i];
        }
    }

    void reset_and_next_row() {}
};

template <int _taps, typename vec_type, const int indices[_taps]>
struct TimeVaryingKernel {
    constexpr static const int *idxs = indices;
    constexpr static int taps = _taps;

    TimeVaryingKernel() = default;

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, const int args[taps]) {
        _Pragma("unroll") for (int i = 0; i < taps; i++) {
            asm volatile("" :::);
            reg[base + i] = reg[args[i]];
        }
    }

    void reset_and_next_row() {}
};

template <typename vec_type>
struct Signal1DConstant {
    using elt_type = typename ElementType<vec_type>::type;

    vec_type data;
    Signal1DConstant() = delete;
    Signal1DConstant(elt_type data) : data(data) {}

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        reg[base] = data;
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
    bool fired;

    Signal1DImpulse() = delete;
    Signal1DImpulse(elt_type value, elt_type zero) : impulse_value(value), zero_value(zero), fired(false) {}

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type current_output;
        if (!fired) {
            fired = true;
            if constexpr (lanes == 1) {
                current_output = impulse_value;
            } else {
                vec_type result = zero_value;
                result[0] = impulse_value;
                current_output = result;
            }
        } else {
            current_output = zero_value; // broadcast zero to all lanes
        }
        reg[base] = current_output;
    }

    void reset_and_next_row() {}
};

template <typename vec_type>
struct Signal1D {
    using elt_type = typename ElementType<vec_type>::type;

    const vec_type *data;
    Signal1D() = delete;
    Signal1D(const elt_type *data) : data((const vec_type *)data) {}

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        memcpy((float *)(&reg[base]), (float *)data, sizeof(vec_type));
        data++;
    }

    // Signal1D streams through contiguous data. Between rows the pointer
    // is already at the start of the next row, so this is a no-op.
    void reset_and_next_row() {}
};

// Context struct to share state between Repeater and Signal2D
template <typename vec_type, int n_rows>
struct RepeaterContext {
    using elt_type = typename ElementType<vec_type>::type;
    int width;
    elt_type *buffer;
    int current_row;

    RepeaterContext() = delete;
    RepeaterContext(int row_size, elt_type zero_value) : width(row_size), current_row(0) {
        buffer = new (std::align_val_t(32)) elt_type[n_rows * width];
        for (int i = 0; i < n_rows * width; i++) {
            buffer[i] = zero_value;
        }
    }

    RepeaterContext(const RepeaterContext &) = delete;
    RepeaterContext &operator=(const RepeaterContext &) = delete;

    RepeaterContext(RepeaterContext &&other) noexcept
        : width(other.width), buffer(other.buffer), current_row(other.current_row) {
        other.buffer = nullptr;
    }

    RepeaterContext &operator=(RepeaterContext &&other) noexcept {
        if (this != &other) {
            delete[] buffer;
            width = other.width;
            current_row = other.current_row;
            buffer = other.buffer;
            other.buffer = nullptr;
        }
        return *this;
    }

    ~RepeaterContext() {
        delete[] buffer;
    }

    // Get the value at the given row offset and explicit column
    vec_type get(int row_offset, int col) {
        int row_idx = (current_row - row_offset + n_rows) % n_rows;
        vec_type result;
        memcpy(&result, &buffer[row_idx * width + col], sizeof(vec_type));
        return result;
    }

    // Store a value in the current row at an explicit column
    void set(vec_type value, int col) {
        memcpy(&buffer[current_row * width + col], &value, sizeof(vec_type));
    }

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &_reg) {
        // no-op: column tracking is now done per-call in Signal2D and Repeater
    }

    void reset_and_next_row() {
        current_row = (current_row + 1) % n_rows;
    }
};

template <typename vec_type, int n_rows>
struct Signal2D {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int lanes = vec_lanes_of(vec_type{});

    RepeaterContext<vec_type, n_rows + 1> *context;
    int my_col;

    Signal2D() = delete;
    Signal2D(RepeaterContext<vec_type, n_rows + 1> *ctx) : context(ctx), my_col(-lanes) {}

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        my_col += lanes;
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            asm volatile("" :::);
            reg[base + i] = context->get(i + 1, my_col);
        }
    }

    void reset_and_next_row() {
        my_col = -lanes;
    }
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

    InputSignal2D() = delete;
    InputSignal2D(const elt_type *data, int stride, elt_type zero)
        : data(data),
          stride(stride),
          current_row(0),
          zero_row(new elt_type[stride]) {
        _Pragma("unroll") for (int i = 0; i < stride; i++) {
            zero_row[i] = zero;
        }
        // d[0] = row 0 (current); d[1..n_rows-1] = zero (no previous rows yet)
        row_ptrs[0] = data - lanes;
        _Pragma("unroll") for (int i = 1; i < n_rows; i++) {
            row_ptrs[i] = zero_row - lanes;
        }
    }

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            row_ptrs[i] += lanes;
            memcpy(&reg[base + i], row_ptrs[i], sizeof(vec_type));
        }
    }

    void reset_and_next_row() {
        current_row++;
        // d[i] = row (current_row - i), zero-padded for rows before 0
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int src_row = current_row - i;
            row_ptrs[i] = ((src_row >= 0) ? data + (size_t)src_row * stride : zero_row) - lanes;
        }
    }
};

// template <typename vec_type, int n_rows>
// auto make_input_signal2d(int base, const typename ElementType<vec_type>::type *data, int stride, typename ElementType<vec_type>::type zero) {
//     return InputSignal2D<vec_type, n_rows>(base, data, stride, zero);
// }

template <typename vec_type, int n_rows>
struct Repeater {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int vec_lanes = vec_lanes_of(vec_type{});

    RepeaterContext<vec_type, n_rows> *context;
    int my_col;

    Repeater() = delete;
    Repeater(RepeaterContext<vec_type, n_rows> *ctx) : context(ctx), my_col(-vec_lanes) {}

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int inner) {
        my_col += vec_lanes;
        vec_type curr_output = reg[inner];
        context->set(curr_output, my_col);
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            reg[base + i] = context->get(i, my_col);
        }
    }

    void reset_and_next_row() {
        my_col = -vec_lanes;
    }
};

#define SConvolveOp(suffix, zero_init, mul_op, add_op)                                                 \
    template <typename vec_type, int taps, const int idxs[taps]>                                       \
    struct SConvolve##suffix {                                                                         \
        using elt_type = typename ElementType<vec_type>::type;                                         \
        constexpr static int vec_lanes = vec_lanes_of(vec_type{});                                     \
        constexpr static int buffer_size =                                                             \
            (idxs[taps - 1] + vec_lanes - 1) / vec_lanes + 1;                                          \
        vec_type prev_input[buffer_size];                                                              \
        vec_type curr_output;                                                                          \
        SConvolve##suffix() {                                                                          \
            _Pragma("unroll") for (int i = 0; i < buffer_size; i++) {                                  \
                prev_input[i] = (zero_init);                                                           \
            }                                                                                          \
        }                                                                                              \
        template <typename Reg_t>                                                                      \
        inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int kernel, int signal) { \
            curr_output = (zero_init);                                                                 \
            _Pragma("unroll") for (int i = 0; i + 1 < buffer_size; i++) {                              \
                asm volatile("" :::);                                                                  \
                prev_input[i] = prev_input[i + 1];                                                     \
            }                                                                                          \
            prev_input[buffer_size - 1] = reg[signal];                                                 \
            _Pragma("unroll") for (int i = 0; i < taps; i++) {                                         \
                int idx = (buffer_size - 1) * vec_lanes - idxs[i];                                     \
                vec_type contrib;                                                                      \
                if (idx % vec_lanes == 0) {                                                            \
                    contrib = mul_op(prev_input[idx / vec_lanes], reg[kernel + i]);                    \
                } else {                                                                               \
                    vec_type va = prev_input[idx / vec_lanes];                                         \
                    vec_type vb = prev_input[idx / vec_lanes + 1];                                     \
                    contrib = mul_op(                                                                  \
                        extract_slice(va, vb, idx % vec_lanes), reg[kernel + i]);                      \
                }                                                                                      \
                curr_output = add_op(curr_output, contrib);                                            \
            }                                                                                          \
            reg[base] = curr_output;                                                                   \
        }                                                                                              \
                                                                                                       \
        void reset_and_next_row() {                                                                    \
            _Pragma("unroll") for (int i = 0; i < buffer_size; i++) {                                  \
                prev_input[i] = (zero_init);                                                           \
            }                                                                                          \
        }                                                                                              \
    };                                                                                                 \
    // template <typename vec_type>                                                                           \
    // auto make_s_convolve##suffix(int base, int kernel, int signal) {                                       \
    //     return SConvolve##suffix<vec_type>(base, kernel, signal);                                          \
    // }

SConvolveOp(, 0, [](auto a, auto b) { return a * b; }, [](auto acc, auto x) { return acc + x; });
// SConvolveOp(, 0, [](auto a, auto b) { return safe_mul(a, b); }, [](auto acc, auto x) { return acc + x; });
SConvolveOp(TropMin, std::numeric_limits<elt_type>::max() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SConvolveOp(TropMax, std::numeric_limits<elt_type>::min() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

#define SRecurseOp(suffix, zero_init, mul_op, add_op)                                                  \
    template <typename vec_type, int taps, const int idxs[taps]>                                       \
    struct SRecurse##suffix {                                                                          \
        using elt_type = typename ElementType<vec_type>::type;                                         \
        constexpr static int vec_lanes = vec_lanes_of(vec_type{});                                     \
        constexpr static int buffer_size =                                                             \
            (idxs[taps - 1] + vec_lanes - 1) / vec_lanes + 1;                                          \
        vec_type prev_output[buffer_size];                                                             \
        SRecurse##suffix() {                                                                           \
            _Pragma("unroll") for (int i = 0; i < buffer_size; i++) {                                  \
                prev_output[i] = (zero_init);                                                          \
            }                                                                                          \
        }                                                                                              \
        vec_type curr_output;                                                                          \
        template <typename Reg_t>                                                                      \
        inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int kernel, int signal) { \
            curr_output = reg[signal];                                                                 \
            _Pragma("unroll") for (int i = 0; i < taps; i++) {                                         \
                int idx = (buffer_size - 1) * vec_lanes - idxs[i];                                     \
                vec_type contrib;                                                                      \
                if (idx % vec_lanes == 0) {                                                            \
                    contrib = mul_op(prev_output[idx / vec_lanes], reg[kernel + i]);                   \
                } else {                                                                               \
                    auto va = prev_output[idx / vec_lanes];                                            \
                    auto vb = prev_output[idx / vec_lanes + 1];                                        \
                    contrib = mul_op(                                                                  \
                        extract_slice(va, vb, idx % vec_lanes), reg[kernel + i]);                      \
                }                                                                                      \
                curr_output = add_op(curr_output, contrib);                                            \
            }                                                                                          \
            _Pragma("unroll") for (int i = 0; i + 1 < buffer_size; i++) {                              \
                asm volatile("" :::);                                                                  \
                prev_output[i] = prev_output[i + 1];                                                   \
            }                                                                                          \
            prev_output[buffer_size - 1] = curr_output;                                                \
            reg[base] = curr_output;                                                                   \
        }                                                                                              \
                                                                                                       \
        void reset_and_next_row() {                                                                    \
            _Pragma("unroll") for (int i = 0; i < buffer_size; i++) {                                  \
                prev_output[i] = (zero_init);                                                          \
            }                                                                                          \
        }                                                                                              \
    };

SRecurseOp(, 0, [](auto a, auto b) { return a * b; }, [](auto acc, auto x) { return acc + x; });
// SRecurseOp(, 0, [](auto a, auto b) { return safe_mul(a, b); }, [](auto acc, auto x) { return acc + x; });
SRecurseOp(TropMin, std::numeric_limits<elt_type>::max() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SRecurseOp(TropMax, std::numeric_limits<elt_type>::min() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

// SRecurseV: vectorised n-component IIR recurrence.
//
// Template parameters:
//   vec_type             – SIMD vector element type
//   n_vec                – number of signal components
//   n_pairs              – number of nonzero (dst_i, src_j) kernel pairs
//   pair_i_arr[n_pairs]  – destination component index for each pair
//   pair_j_arr[n_pairs]  – source component index for each pair
//   pair_taps_arr[n_pairs] – tap count for each pair
//   pair_offsets_arr[n_pairs] – offset of each pair's tap indices in flat_idxs_arr
//   n_flat               – total taps across all pairs
//   flat_idxs_arr[n_flat] – all tap indices concatenated in pair order
//   comp_buf_sizes_arr[n_vec]   – history buffer size (in vec_type slots) per component
//   comp_buf_offsets_arr[n_vec] – start offset of each component's buffer in prev_output
//   n_total_buf          – total size of prev_output  (= sum of comp_buf_sizes)
//
// run(reg, out_base, kernel_bases[n_pairs], in_base):
//   – reads n_vec inputs from reg[in_base .. in_base+n_vec-1]
//   – reads kernel values from reg[kernel_bases[p] + k] for pair p, tap k
//   – accumulates cross-component IIR contributions from own prev_output buffers
//   – writes n_vec outputs to reg[out_base .. out_base+n_vec-1]
//   – updates prev_output history buffers
//
// The buffer index formula (same as SRecurse) is:
//   buf_idx = (comp_buf_sizes[j] - 1) * vec_lanes - idx_k
// which is guaranteed >= 0 for any idx_k present in the stripped kernel for component j.
#define SRecurseVOp(suffix, zero_init, mul_op, add_op)                                                   \
template <typename vec_type,                                                                              \
          int n_vec,                                                                                       \
          int n_pairs,                                                                                     \
          const int pair_i_arr[n_pairs],                                                                   \
          const int pair_j_arr[n_pairs],                                                                   \
          const int pair_taps_arr[n_pairs],                                                                \
          const int pair_offsets_arr[n_pairs],                                                             \
          int n_flat,                                                                                      \
          const int flat_idxs_arr[n_flat],                                                                 \
          const int comp_buf_sizes_arr[n_vec],                                                             \
          const int comp_buf_offsets_arr[n_vec],                                                           \
          int n_total_buf>                                                                                 \
struct SRecurseV##suffix {                                                                                \
    using elt_type = typename ElementType<vec_type>::type;                                                \
    constexpr static int vec_lanes = vec_lanes_of(vec_type{});                                            \
    vec_type prev_output[n_total_buf];                                                                    \
                                                                                                          \
    SRecurseV##suffix() {                                                                                 \
        _Pragma("unroll") for (int i = 0; i < n_total_buf; i++) {                                        \
            prev_output[i] = (zero_init);                                                                 \
        }                                                                                                 \
    }                                                                                                     \
                                                                                                          \
    template <typename Reg_t>                                                                             \
    inline __attribute__((always_inline)) void run(                                                       \
            Reg_t &reg, int out_base, const int kernel_bases[n_pairs], int in_base) {                    \
        vec_type f[n_vec];                                                                                \
        _Pragma("unroll") for (int i = 0; i < n_vec; i++) f[i] = reg[in_base + i];                      \
                                                                                                          \
        for_<n_pairs>([&](auto p_num) {                                                                   \
            constexpr int p = p_num.value;                                                                \
            constexpr int ci = pair_i_arr[p];                                                             \
            constexpr int cj = pair_j_arr[p];                                                             \
            constexpr int taps_p = pair_taps_arr[p];                                                      \
            constexpr int buf_size_j = comp_buf_sizes_arr[cj];                                            \
            constexpr int buf_off_j = comp_buf_offsets_arr[cj];                                           \
            for_<taps_p>([&](auto k_num) {                                                                \
                constexpr int k = k_num.value;                                                            \
                constexpr int idx_k = flat_idxs_arr[pair_offsets_arr[p] + k];                            \
                constexpr int buf_idx = (buf_size_j - 1) * vec_lanes - idx_k;                            \
                vec_type hist;                                                                             \
                if constexpr (buf_idx % vec_lanes == 0) {                                                 \
                    hist = prev_output[buf_off_j + buf_idx / vec_lanes];                                  \
                } else {                                                                                   \
                    hist = extract_slice(prev_output[buf_off_j + buf_idx / vec_lanes],                    \
                                        prev_output[buf_off_j + buf_idx / vec_lanes + 1],                 \
                                        buf_idx % vec_lanes);                                             \
                }                                                                                         \
                f[ci] = add_op(f[ci], mul_op(hist, reg[kernel_bases[p] + k]));                           \
            });                                                                                            \
        });                                                                                                \
                                                                                                          \
        for_<n_vec>([&](auto j_num) {                                                                     \
            constexpr int j = j_num.value;                                                                \
            constexpr int buf_size = comp_buf_sizes_arr[j];                                               \
            constexpr int buf_off = comp_buf_offsets_arr[j];                                              \
            for_<buf_size - 1>([&](auto kk_num) {                                                         \
                constexpr int kk = kk_num.value;                                                          \
                prev_output[buf_off + kk] = prev_output[buf_off + kk + 1];                               \
            });                                                                                            \
            prev_output[buf_off + buf_size - 1] = f[j];                                                  \
            reg[out_base + j] = f[j];                                                                     \
        });                                                                                                \
    }                                                                                                     \
                                                                                                          \
    void reset_and_next_row() {                                                                           \
        _Pragma("unroll") for (int i = 0; i < n_total_buf; i++) {                                        \
            prev_output[i] = (zero_init);                                                                 \
        }                                                                                                 \
    }                                                                                                     \
};


        // for (int i = 0; i < n_vec; i++) { for (int j = 0; j < comp_buf_sizes_arr[i]; j++) { fprintf(stderr, "%f ", prev_output[comp_buf_offsets_arr[i] + j]); }}                                                                                                  \
        // printf("\n");                                                                                         \

SRecurseVOp(, 0, [](auto a, auto b) { return a * b; }, [](auto acc, auto x) { return acc + x; });
SRecurseVOp(TropMin, std::numeric_limits<elt_type>::max() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SRecurseVOp(TropMax, std::numeric_limits<elt_type>::min() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

// Tuple<vec_type, n>: copy n independently-allocated signals into n consecutive registers.
// After run(), component i is at reg[base + i].
// srcs[i] is the source register index for component i.
template <typename vec_type, int n>
struct Tuple {
    Tuple() = default;

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, const int srcs[n]) {
        _Pragma("unroll") for (int i = 0; i < n; i++) {
            reg[base + i] = reg[srcs[i]];
        }
    }

    void reset_and_next_row() {}
};

template <typename vec_type_in, typename vec_type_out, int taps>
struct KConvertN2One {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_out / lanes_in;
    static_assert(lanes_out % lanes_in == 0, "lanes_in must be multiple of lanes_out");

    KConvertN2One() = default;
    template <typename Reg_t_in, typename Reg_t_out>
    inline __attribute__((always_inline)) void run(Reg_t_in &reg_in, Reg_t_out &reg_out, int base, int inner_base) {
        // inner_base is the first register of the first factor block; blocks are contiguous
        // (each block has `taps` registers: inner_base + i*taps .. inner_base + i*taps + taps-1)
        vec_type_in out[taps][factor];
        _Pragma("unroll") for (int i = 0; i < factor; i++) {
            _Pragma("unroll") for (int j = 0; j < taps; j++) {
                out[j][i] = reg_in[inner_base + i * taps + j];
            }
        }
        memcpy(&reg_out[base], out, sizeof(vec_type_out) * taps);
    }

    void reset_and_next_row() {}
};

template <typename vec_type_in, typename vec_type_out>
struct ConvertN2One {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_out / lanes_in;
    static_assert(lanes_out % lanes_in == 0, "lanes_in must be multiple of lanes_out");

    ConvertN2One() = default;
    template <typename Reg_t_in, typename Reg_t_out>
    inline __attribute__((always_inline)) void run(Reg_t_in &reg_in, Reg_t_out &reg_out, int base, int inner_base) {
        // inner_base is the first of `factor` contiguous narrow-lane registers to merge
        vec_type_in out[factor];
        _Pragma("unroll") for (int i = 0; i < factor; i++) {
            asm volatile("" :::);
            out[i] = reg_in[inner_base + i];
        }
        memcpy(&reg_out[base], out, sizeof(vec_type_out));
    }

    void reset_and_next_row() {}
};

template <typename vec_type_in, typename vec_type_out, int taps, int offset>
struct KConvertOne2N {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_in / lanes_out;
    static_assert(lanes_in % lanes_out == 0, "lanes_out must be multiple of lanes_in");

    KConvertOne2N() = default;

    template <typename Reg_t_in, typename Reg_t_out>
    inline __attribute__((always_inline)) void run(Reg_t_in &reg_in, Reg_t_out &reg_out, int base, int inner) {
        _Pragma("unroll") for (int i = 0; i < taps; i++) {
            reg_out[base + i] = ExtractSubVector<vec_type_in, vec_type_out, offset>::extract_sub_vector(reg_in[inner + i]);
        }
    }

    void reset_and_next_row() {
    }
};

template <typename vec_type_in, typename vec_type_out, int offset>
struct ConvertOne2N {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_in / lanes_out;
    static_assert(lanes_in % lanes_out == 0, "lanes_out must be multiple of lanes_in");

    ConvertOne2N() = default;

    template <typename Reg_t_in, typename Reg_t_out>
    inline __attribute__((always_inline)) void run(Reg_t_in &reg_in, Reg_t_out &reg_out, int base, int inner) {
        reg_out[base] = ExtractSubVector<vec_type_in, vec_type_out, offset>::extract_sub_vector(reg_in[inner]);
    }

    void reset_and_next_row() {
    }
};

// ConvertN2OneV / ConvertOne2NV: like ConvertN2One / ConvertOne2N but for
// n_vec-component vectors whose components are interleaved in register space
// (stride n_vec).  inner_base points to the first of factor*n_vec consecutive
// registers laid out as: [slice0_comp0, slice0_comp1, ..., slice0_comp{n_vec-1},
//                         slice1_comp0, ..., slice{factor-1}_comp{n_vec-1}].
template <typename vec_type_in, typename vec_type_out, int n_vec>
struct ConvertN2OneV {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_out / lanes_in;
    static_assert(lanes_out % lanes_in == 0, "lanes_in must divide lanes_out");

    ConvertN2OneV() = default;
    template <typename Reg_t_in, typename Reg_t_out>
    inline __attribute__((always_inline)) void run(Reg_t_in &reg_in, Reg_t_out &reg_out, int base, int inner_base) {
        _Pragma("unroll") for (int j = 0; j < n_vec; j++) {
            vec_type_in parts[factor];
            _Pragma("unroll") for (int i = 0; i < factor; i++) {
                parts[i] = reg_in[inner_base + j + i * n_vec];
            }
            memcpy(&reg_out[base + j], parts, sizeof(vec_type_out));
        }
    }

    void reset_and_next_row() {}
};

template <typename vec_type_in, typename vec_type_out, int n_vec, int offset>
struct ConvertOne2NV {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_in / lanes_out;
    static_assert(lanes_in % lanes_out == 0, "lanes_out must divide lanes_in");

    ConvertOne2NV() = default;
    template <typename Reg_t_in, typename Reg_t_out>
    inline __attribute__((always_inline)) void run(Reg_t_in &reg_in, Reg_t_out &reg_out, int base, int inner_base) {
        _Pragma("unroll") for (int j = 0; j < n_vec; j++) {
            reg_out[base + j] = ExtractSubVector<vec_type_in, vec_type_out, offset>::extract_sub_vector(reg_in[inner_base + j]);
        }
    }

    void reset_and_next_row() {}
};

#define BinOp(name, operator)                                                                  \
    template <typename vec_type>                                                               \
    struct name {                                                                              \
        name() = default;                                                                      \
                                                                                               \
        template <typename Reg_t>                                                              \
        inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int s1, int s2) { \
            reg[base] = operator(reg[s1], reg[s2]);                                            \
        }                                                                                      \
                                                                                               \
        void reset_and_next_row() {                                                            \
        }                                                                                      \
    };

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

template <typename vec_type>
struct SNeg {
    SNeg() = default;

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int s1) {
        reg[base] = -reg[s1];
    }

    void reset_and_next_row() {}
};

// template <typename vec_type>
// auto make_SNeg(int s1) {
//     return SNeg<vec_type>(s1);
// }

// template <typename vec_type_in, typename vec_type_out>
// auto make_k_convert_n2one(int *inner) {
//     // TODO: add factor to type parameters (?)
//     return KConvertN2One<vec_type_in, vec_type_out>(inner);
// }

// template <typename vec_type_in, typename vec_type_out>
// auto make_convert_n2one(int *inner) {
//     return ConvertN2One<vec_type_in, vec_type_out>(inner);
// }

// template <typename vec_type_in, typename vec_type_out>
// auto make_k_convert_one2n(int inner) {
//     return KConvertOne2N<vec_type_in, vec_type_out>(inner);
// }

// template <typename vec_type_in, typename vec_type_out>
// auto make_convert_one2n(int inner) {
//     return ConvertOne2N<vec_type_in, vec_type_out>(inner);
// }

// IthRow: Extract the ith row from a 2D signal (Signal2D) as a 1D signal
template <typename vec_type, int n_rows, int row_index>
struct IthRow {
    using elt_type = typename ElementType<vec_type>::type;

    IthRow() {
        static_assert(row_index >= 0 && row_index < n_rows, "row_index out of bounds");
    }

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int signal2d) {
        reg[base] = reg[signal2d + row_index];
    }
    void reset_and_next_row() {
    }
};

// ---------------------------------------------------------------------------
// VectorAt: extract component `index` from an n-vector signal at sv_base.
// Reads reg[sv_base + index] and writes to reg[base].
// ---------------------------------------------------------------------------

template <typename vec_type, int n, int index>
struct VectorAt {
    VectorAt() {
        static_assert(index >= 0 && index < n, "VectorAt: index out of range");
    }

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int sv_base) {
        reg[base] = reg[sv_base + index];
    }

    void reset_and_next_row() {}
};
