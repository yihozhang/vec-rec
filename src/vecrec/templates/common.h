#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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


#define EXTRACT_SLICE_1(type) \
    inline type extract_slice(type a, type b, int offset) { \
        switch (offset) { \
        case 0: \
            return a; \
        case 1: \
            return b; \
        }; \
        assert(false); \
        return type{}; \
    }

#define EXTRACT_SLICE_2(type_vec2) \
    inline type_vec2 extract_slice(type_vec2 a, type_vec2 b, int offset) { \
        switch (offset) { \
        case 0: \
            return a; \
        case 1: \
            return __builtin_shufflevector(a, b, 1, 2); \
        case 2: \
            return b; \
        }; \
        assert(false); \
        return type_vec2{}; \
    }

#define EXTRACT_SLICE_4(type_vec4) \
    inline type_vec4 extract_slice(type_vec4 a, type_vec4 b, int offset) { \
        switch (offset) { \
        case 0: \
            return a; \
        case 1: \
            return __builtin_shufflevector(a, b, 1, 2, 3, 4); \
        case 2: \
            return __builtin_shufflevector(a, b, 2, 3, 4, 5); \
        case 3: \
            return __builtin_shufflevector(a, b, 3, 4, 5, 6); \
        case 4: \
            return b; \
        }; \
        assert(false); \
        return type_vec4{}; \
    }

#define EXTRACT_SLICE_8(type_vec8) \
    inline type_vec8 extract_slice(type_vec8 a, type_vec8 b, int offset) { \
        switch (offset) { \
        case 0: \
            return a; \
        case 1: \
            return __builtin_shufflevector(a, b, 1, 2, 3, 4, 5, 6, 7, 8); \
        case 2: \
            return __builtin_shufflevector(a, b, 2, 3, 4, 5, 6, 7, 8, 9); \
        case 3: \
            return __builtin_shufflevector(a, b, 3, 4, 5, 6, 7, 8, 9, 10); \
        case 4: \
            return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8, 9, 10, 11); \
        case 5: \
            return __builtin_shufflevector(a, b, 5, 6, 7, 8, 9, 10, 11, 12); \
        case 6: \
            return __builtin_shufflevector(a, b, 6, 7, 8, 9, 10, 11, 12, 13); \
        case 7: \
            return __builtin_shufflevector(a, b, 7, 8, 9, 10, 11, 12, 13, 14); \
        case 8: \
            return b; \
        }; \
        assert(false); \
        return type_vec8{}; \
    }

#define EXTRACT_SLICE_16(type_vec16) \
    inline type_vec16 extract_slice(type_vec16 a, type_vec16 b, int offset) { \
        switch (offset) { \
        case 0: \
            return a; \
        case 1: \
            return __builtin_shufflevector(a, b, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16); \
        case 2: \
            return __builtin_shufflevector(a, b, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17); \
        case 3: \
            return __builtin_shufflevector(a, b, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18); \
        case 4: \
            return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19); \
        case 5: \
            return __builtin_shufflevector(a, b, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20); \
        case 6: \
            return __builtin_shufflevector(a, b, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21); \
        case 7: \
            return __builtin_shufflevector(a, b, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22); \
        case 8: \
            return __builtin_shufflevector(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23); \
        case 9: \
            return __builtin_shufflevector(a, b, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24); \
        case 10: \
            return __builtin_shufflevector(a, b, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25); \
        case 11: \
            return __builtin_shufflevector(a, b, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26); \
        case 12: \
            return __builtin_shufflevector(a, b, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27); \
        case 13: \
            return __builtin_shufflevector(a, b, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28); \
        case 14: \
            return __builtin_shufflevector(a, b, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29); \
        case 15: \
            return __builtin_shufflevector(a, b, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30); \
        case 16: \
            return b; \
        }; \
        assert(false); \
        return type_vec16{}; \
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
template<std::size_t N>
struct num { static const constexpr auto value = N; };

template <class F, std::size_t... Is>
void for_(F func, std::index_sequence<Is...>)
{
  (func(num<Is>{}), ...);
}

template <std::size_t N, typename F>
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}

template <typename vec_type_in, typename vec_type_out, int n>
struct ExtractSubVector {};

#define EXTRACT_SUB_VECTOR1(type_vec1) \
    template <typename vec_type_in, int n> \
    struct ExtractSubVector<vec_type_in, type_vec1, n> { \
        static type_vec1 extract_sub_vector(vec_type_in a) { \
            return a[n]; \
        } \
    };

#define EXTRACT_SUB_VECTOR2(type_vec2) \
    template <typename vec_type_in, int n> \
    struct ExtractSubVector<vec_type_in, type_vec2, n> { \
        static type_vec2 extract_sub_vector(vec_type_in a) { \
            return __builtin_shufflevector(a, a, 2 * n, 2 * n + 1); \
        } \
    };

#define EXTRACT_SUB_VECTOR4(type_vec4) \
    template <typename vec_type_in, int n> \
    struct ExtractSubVector<vec_type_in, type_vec4, n> { \
        static type_vec4 extract_sub_vector(vec_type_in a) { \
            return __builtin_shufflevector(a, a, 4 * n, 4 * n + 1, 4 * n + 2, 4 * n + 3); \
        } \
    };

#define EXTRACT_SUB_VECTOR8(type_vec8) \
    template <typename vec_type_in, int n> \
    struct ExtractSubVector<vec_type_in, type_vec8, n> { \
        static type_vec8 extract_sub_vector(vec_type_in a) { \
            return __builtin_shufflevector(a, a, 8 * n, 8 * n + 1, 8 * n + 2, 8 * n + 3, 8 * n + 4, 8 * n + 5, 8 * n + 6, 8 * n + 7); \
        } \
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

template <int _taps, typename vec_type, const int indices[_taps], const typename ElementType<vec_type>::type vals[_taps]>
struct TimeInvariantKernel {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static const int *idxs = indices;
    constexpr static int taps = _taps;

    TimeInvariantKernel() {}

    void run(vec_type *out) {
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = vals[i];
        }
    }

    void reset_and_next_row() {}
};

template <int taps, typename vec_type, int curr, typename... Args>
void write_coeff(vec_type *coeff, std::tuple<Args...> &args) {
    if constexpr (curr < taps) {
        vec_type result;
        std::get<curr>(args).run(&result);
        coeff[curr] = result;
        write_coeff<taps, vec_type, curr + 1, Args...>(coeff, args);
    }
}

template <int _taps, typename vec_type, const int indices[_taps], typename... Args>
struct TimeVaryingKernel {
    constexpr static const int *idxs = indices;
    constexpr static int taps = _taps;
    std::tuple<Args...> args;

    TimeVaryingKernel(Args... args) : args(args...) {}

    void run(vec_type *out) {
        write_coeff<_taps, vec_type, 0, Args...>(out, args);
    }

    void reset_and_next_row() {
        reset_tv_kernel<0>();
    }

private:
    template <int I>
    void reset_tv_kernel() {
        if constexpr (I < _taps) {
            std::get<I>(args).reset_and_next_row();
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
    void run(vec_type *d) {
        *d = data;
    }

    void reset_and_next_row() {}
};

template <typename vec_type>
struct Signal1D {
    using elt_type = typename ElementType<vec_type>::type;
    // constexpr static int taps = 1;
    // constexpr static const int idxs[1] = {0};

    const vec_type *data;
    Signal1D(const elt_type *data) : data((const vec_type *)data) {}

    // The caller is responsible for ensuring the array access
    // does not go out of bound
    // const vec_type *run() {
    void run(vec_type *d) {
        memcpy((float*)d, (float*)data, sizeof(vec_type));
        data++;
    }

    // Signal1D streams through contiguous data. Between rows the pointer
    // is already at the start of the next row, so this is a no-op.
    void reset_and_next_row() {}
};

// Context struct to share state between Repeater and Signal2D
template <typename vec_type, int n_rows>
struct RepeaterContext {
    constexpr static int width = 1024;
    
    vec_type buffer[n_rows][width];
    int current_row;
    int current_col;
    
    RepeaterContext() : current_row(0), current_col(0) {
        memset(buffer, 0, sizeof(buffer));
    }
    
    // Get the value at the given row offset and current column
    vec_type get(int row_offset) {
        int row_idx = (current_row - row_offset + n_rows) % n_rows;
        return buffer[row_idx][current_col];
    }
    
    // Store a value in the current row and column
    void set(vec_type value) {
        buffer[current_row][current_col] = value;
    }
    
    void advance_col() {
        current_col++;
        if (current_col >= width) {
            current_col = 0;
        }
    }
    
    void next_row() {
        current_row = (current_row + 1) % n_rows;
        current_col = 0;
        memset(buffer[current_row], 0, sizeof(buffer[current_row]));
    }
};

template <typename vec_type, int n_rows>
struct Signal2D {
    using elt_type = typename ElementType<vec_type>::type;
    
    RepeaterContext<vec_type, n_rows + 1> *context;
    
    Signal2D(RepeaterContext<vec_type, n_rows + 1> *ctx) : context(ctx) {
    }

    void run(vec_type *d) {
        // Read from previous rows at the current position
        for (int i = 0; i < n_rows; i++) {
            d[i] = context->get(i + 1);
        }
    }

    void reset_and_next_row() {}
};

template <typename vec_type, int n_rows, typename Inner>
struct Repeater {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int vec_lanes = vec_lanes_of(vec_type{});
    constexpr static int width = RepeaterContext<vec_type, n_rows>::width;
    
    RepeaterContext<vec_type, n_rows> *context;
    Inner inner;
    
    Repeater(RepeaterContext<vec_type, n_rows> *ctx, Inner inner) 
        : context(ctx), inner(inner) {
    }
    
    void run(vec_type *out) {
        // Compute the current row output using the inner signal
        // (which may read from the buffered previous rows via Signal2D)
        vec_type curr_output;
        inner.run(&curr_output);
        
        // Store the current output in the buffer
        context->set(curr_output);
        
        // Return n_rows vectors: current row at out[0], previous rows at out[1..n-1]
        // This makes Repeater stream as a 2D signal
        for (int i = 0; i < n_rows; i++) {
            out[i] = context->get(i);
        }
        
        // Advance column position
        context->advance_col();
    }
    
    void reset_and_next_row() {
        // Move to the next row
        context->next_row();
        
        // Reset the inner signal
        inner.reset_and_next_row();
    }
};

template <typename vec_type, typename K, typename S>
struct SConvolve {
    K kernel;
    S signal;

    constexpr static int vec_lanes = vec_lanes_of(vec_type{});
    constexpr static int buffer_size = (K::idxs[K::taps - 1] + vec_lanes - 1) / vec_lanes + 1;

    vec_type prev_input[buffer_size] = {};

    SConvolve(K kernel, S signal) : kernel(kernel), signal(signal) {}

    void run(vec_type *d) {
        vec_type curr_output;
        vec_type curr_input;

        signal.run(&curr_input);
#pragma unroll
        for (int i = 0; i + 1 < buffer_size; i++) {
            asm volatile("" :::);
            prev_input[i] = prev_input[i + 1];
        }
        prev_input[buffer_size - 1] = curr_input;

        vec_type k[K::taps];
        kernel.run(k);
        curr_output = 0;

        for (int i = 0; i < K::taps; i++) {
            int idx = (buffer_size - 1) * vec_lanes - K::idxs[i];
            if (idx % vec_lanes == 0) {
                curr_output += prev_input[idx / vec_lanes] * k[i];
            } else {
                vec_type va = prev_input[idx / vec_lanes];
                vec_type vb = prev_input[idx / vec_lanes + 1];
                curr_output += extract_slice(va, vb, idx % vec_lanes) * k[i];
            }
        }
        *d = curr_output;
    }

    void reset_and_next_row() {
        memset(prev_input, 0, sizeof(prev_input));
        kernel.reset_and_next_row();
        signal.reset_and_next_row();
    }
};

template <typename vec_type, typename K, typename S>
struct SRecurse {
    K kernel;
    S signal;

    constexpr static int vec_lanes = vec_lanes_of(vec_type{});
    constexpr static int buffer_size = (K::idxs[K::taps - 1] + vec_lanes - 1) / vec_lanes + 1;

    vec_type prev_output[buffer_size] = {};

    SRecurse(K kernel, S signal) : kernel(kernel), signal(signal) {}

    void run(vec_type *out) {
        vec_type curr_input;
        vec_type curr_output;
        vec_type k[K::taps];

        kernel.run(k);
        signal.run(&curr_input);
        curr_output = curr_input;

#pragma unroll
        for (int i = 0; i < K::taps; i++) {
            int idx = (buffer_size - 1) * vec_lanes - K::idxs[i];
            if (idx % vec_lanes == 0) {
                curr_output += prev_output[idx / vec_lanes] * k[i];
            } else {
                auto va = prev_output[idx / vec_lanes];
                auto vb = prev_output[idx / vec_lanes + 1];
                curr_output += extract_slice(va, vb, idx % vec_lanes) * k[i];
            }
        }

#pragma unroll
        for (int i = 0; i + 1 < buffer_size; i++) {
            // To avoid LLVM from turning it into a memmove intrinsic
            asm volatile("" :::);
            prev_output[i] = prev_output[i + 1];
        }
        prev_output[buffer_size - 1] = curr_output;
        *out = curr_output;
    }

    void reset_and_next_row() {
        memset(prev_output, 0, sizeof(prev_output));
        kernel.reset_and_next_row();
        signal.reset_and_next_row();
    }
};


template <typename vec_type_in, typename vec_type_out, typename Inner>
struct KConvertN2One {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_out / lanes_in;
    static_assert(lanes_out % lanes_in == 0, "lanes_in must be multiple of lanes_out");
    constexpr static int taps = Inner::taps;
    constexpr static const int *idxs = Inner::idxs;

    Inner inner;

    KConvertN2One(Inner inner) : inner(inner) {}
    void run(vec_type_out *out) {

        vec_type_in *p = (vec_type_in *)out;
#pragma unroll
        for (int i = 0; i < factor; i++) {
            vec_type_in in[taps];
            inner.run(&in);
#pragma unroll
            for (int j = 0; j < taps; j++) {
                p[j * factor + i] = in[j];
            }
            *p++ = in;
        }
    }

    void reset_and_next_row() { inner.reset_and_next_row(); }
};

template <typename vec_type_in, typename vec_type_out, typename Inner>
struct ConvertN2One {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_out / lanes_in;
    static_assert(lanes_out % lanes_in == 0, "lanes_in must be multiple of lanes_out");

    Inner inner;

    ConvertN2One(Inner inner) : inner(inner) {}
    void run(vec_type_out *out) {

        vec_type_in *p = (vec_type_in *)out;
#pragma unroll
        for (int i = 0; i < factor; i++) {
            vec_type_in in;
            inner.run(&in);
            *p++ = in;
        }
    }

    void reset_and_next_row() { inner.reset_and_next_row(); }
};

template <typename vec_type_in, typename vec_type_out, typename Inner>
struct KConvertOne2N {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_in / lanes_out;
    static_assert(lanes_in % lanes_out == 0, "lanes_out must be multiple of lanes_in");
    constexpr static int taps = Inner::taps;
    constexpr static const int *idxs = Inner::idxs;

    Inner inner;
    vec_type_in buffer[taps];
    vec_type_out out_buffer[taps][factor];
    int offset;

    KConvertOne2N(Inner inner) : inner(inner), offset(0) {}

    void run(vec_type_out *out) {
        if (offset == 0) {
            inner.run(buffer);
#pragma unroll
            for (int i = 0; i < taps; i++) {
                for_<factor>([&] (auto j) {
                    out_buffer[i][j.value] = ExtractSubVector<vec_type_in, vec_type_out, j.value>::extract_sub_vector(buffer[i]);
                });
            }
        }
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = out_buffer[i][offset];
        }
        offset = (offset + 1) % factor;
    }

    void reset_and_next_row() {
        offset = 0;
        inner.reset_and_next_row();
    }
};
#include<iostream>
template <typename vec_type_in, typename vec_type_out, typename Inner>
struct ConvertOne2N {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_in / lanes_out;
    static_assert(lanes_in % lanes_out == 0, "lanes_out must be multiple of lanes_in");

    Inner inner;
    vec_type_in buffer;
    vec_type_out out_buffer[factor];
    int offset;

    ConvertOne2N(Inner inner) : inner(inner), offset(0) {}

    void run(vec_type_out *out) {
        if (offset == 0) {
            inner.run(&buffer);

            for_<factor>([&] (auto i) {
                out_buffer[i.value] = ExtractSubVector<vec_type_in, vec_type_out, i.value>::extract_sub_vector(buffer);
            });
        }
        *out = out_buffer[offset];
        offset = (offset + 1) % factor;
    }

    void reset_and_next_row() {
        offset = 0;
        inner.reset_and_next_row();
    }
};

#define BinOp(name, operator)                               \
    template <typename vec_type, typename S1, typename S2>  \
    struct name {                                           \
        S1 s1;                                              \
        S2 s2;                                              \
        name(S1 s1, S2 s2) : s1(s1), s2(s2) {}              \
                                                            \
        void run(vec_type *out) {                           \
            vec_type left, right;                           \
            s1.run(&left);                                  \
            s2.run(&right);                                 \
            *out = operator(left, right);                   \
        }                                                   \
                                                            \
        void reset_and_next_row() {                         \
            s1.reset_and_next_row();                        \
            s2.reset_and_next_row();                        \
        }                                                   \
    };                                                      \
    template <typename vec_type, typename S1, typename S2>  \
    auto make_##name(S1 s1, S2 s2) { return name<vec_type, S1, S2>(s1, s2); }

BinOp(SAdd, [] (auto left, auto right) { return left + right; });
BinOp(SSub, [] (auto left, auto right) { return left - right; });
BinOp(PointwiseMul, [] (auto left, auto right) { return left * right; });
BinOp(PointwiseDiv, [] (auto left, auto right) { return left / right; });

// No Sub and Div
BinOp(SAddTropMax, [] (auto left, auto right) { return __builtin_elementwise_max(left, right); });
BinOp(SAddTropMin, [] (auto left, auto right) { return __builtin_elementwise_min(left, right); });
BinOp(PointwiseMulTropMax, [] (auto left, auto right) { return left + right; });
BinOp(PointwiseMulTropMin, [] (auto left, auto right) { return left + right; });

template <typename vec_type, typename S1>
struct SNeg {
    S1 s1;
    SNeg(S1 s1) : s1(s1) {}

    void run(vec_type *out) {
        vec_type data;
        s1.run(&data);
        *out = -data;
    }

    void reset_and_next_row() { s1.reset_and_next_row(); }
};

template<typename vec_type_in, typename vec_type_out, typename Inner>
auto make_k_convert_n2one(Inner inner) {
    return KConvertN2One<vec_type_in, vec_type_out, Inner>(inner);
}

template<typename vec_type_in, typename vec_type_out, typename Inner>
auto make_convert_n2one(Inner inner) {
    return ConvertN2One<vec_type_in, vec_type_out, Inner>(inner);
}

template<typename vec_type_in, typename vec_type_out, typename Inner>
auto make_k_convert_one2n(Inner inner) {
    return KConvertOne2N<vec_type_in, vec_type_out, Inner>(inner);
}

template<typename vec_type_in, typename vec_type_out, typename Inner>
auto make_convert_one2n(Inner inner) {
    return ConvertOne2N<vec_type_in, vec_type_out, Inner>(inner);
}

template <typename vec_type, typename K, typename S>
auto make_s_convolve(K kernel, S signal) {
    return SConvolve<vec_type, K, S>(kernel, signal);
}

// IthRow: Extract the ith row from a 2D signal (Signal2D) as a 1D signal
template <typename vec_type, int n_rows, int row_index, typename Signal2DType>
struct IthRow {
    using elt_type = typename ElementType<vec_type>::type;
    
    Signal2DType signal2d;
    
    IthRow(Signal2DType sig) : signal2d(sig) {
        static_assert(row_index >= 0 && row_index < n_rows, "row_index out of bounds");
    }
    
    void run(vec_type *d) {
        // Read all n_rows from the 2D signal
        vec_type rows[n_rows];
        signal2d.run(rows);
        
        // Return only the row at row_index
        *d = rows[row_index];
    }
    
    void reset_and_next_row() {
        signal2d.reset_and_next_row();
    }
};

template <typename vec_type, typename K, typename S>
auto make_s_recurse(K kernel, S signal) {
    return SRecurse<vec_type, K, S>(kernel, signal);
}

template <typename vec_type, int n_rows, typename Inner>
auto make_repeater(RepeaterContext<vec_type, n_rows> *ctx, Inner inner) {
    return Repeater<vec_type, n_rows, Inner>(ctx, inner);
}

template <typename vec_type, int n_rows>
auto make_signal2d(RepeaterContext<vec_type, n_rows + 1> *ctx) {
    return Signal2D<vec_type, n_rows>(ctx);
}

template <typename vec_type, int n_rows, int row_index, typename Signal2DType>
auto make_ith_row(Signal2DType sig) {
    return IthRow<vec_type, n_rows, row_index, Signal2DType>(sig);
}

template <int _taps, typename vec_type, const int indices[_taps], typename... Args>
auto make_time_varying_kernel(Args... args) {
    return TimeVaryingKernel<_taps, vec_type, indices, Args...>(args...);
}
