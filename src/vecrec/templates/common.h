#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

typedef float float_vec4 __attribute__((ext_vector_type(4)));
typedef float float_vec8 __attribute__((ext_vector_type(8)));
typedef float float_vec16 __attribute__((ext_vector_type(16)));

constexpr int vec_lanes_of(float) { return 1; }
constexpr int vec_lanes_of(float_vec4) { return 4; }
constexpr int vec_lanes_of(float_vec8) { return 8; }
constexpr int vec_lanes_of(float_vec16) { return 16; }

template <typename T>
struct ElementType {};

template <>
struct ElementType<float> {
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

inline std::string string_of(float_vec4 x) {
    return std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) + ", " + std::to_string(x[3]);
}

inline std::string string_of(float_vec8 x) {
    return std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) + ", " + std::to_string(x[3]) + ", " +
           std::to_string(x[5]) + ", " + std::to_string(x[6]) + ", " + std::to_string(x[7]) + ", " + std::to_string(x[8]);
}

inline float extract_slice(float a, float b, int offset) {
    switch (offset) {
    case 0:
        return a;
    case 1:
        return b;
    };
    assert(false);
    return float{};
}

inline float_vec4 extract_slice(float_vec4 a, float_vec4 b, int offset) {
    switch (offset) {
    case 0:
        return a;
    case 1:
        return __builtin_shufflevector(a, b, 1, 2, 3, 4);
    case 2:
        return __builtin_shufflevector(a, b, 2, 3, 4, 5);
    case 3:
        return __builtin_shufflevector(a, b, 3, 4, 5, 6);
    case 4:
        return b;
    };
    assert(false);
    return float_vec4{};
}

inline float_vec8 extract_slice(float_vec8 a, float_vec8 b, int offset) {
    switch (offset) {
    case 0:
        return a;
    case 1:
        return __builtin_shufflevector(a, b, 1, 2, 3, 4, 5, 6, 7, 8);
    case 2:
        return __builtin_shufflevector(a, b, 2, 3, 4, 5, 6, 7, 8, 9);
    case 3:
        return __builtin_shufflevector(a, b, 3, 4, 5, 6, 7, 8, 9, 10);
    case 4:
        return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8, 9, 10, 11);
    case 5:
        return __builtin_shufflevector(a, b, 5, 6, 7, 8, 9, 10, 11, 12);
    case 6:
        return __builtin_shufflevector(a, b, 6, 7, 8, 9, 10, 11, 12, 13);
    case 7:
        return __builtin_shufflevector(a, b, 7, 8, 9, 10, 11, 12, 13, 14);
    case 8:
        return b;
    };
    assert(false);
    return float_vec8{};
}

inline float_vec16 extract_slice(float_vec16 a, float_vec16 b, int offset) {
    switch (offset) {
    case 0:
        return a;
    case 1:
        return __builtin_shufflevector(a, b, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    case 2:
        return __builtin_shufflevector(a, b, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
    case 3:
        return __builtin_shufflevector(a, b, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
    case 4:
        return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19);
    case 5:
        return __builtin_shufflevector(a, b, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    case 6:
        return __builtin_shufflevector(a, b, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);
    case 7:
        return __builtin_shufflevector(a, b, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22);
    case 8:
        return __builtin_shufflevector(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
    case 9:
        return __builtin_shufflevector(a, b, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24);
    case 10:
        return __builtin_shufflevector(a, b, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25);
    case 11:
        return __builtin_shufflevector(a, b, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26);
    case 12:
        return __builtin_shufflevector(a, b, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27);
    case 13:
        return __builtin_shufflevector(a, b, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28);
    case 14:
        return __builtin_shufflevector(a, b, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29);
    case 15:
        return __builtin_shufflevector(a, b, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30);
    case 16:
        return b;
    };
    assert(false);
    return float_vec16{};
}

template <int taps, typename vec_type, const int indices[taps], const typename ElementType<vec_type>::type vals[taps]>
struct TimeInvariantKernel {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int idxs[taps] = indices;

    TimeInvariantKernel() {}

    void run(vec_type *out) {
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = vals[i];
        }
    }
};

template <int taps, typename vec_type, int curr, typename... Args>
void write_coeff(vec_type *coeff, std::tuple<Args...> &args) {
    if constexpr (curr < taps) {
        vec_type result;
        std::get<curr>(args).run(&result);
        coeff[curr] = result;
        write_coeff<vec_type, taps, curr + 1, Args...>(coeff, args);
    }
}

template <int taps, typename vec_type, const int indices[taps], typename... Args>
struct TimeVaryingKernel {
    std::tuple<Args...> args;

    TimeVaryingKernel(Args... args) : args(args...) {}

    void run(vec_type *out) {
        write_coeff<vec_type, taps, 0, Args...>(out, args);
    }
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
};

template <typename vec_type>
struct Signal1D {
    using elt_type = typename ElementType<vec_type>::type;

    const vec_type *data;
    Signal1D(const elt_type *data) : data((const vec_type *)data) {}

    // The caller is responsible for ensuring the array access
    // does not go out of bound
    // const vec_type *run() {
    void run(vec_type *d) {
        *d = *data++;
    }
};

template <int taps, typename vec_type, typename K, typename S>
struct SConvolve {
    K kernel;
    S signal;

    constexpr static int vec_lanes = vec_lanes_of(vec_type{});
    constexpr static int buffer_size = taps / vec_lanes;

    vec_type prev_input[buffer_size];

    SConvolve(K kernel, S signal) : kernel(kernel), signal(signal) {}

    void run(vec_type *__restrict__ d) __restrict__ {
        vec_type curr_output;
        vec_type curr_input;

        signal.run(&curr_input);
#pragma unroll
        for (int i = 0; i + 1 < buffer_size; i++) {
            asm volatile("" :::);
            prev_input[i] = prev_input[i + 1];
        }
        prev_input[buffer_size - 1] = curr_input;

        vec_type k;
        kernel.run(&k);
        curr_output = 0;

        for (int i = 0; i < taps; i++) {
            // TODO: This is out of bound when i = 0
            int idx = (buffer_size - 1) * vec_lanes - i;
            vec_type va = prev_input[idx / vec_lanes];
            vec_type vb = prev_input[idx / vec_lanes + 1];
            curr_output += extract_slice(va, vb, idx % vec_lanes) * k[i];
        }
        *d = curr_output;
    }
};

template <int taps, typename vec_type, typename K, typename S>
struct SRecurse {
    K kernel;
    S signal;

    constexpr static int vec_lanes = vec_lanes_of(vec_type{});
    constexpr static int buffer_size = (taps - 1 + vec_lanes - 1) / vec_lanes + 1;

    vec_type prev_output[buffer_size] = {};

    SRecurse(K kernel, S signal) : kernel(kernel), signal(signal) {}

    void run(vec_type *__restrict__ out) __restrict__ {
        vec_type curr_input;
        vec_type curr_output;
        vec_type k[taps];

        kernel.run(k);
        for (int i = 0; i < vec_lanes; i++) {
            // Cannot depend on things being computed.
            assert(__builtin_reduce_max(k[i] == 0.));
        }

#pragma unroll
        for (int i = 0; i + 1 < buffer_size; i++) {
            // To avoid LLVM from turning it into a memmove intrinsic
            asm volatile("" :::);
            prev_output[i] = prev_output[i + 1];
        }

        signal.run(&curr_input);
        curr_output = curr_input;

        // ignore k[0]..k[vec_lanes-1]
#pragma unroll
        for (int i = vec_lanes; i < taps; i++) {
            int idx = (buffer_size - 1) * vec_lanes - i;
            if (idx % vec_lanes == 0) {
                curr_output += prev_output[idx / vec_lanes] * k[i];
            } else {
                auto va = prev_output[idx / vec_lanes];
                auto vb = prev_output[idx / vec_lanes + 1];
                curr_output += extract_slice(va, vb, idx % vec_lanes) * k[i];
            }
        }
        prev_output[buffer_size - 1] = curr_output;
        *out = curr_output;
    }
};

template <typename vec_type_in, typename vec_type_out, typename Inner>
struct ConvertN2One {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_in / lanes_out;
    static_assert(lanes_in % lanes_out == 0, "lanes_in must be multiple of lanes_out");

    Inner inner;

    ConvertN2One(Inner inner) : inner(inner) {}
    void run(vec_type_out __restrict__ *out) __restrict__ {

        vec_type_in *p = (vec_type_in *)out;
#pragma unroll
        for (int i = 0; i < factor; i++) {
            vec_type_in in;
            inner.run(&in);
            *p++ = in;
        }
    }
};

template <typename vec_type_in, typename vec_type_out, typename Inner>
struct ConvertOne2N {
    constexpr static int lanes_in = vec_lanes_of(vec_type_in{});
    constexpr static int lanes_out = vec_lanes_of(vec_type_out{});
    constexpr static int factor = lanes_out / lanes_in;
    static_assert(lanes_out % lanes_in == 0, "lanes_out must be multiple of lanes_in");

    Inner inner;
    vec_type_in buffer;
    int offset;

    ConvertOne2N(Inner inner) : inner(inner), offset(0) {}

    void run(vec_type_out __restrict__ *out) __restrict__ {
        if (offset == 0) {
            inner.run(&buffer);
        }

        *out = (vec_type_out *)buffer + offset;
        offset = (offset + 1) % factor;
    }
};

#define BinOp(name, operator)                               \
    template <typename vec_type, typename S1, typename S2>  \
    struct name {                                           \
        S1 s1;                                              \
        S2 s2;                                              \
        name(S1 s1, S2 s2) : s1(s1), s2(s2) {}              \
                                                            \
        void run(vec_type *__restrict__ out) __restrict__ { \
            vec_type left = s1.run();                       \
            vec_type right = s2.run();                      \
            *out = left operator right;                     \
        }                                                   \
    }

BinOp(SAdd, +);
BinOp(SSub, -);
BinOp(PointwiseMul, *);
BinOp(PointwiseDiv, /);

template <typename vec_type, typename S1>
struct SNeg {
    S1 s1;
    SNeg(S1 s1) : s1(s1) {}

    void run(vec_type *__restrict__ out) __restrict__ {
        vec_type data = s1.run();
        *out = *data;
    }
};

template <int taps, typename vec_type, typename K, typename S>
auto make_s_convolve(K kernel, S signal) {
    return SConvolve<taps, vec_type, K, S>(kernel, signal);
}

template <int taps, typename vec_type, typename K, typename S>
auto make_s_recurse(K kernel, S signal) {
    return SRecurse<taps, vec_type, K, S>(kernel, signal);
}

// constexpr int taps = 10;
// using VecType = float_vec8;

// // using KernelType = TimeVaryingKernel<float, 3, Signal1D<float, float>, Signal1D<float, float>, Signal1D<float, float>>;
// using KernelType = TimeInvariantKernel<VecType, taps>;

// int main() {
//     float *k1 = new float[2048 * 2048];
//     float *k2 = new float[2048 * 2048];
//     float *k3 = new float[2048 * 2048];
//     float *s = new float[2048 * 2048];
//     for (int i = 0; i < 2048 * 2048; i++) {
//         k1[i] = 0;
//         k2[i] = (1.0 * std::rand()) / RAND_MAX + 1;
//         k3[i] = (1.0 * std::rand()) / RAND_MAX;
//         // s[i] = (1.0 * std::rand()) / RAND_MAX;
//         s[i] = 0;
//     }
//     s[0] = 1;
//     s[1] = 1;
//     s[2] = 2;
//     s[3] = 3;
//     s[4] = 5;
//     s[5] = 8;
//     s[6] = 13;
//     s[7] = 21;
//     Signal1D<float> ks1(k1);
//     Signal1D<float> ks2(k2);
//     Signal1D<float> ks3(k3);
//     // KernelType kernel(ks1, ks2, ks3);

//     // float tik[] = {0,1, 1};
//     float tik[taps] = {0, 0, 0, 0, 0, 0, 0, 0, 34, 21};
//     KernelType kernel(tik);

//     Signal1D<VecType> signal(s);
//     SRecurse<VecType, taps, KernelType, Signal1D<VecType>> k(kernel, signal);

//     VecType result;
//     for (int i = 0; i < 4; i++) {
//         k.run(&result);
//         // for (int j = 0; j < vec_lanes_of(result); j++) {
//         //     std::cout << *((const float*)&result + j) << "\n";
//         // }
//         // std::cout << *k.run() << "\n";
//     }
//     k.run(&result);
//     for (int j = 0; j < vec_lanes_of(result); j++) {
//         std::cout << *((const float *)&result + j) << "\n";
//     }
// }
