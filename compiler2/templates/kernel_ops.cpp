#include "common.h"
#include <cstdlib>
#include <iostream>
#include <tuple>

template <int taps, typename vec_type, typename K1, typename K2>
struct KAdd {
    K1 a;
    K2 b;

    KAdd(K1 a, K2 b) : a(a), b(b) {}

    void run(vec_type *out) {
        vec_type a_coeff[taps], b_coeff[taps];
        a.run(&a_coeff);
        b.run(&b_coeff);
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = a_coeff[i] + b_coeff[i];
        }
    }
};

template <int taps, typename vec_type, typename K1, typename K2>
struct KSub {
    K1 a;
    K2 b;

    KSub(K1 a, K2 b) : a(a), b(b) {}

    void run(vec_type *out) {
        vec_type a_coeff, b_coeff;
        a.run(&a_coeff);
        b.run(&b_coeff);
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = a_coeff[i] - b_coeff[i];
        }
    }
};

template <int taps1, int taps2, typename vec_type, typename K1, typename K2>
struct KConvolve {
    K1 a;
    K2 b;

    KConvolve(K1 a, K2 b) : a(a), b(b) {}

    void run(vec_type *out) {
        vec_type a_coeff[taps1], b_coeff[taps2];
        a.run(a_coeff);
        b.run(b_coeff);
#pragma unroll
        for (int i = 0; i < taps1 + taps2 - 1; i++) {
            out[i] = 0;
            for (int j = 0; j < taps1; j++) {
                out[i] += a_coeff[j] * b_coeff[i - j];
            }
        }
    }
};

template <int taps, typename vec_type>
struct TimeInvariantKernel {
    using elt_type = typename ElementType<vec_type>::type;
    vec_type coeff[taps];

    TimeInvariantKernel(const elt_type *coeff) {
        for (int i = 0; i < taps; i++) {
            this->coeff[i] = coeff[i];
        }
    }

    void run(vec_type *out) {
#pragma unroll
        for (int i = 0; i < taps; i++) {
            out[i] = coeff[i];
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

template <int taps, typename vec_type, typename... Args>
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
    Signal1DConstant(elt_type) {
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
    S signal;
    K kernel;

    constexpr static int vec_lanes = vec_lanes_of(vec_type{});
    constexpr static int buffer_size = taps / vec_lanes;

    vec_type prev_input[buffer_size];

    SConvolve(S signal, K kernel) : signal(signal), kernel(kernel) {}

    void run(vec_type *__restrict__ d) __restrict__ {
        vec_type curr_output;
        vec_type curr_input;

        signal.run(&curr_input);
#pragma unroll
        for (int i = 0; i + 1 < buffer_size; i++) {
            asm volatile("" :::);
            prev_input[i] = prev_input[i + 1];
        }
        prev_input[buffer_size - 1] = *curr_input;

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

#define BinOp(name, operator)                               \
    template <typename vec_type, typename S1, typename S2>  \
    struct name {                                           \
        S1 s1;                                              \
        S2 s2;                                              \
        name(s1, s2) : s1(s1), s2(s2) {}                    \
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
    SAdd(s1) : s1(s1) {}

    void run(vec_type *__restrict__ out) __restrict__ {
        vec_type data = s1.run();
        *out = *data;
    }
};

constexpr int taps = 10;
using VecType = float_vec8;

// using KernelType = TimeVaryingKernel<float, 3, Signal1D<float, float>, Signal1D<float, float>, Signal1D<float, float>>;
using KernelType = TimeInvariantKernel<VecType, taps>;

int main() {
    float *k1 = new float[2048 * 2048];
    float *k2 = new float[2048 * 2048];
    float *k3 = new float[2048 * 2048];
    float *s = new float[2048 * 2048];
    for (int i = 0; i < 2048 * 2048; i++) {
        k1[i] = 0;
        k2[i] = (1.0 * std::rand()) / RAND_MAX + 1;
        k3[i] = (1.0 * std::rand()) / RAND_MAX;
        // s[i] = (1.0 * std::rand()) / RAND_MAX;
        s[i] = 0;
    }
    s[0] = 1;
    s[1] = 1;
    s[2] = 2;
    s[3] = 3;
    s[4] = 5;
    s[5] = 8;
    s[6] = 13;
    s[7] = 21;
    Signal1D<float> ks1(k1);
    Signal1D<float> ks2(k2);
    Signal1D<float> ks3(k3);
    // KernelType kernel(ks1, ks2, ks3);

    // float tik[] = {0,1, 1};
    float tik[taps] = {0, 0, 0, 0, 0, 0, 0, 0, 34, 21};
    KernelType kernel(tik);

    Signal1D<VecType> signal(s);
    SRecurse<VecType, taps, KernelType, Signal1D<VecType>> k(kernel, signal);

    VecType result;
    for (int i = 0; i < 4; i++) {
        k.run(&result);
        // for (int j = 0; j < vec_lanes_of(result); j++) {
        //     std::cout << *((const float*)&result + j) << "\n";
        // }
        // std::cout << *k.run() << "\n";
    }
    k.run(&result);
    for (int j = 0; j < vec_lanes_of(result); j++) {
        std::cout << *((const float *)&result + j) << "\n";
    }
}
