// Compile with:
// clang++ so-iir.cpp -Wall -O3 -ffast-math -march=skylake-avx512

#include <vector>
#include <chrono>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#define VEC_LANES 16

constexpr int vec_lanes = VEC_LANES;
typedef float float_vec __attribute__((ext_vector_type(vec_lanes)));

#if VEC_LANES == 8
inline float_vec extract_slice(float_vec a, float_vec b, int offset) {
    switch (offset) {
    case 0: return a;
    case 1: return __builtin_shufflevector(a, b, 1, 2, 3, 4, 5, 6, 7, 8);
    case 2: return __builtin_shufflevector(a, b, 2, 3, 4, 5, 6, 7, 8, 9);
    case 3: return __builtin_shufflevector(a, b, 3, 4, 5, 6, 7, 8, 9, 10);
    case 4: return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8, 9, 10, 11);
    case 5: return __builtin_shufflevector(a, b, 5, 6, 7, 8, 9, 10, 11, 12);
    case 6: return __builtin_shufflevector(a, b, 6, 7, 8, 9, 10, 11, 12, 13);
    case 7: return __builtin_shufflevector(a, b, 7, 8, 9, 10, 11, 12, 13, 14);
    case 8: return b;
    };
    assert(false);
    return float_vec{};
}
#endif

#if VEC_LANES == 16
inline float_vec extract_slice(float_vec a, float_vec b, int offset) {
    switch (offset) {
    case 0: return a;
    case 1: return __builtin_shufflevector(a, b, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    case 2: return __builtin_shufflevector(a, b, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
    case 3: return __builtin_shufflevector(a, b, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
    case 4: return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19);
    case 5: return __builtin_shufflevector(a, b, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    case 6: return __builtin_shufflevector(a, b, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);
    case 7: return __builtin_shufflevector(a, b, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22);
    case 8: return __builtin_shufflevector(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
    case 9: return __builtin_shufflevector(a, b, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24);
    case 10: return __builtin_shufflevector(a, b, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25);
    case 11: return __builtin_shufflevector(a, b, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26);
    case 12: return __builtin_shufflevector(a, b, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27);
    case 13: return __builtin_shufflevector(a, b, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28);
    case 14: return __builtin_shufflevector(a, b, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29);
    case 15: return __builtin_shufflevector(a, b, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30);
    case 16: return b;
    };
    assert(false);
    return float_vec{};
}
#endif

#if VEC_LANES == 4
inline float_vec extract_slice(float_vec a, float_vec b, int offset) {
    switch (offset) {
    case 0: return a;
    case 1: return __builtin_shufflevector(a, b, 1, 2, 3, 4);
    case 2: return __builtin_shufflevector(a, b, 2, 3, 4, 5);
    case 3: return __builtin_shufflevector(a, b, 3, 4, 5, 6);
    case 4: return b;
    };
    assert(false);
    return float_vec{};
}
#endif

struct alignas(64) IIR2 {
    float_vec prev_output[2] = {};

    const float alpha = 0.f;
    const float beta = 0.f;

    void run(const float_vec *__restrict__ input, float_vec *__restrict__ output) {
        auto v = *input;
        v += alpha * prev_output[0];
        v += beta * prev_output[1];
        prev_output[1] = prev_output[0];
        *output = prev_output[0] = v;
    }

    void reset() {
        std::memset(prev_output, 0, sizeof(prev_output));
    }

    IIR2(float c0, float c1)
        : alpha(c0), beta(c1) {
        std::memset(prev_output, 0, sizeof(prev_output));
    }
};

template<int stride, int taps, bool first_tap_is_one>
struct FIR {
    float coeff[taps] = {};

    // How many vectors of input does it take to produce one vector of output (rounded up)
    constexpr static int buffer_size = ((taps - 1) * stride + vec_lanes - 1) / vec_lanes + 1;

    float_vec prev_input[buffer_size] = {};

    void run(const float_vec *__restrict__ input, float_vec *__restrict__ output) {
        float_vec acc{};

#pragma unroll
        for (int i = 0; i + 1 < buffer_size; i++) {
            prev_input[i] = prev_input[i+1];
        }
        prev_input[buffer_size - 1] = *input;

        // Use vector shuffles to extract previous inputs. Faster than unaligned
        // loads from the stack.
        size_t idx = (buffer_size - 1) * vec_lanes;
#pragma unroll
        for (int i = 0; i < taps; i++) {
            // Note all the ifs in here resolve statically, because the loop is unrolled
            if (i == 0 && first_tap_is_one) {
                acc = *input;
            } else {
                float_vec v;
                if (idx % vec_lanes == 0) {
                    // No shuffle required
                    v = prev_input[idx / vec_lanes];
                } else {
                    float_vec va = prev_input[idx / vec_lanes];
                    float_vec vb = prev_input[idx / vec_lanes + 1];
                    v = extract_slice(va, vb, idx % vec_lanes);
                }
                acc += v * coeff[i];
            }
            idx -= stride;
        }

        *output = acc;
    }

    void reset() {
        memset(prev_input, 0, sizeof(prev_input));
    }

    FIR(const std::vector<float> coeffs) {
        int i = 0;
        for (float c : coeffs) {
            coeff[i++] = c;
        }
        memset(prev_input, 0, sizeof(prev_input));
        assert(!first_tap_is_one || coeffs[0] == 1.f);
    }
};

template<typename A, typename B>
struct Cascade {
    A a;
    B b;

    void run(const float_vec *__restrict__ input, float_vec *__restrict__ output) {
        float_vec tmp;
        a.run(input, &tmp);
        b.run(&tmp, output);
    }

    void reset() {
        a.reset();
        b.reset();
    }

    Cascade(A a, B b)
        : a(a), b(b) {
    }
};

template<int stride = 1>
auto make_iir(double alpha, double beta) {
    if constexpr (stride == vec_lanes) {
        return IIR2(alpha, beta);
    } else {
        return Cascade(FIR<stride,
                       3, // taps
                       true // first tap is one
                       >({1.f, (float)alpha, (float)(-beta)}),
                       make_iir<stride * 2>(alpha * alpha + 2 * beta, -beta * beta));
    }
}

int main(int argc, char **argv) {

    float alpha = 1.8f, beta = -0.9f;

    auto iir = make_iir(alpha, beta);

    const size_t size = 1024 * 16;
    const int iters = 1024 * 16;

    float *input = (float *)aligned_alloc(64, size * sizeof(float));
    float *output = (float *)aligned_alloc(64, size * sizeof(float));
    float *reference_output = (float *)aligned_alloc(64, size * sizeof(float));

    std::memset(input, 0, size * sizeof(float));
    std::memset(output, 0, size * sizeof(float));
    std::memset(reference_output, 0, size * sizeof(float));
    for (int i = 0;  i < size; i++) {
        input[i] = i + 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iters; t++) {
        iir.reset();
        const float_vec *in_ptr = (const float_vec *)(input);
        float_vec *out_ptr = (float_vec *)(output);
        for (size_t i = 0; i < size; i += vec_lanes) {
            iir.run(in_ptr++, out_ptr++);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
#pragma unroll VEC_LANES
    for (int t = 0; t < iters; t++) {
        const float *in_ptr = input;
        float *out_ptr = reference_output;
        float prev0 = 0.f;
        float prev1 = 0.f;
        for (size_t i = 0; i < size; i++) {
            float next = alpha * prev0 + beta * prev1 + *in_ptr++;
            prev1 = prev0;
            prev0 = next;
            *out_ptr++ = next;
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        printf("%f %f\n", output[i], reference_output[i]);
    }
    printf("\n");

    free(input);
    free(output);

    double t_vec = std::chrono::duration<float>(t2 - t1).count();
    double t_scalar = std::chrono::duration<float>(t3 - t2).count();
    printf("Vectorized: %f seconds\n", t_vec);
    printf("Scalar: %f seconds\n", t_scalar);
    printf("Speed-up: %f\n", t_scalar / t_vec);
    printf("Vectorized throughput: %f giga-floats per second\n",
           size * iters / t_vec * 1e-9);

    return 0;
}
