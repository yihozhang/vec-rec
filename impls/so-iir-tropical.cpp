#include <vector>
#include <chrono>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define VEC_LANES 4

constexpr int vec_lanes = VEC_LANES;
typedef int int_vec __attribute__((ext_vector_type(vec_lanes)));

#if VEC_LANES == 4
inline int_vec extract_slice(int_vec a, int_vec b, int offset) {
    switch (offset) {
    case 0: return a;
    case 1: return __builtin_shufflevector(a, b, 1, 2, 3, 4);
    case 2: return __builtin_shufflevector(a, b, 2, 3, 4, 5);
    case 3: return __builtin_shufflevector(a, b, 3, 4, 5, 6);
    case 4: return b;
    };
    assert(false);
    return int_vec{};
}
#endif

#if VEC_LANES == 8
inline int_vec extract_slice(int_vec a, int_vec b, int offset) {
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
    return int_vec{};
}
#endif

#if VEC_LANES == 16
inline int_vec extract_slice(int_vec a, int_vec b, int offset) {
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
    return int_vec{};
}
#endif

inline int_vec max(int_vec a, int_vec b) {
    return a > b ? a : b;
}

template<int taps>
struct Recurrence {
    int coeff[taps] = {};
    // optimal coefficients (for second-order and vec_lanes = 4, 
    // opt_coeff_offset[4] = max(4a, 2a+b, 2b)
    // opt_coeff_offset[5] = max(5a, 3a+b, a+2b)
    int opt_coeff_offset[taps + vec_lanes - 1] = {};
    int_vec prev_output0{}, prev_output1{};
    int_vec prev_input{};

    Recurrence(const std::vector<int> coeffs) {
        static_assert(taps < vec_lanes);
        assert(coeffs.size() == taps);
        assert(coeffs[0] == 0);
        int i = 0;
        for (int c : coeffs) {
            coeff[i++] = c;
        }

        
        for (int i = 0; i < taps + vec_lanes - 1; i++) {
            for (int j = 0; j < taps; j++) {
                if (i + j < taps + vec_lanes - 1) {
                    opt_coeff_offset[i + j] = std::max(opt_coeff_offset[i + j], opt_coeff_offset[i] + coeff[j]);
                }
            }
        }

        std::cout << "Delayed coefficients:\n";
        for (auto c: opt_coeff_offset) {
            std::cout << c << " ";
        }
        std::cout << "\n";
    }

    void reset() {
        prev_output0 = {};
        prev_output1 = {};
        prev_input = {};
    }

    void init(const int_vec *__restrict__ input, int_vec *__restrict__ output) {
        auto in_ptr = (const int *) input;
        auto out_ptr = (int *) output;

        int prev[taps - 1] = {};
#pragma unroll
        for (size_t i = 0; i < vec_lanes * 2; i++) {
            int next = *in_ptr++;
#pragma unroll
            for (int j = 1; j < taps; j++)  {
                next = std::max(next, coeff[j] + prev[j - 1]);
            }
#pragma unroll
            for (int i = 1; i < taps - 1; i++) {
                prev[i] = prev[i-1];
            }
            prev[0] = next;

            *out_ptr++ = next;
        }

        prev_input = *(input + 1);
        prev_output1 = *(output + 1);
        prev_output0 = *output;

    }

    void run(const int_vec *__restrict__ input, int_vec *__restrict__ output) {
        // covers the case i = 0
        int_vec inp = *input;
        int_vec result = inp;
#pragma unroll
        for (int i = 1; i < taps + vec_lanes - 1; i++) {
            if (i < vec_lanes) {
                int_vec v = extract_slice(inp, prev_input, i);
                result = max(result, v + opt_coeff_offset[i]);
            } else {
                int_vec v = extract_slice(prev_output0, prev_output1, vec_lanes - (i - vec_lanes));
                result = max(result, v + opt_coeff_offset[i]);
            }
        }

        prev_output0 = prev_output1;
        prev_output1 = result;
        prev_input = inp;

        *output = result;
    }
};

int main(int argc, char **argv) {

    int alpha = 2, beta = 5;

    const size_t size = 1024 * 16;
    const int iters = 1024 * 16;

    int *input = (int *)aligned_alloc(64, size * sizeof(int));
    int *output = (int *)aligned_alloc(64, size * sizeof(int));
    int *reference_output = (int *)aligned_alloc(64, size * sizeof(int));

    std::memset(input, 0, size * sizeof(int));
    std::memset(output, 0, size * sizeof(int));
    std::memset(reference_output, 0, size * sizeof(int));
    // input[0] = 1;

    Recurrence<3> algo({0, alpha, beta});

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iters; t++) {
        const int_vec *in_ptr = (const int_vec *)(input);
        int_vec *out_ptr = (int_vec *)(output);
        algo.reset();
        algo.init(in_ptr, out_ptr);
        in_ptr += 2;
        out_ptr += 2;
        for (size_t i = vec_lanes * 2; i < size; i += vec_lanes) {
            algo.run(in_ptr++, out_ptr++);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
#pragma unroll VEC_LANES
    for (int t = 0; t < iters; t++) {
        const int *in_ptr = input;
        int *out_ptr = reference_output;
        int prev0 = 0;
        int prev1 = 0;
        for (size_t i = 0; i < size; i++) {
            int next = std::max(std::max(alpha + prev0, beta + prev1), *in_ptr++);
            prev1 = prev0;
            prev0 = next;
            *out_ptr++ = next;
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        printf("%d %d\n", output[i], reference_output[i]);
    }
    printf("\n");

    free(input);
    free(output);

    double t_vec = std::chrono::duration<float>(t2 - t1).count();
    double t_scalar = std::chrono::duration<float>(t3 - t2).count();
    printf("Vectorized: %f seconds\n", t_vec);
    printf("Scalar: %f seconds\n", t_scalar);
    printf("Speed-up: %f\n", t_scalar / t_vec);
    printf("Vectorized throughput: %f giga-ints per second\n",
           size * iters / t_vec * 1e-9);

    return 0;
}
