// Allow callers to skip re-including common.h if it was already embedded by
// another VecRec kernel header in the same translation unit.
#ifdef VECREC_COMMON_H_ALREADY_INCLUDED
// Nothing — all types and functions are already defined.
#else
#define VECREC_COMMON_H_ALREADY_INCLUDED

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

// GCC compatibility shims for Clang-specific elementwise builtins.
// __builtin_elementwise_max/min/add_sat are Clang 15+ extensions.
// GCC 12+ supports them via __builtin_reduce_max/__builtin_ia32_* intrinsics,
// but the simplest portable fallback is std::max/std::min + manual saturation.
#ifndef __clang__
#  include <type_traits>
template<typename T>
static inline T __builtin_elementwise_max(T a, T b) { return a > b ? a : b; }
template<typename T>
static inline T __builtin_elementwise_min(T a, T b) { return a < b ? a : b; }
template<typename T>
static inline T __builtin_elementwise_add_sat(T a, T b) {
    using U = typename std::make_unsigned<T>::type;
    static constexpr T lo = std::numeric_limits<T>::lowest();
    static constexpr T hi = std::numeric_limits<T>::max();
    T result;
    // Detect overflow: if signs of a and b are the same and result differs, saturate
    if (__builtin_add_overflow(a, b, &result)) {
        return (a < 0) ? lo : hi;
    }
    return result;
}
#endif
typedef float float_vec1;
typedef int32_t int32_vec1;
typedef int64_t int64_vec1;
typedef int8_t  int8_vec1;
typedef int16_t int16_vec1;

#ifdef __clang__
// Clang ext_vector_type: creates distinct SIMD vector types.
typedef float float_vec2 __attribute__((ext_vector_type(2)));
typedef float float_vec4 __attribute__((ext_vector_type(4)));
typedef float float_vec8 __attribute__((ext_vector_type(8)));
typedef float float_vec16 __attribute__((ext_vector_type(16)));

typedef double double_vec1;
typedef double double_vec2 __attribute__((ext_vector_type(2)));
typedef double double_vec4 __attribute__((ext_vector_type(4)));
typedef double double_vec8 __attribute__((ext_vector_type(8)));
typedef double double_vec16 __attribute__((ext_vector_type(16)));

typedef int32_t int32_vec2 __attribute__((ext_vector_type(2)));
typedef int32_t int32_vec4 __attribute__((ext_vector_type(4)));
typedef int32_t int32_vec8 __attribute__((ext_vector_type(8)));
typedef int32_t int32_vec16 __attribute__((ext_vector_type(16)));

typedef int64_t int64_vec2 __attribute__((ext_vector_type(2)));
typedef int64_t int64_vec4 __attribute__((ext_vector_type(4)));
typedef int64_t int64_vec8 __attribute__((ext_vector_type(8)));

typedef int8_t int8_vec2  __attribute__((ext_vector_type(2)));
typedef int8_t int8_vec4  __attribute__((ext_vector_type(4)));
typedef int8_t int8_vec8  __attribute__((ext_vector_type(8)));
typedef int8_t int8_vec16 __attribute__((ext_vector_type(16)));
typedef int8_t int8_vec32 __attribute__((ext_vector_type(32)));
typedef int8_t int8_vec64 __attribute__((ext_vector_type(64)));

typedef int16_t int16_vec2  __attribute__((ext_vector_type(2)));
typedef int16_t int16_vec4  __attribute__((ext_vector_type(4)));
typedef int16_t int16_vec8  __attribute__((ext_vector_type(8)));
typedef int16_t int16_vec16 __attribute__((ext_vector_type(16)));
typedef int16_t int16_vec32 __attribute__((ext_vector_type(32)));
// Oversized logical vectors (wider than one physical register): clang lowers
// ops on these by splitting across multiple SIMD registers. Used by wide-epi
// schedules (epi_multiple > native_lanes) to keep more vectors in flight.
typedef int16_t int16_vec64  __attribute__((ext_vector_type(64)));
typedef int16_t int16_vec128 __attribute__((ext_vector_type(128)));
typedef int32_t int32_vec32  __attribute__((ext_vector_type(32)));
typedef int32_t int32_vec64  __attribute__((ext_vector_type(64)));
typedef int8_t  int8_vec128  __attribute__((ext_vector_type(128)));

constexpr int vec_lanes_of(float)        { return 1; }
constexpr int vec_lanes_of(float_vec2)   { return 2; }
constexpr int vec_lanes_of(float_vec4)   { return 4; }
constexpr int vec_lanes_of(float_vec8)   { return 8; }
constexpr int vec_lanes_of(float_vec16)  { return 16; }
constexpr int vec_lanes_of(double)       { return 1; }
constexpr int vec_lanes_of(double_vec2)  { return 2; }
constexpr int vec_lanes_of(double_vec4)  { return 4; }
constexpr int vec_lanes_of(double_vec8)  { return 8; }
constexpr int vec_lanes_of(double_vec16) { return 16; }
constexpr int vec_lanes_of(int32_t)      { return 1; }
constexpr int vec_lanes_of(int32_vec2)   { return 2; }
constexpr int vec_lanes_of(int32_vec4)   { return 4; }
constexpr int vec_lanes_of(int32_vec8)   { return 8; }
constexpr int vec_lanes_of(int32_vec16)  { return 16; }
constexpr int vec_lanes_of(int64_t)      { return 1; }
constexpr int vec_lanes_of(int64_vec2)   { return 2; }
constexpr int vec_lanes_of(int64_vec4)   { return 4; }
constexpr int vec_lanes_of(int64_vec8)   { return 8; }
constexpr int vec_lanes_of(int8_t)       { return 1; }
constexpr int vec_lanes_of(int8_vec2)    { return 2; }
constexpr int vec_lanes_of(int8_vec4)    { return 4; }
constexpr int vec_lanes_of(int8_vec8)    { return 8; }
constexpr int vec_lanes_of(int8_vec16)   { return 16; }
constexpr int vec_lanes_of(int8_vec32)   { return 32; }
constexpr int vec_lanes_of(int8_vec64)   { return 64; }
constexpr int vec_lanes_of(int16_t)      { return 1; }
constexpr int vec_lanes_of(int16_vec2)   { return 2; }
constexpr int vec_lanes_of(int16_vec4)   { return 4; }
constexpr int vec_lanes_of(int16_vec8)   { return 8; }
constexpr int vec_lanes_of(int16_vec16)  { return 16; }
constexpr int vec_lanes_of(int16_vec32)  { return 32; }
constexpr int vec_lanes_of(int16_vec64)  { return 64; }
constexpr int vec_lanes_of(int16_vec128) { return 128; }
constexpr int vec_lanes_of(int32_vec32)  { return 32; }
constexpr int vec_lanes_of(int32_vec64)  { return 64; }
constexpr int vec_lanes_of(int8_vec128)  { return 128; }

#else  // GCC or other: no distinct vector types; kernel runs scalar/SSE intrinsics via -march=native
// GCC ignores ext_vector_type so we alias the multi-lane types to the scalar.
// The kernel schedules with epi=1 work correctly with scalar fallback.
typedef float   float_vec2, float_vec4, float_vec8, float_vec16;
typedef double  double_vec2, double_vec4, double_vec8, double_vec16;
typedef int32_t int32_vec2, int32_vec4, int32_vec8, int32_vec16;
typedef int64_t int64_vec2, int64_vec4, int64_vec8;
typedef int8_t  int8_vec2, int8_vec4, int8_vec8, int8_vec16, int8_vec32, int8_vec64;
typedef int16_t int16_vec2, int16_vec4, int16_vec8, int16_vec16, int16_vec32;

constexpr int vec_lanes_of(float)   { return 1; }
constexpr int vec_lanes_of(double)  { return 1; }
constexpr int vec_lanes_of(int32_t) { return 1; }
constexpr int vec_lanes_of(int64_t) { return 1; }
constexpr int vec_lanes_of(int8_t)  { return 1; }
constexpr int vec_lanes_of(int16_t) { return 1; }
#endif  // __clang__

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
struct ElementType<float> { using type = float; };
template <>
struct ElementType<int32_t> { using type = int32_t; };
template <>
struct ElementType<int64_t> { using type = int64_t; };

template <>
struct ElementType<double> { using type = double; };

#ifdef __clang__
template <> struct ElementType<float_vec2>  { using type = float; };
template <> struct ElementType<float_vec4>  { using type = float; };
template <> struct ElementType<float_vec8>  { using type = float; };
template <> struct ElementType<float_vec16> { using type = float; };
template <> struct ElementType<double_vec2>  { using type = double; };
template <> struct ElementType<double_vec4>  { using type = double; };
template <> struct ElementType<double_vec8>  { using type = double; };
template <> struct ElementType<double_vec16> { using type = double; };
template <> struct ElementType<int32_vec2>  { using type = int32_t; };
template <> struct ElementType<int32_vec4>  { using type = int32_t; };
template <> struct ElementType<int32_vec8>  { using type = int32_t; };
template <> struct ElementType<int32_vec16> { using type = int32_t; };
template <> struct ElementType<int64_vec2>  { using type = int64_t; };
template <> struct ElementType<int64_vec4>  { using type = int64_t; };
template <> struct ElementType<int64_vec8>  { using type = int64_t; };
#endif  // __clang__

template <> struct ElementType<int8_t>  { using type = int8_t; };
template <> struct ElementType<int16_t> { using type = int16_t; };

#ifdef __clang__
template <> struct ElementType<int8_vec2>   { using type = int8_t; };
template <> struct ElementType<int8_vec4>   { using type = int8_t; };
template <> struct ElementType<int8_vec8>   { using type = int8_t; };
template <> struct ElementType<int8_vec16>  { using type = int8_t; };
template <> struct ElementType<int8_vec32>  { using type = int8_t; };
template <> struct ElementType<int8_vec64>  { using type = int8_t; };
template <> struct ElementType<int16_vec2>  { using type = int16_t; };
template <> struct ElementType<int16_vec4>  { using type = int16_t; };
template <> struct ElementType<int16_vec8>  { using type = int16_t; };
template <> struct ElementType<int16_vec16> { using type = int16_t; };
template <> struct ElementType<int16_vec32> { using type = int16_t; };
template <> struct ElementType<int16_vec64>  { using type = int16_t; };
template <> struct ElementType<int16_vec128> { using type = int16_t; };
template <> struct ElementType<int32_vec32>  { using type = int32_t; };
template <> struct ElementType<int32_vec64>  { using type = int32_t; };
template <> struct ElementType<int8_vec128>  { using type = int8_t; };
#endif  // __clang__

#ifdef __clang__
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
#endif  // __clang__

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
EXTRACT_SLICE_1(int32_vec1)
EXTRACT_SLICE_1(int64_vec1)
EXTRACT_SLICE_1(int8_vec1)
EXTRACT_SLICE_1(int16_vec1)

#ifdef __clang__
EXTRACT_SLICE_2(float_vec2)
EXTRACT_SLICE_4(float_vec4)
EXTRACT_SLICE_8(float_vec8)
EXTRACT_SLICE_16(float_vec16)
EXTRACT_SLICE_1(double)
EXTRACT_SLICE_2(double_vec2)
EXTRACT_SLICE_4(double_vec4)
EXTRACT_SLICE_8(double_vec8)
EXTRACT_SLICE_16(double_vec16)
EXTRACT_SLICE_2(int32_vec2)
EXTRACT_SLICE_4(int32_vec4)
EXTRACT_SLICE_8(int32_vec8)
EXTRACT_SLICE_16(int32_vec16)
EXTRACT_SLICE_2(int64_vec2)
EXTRACT_SLICE_4(int64_vec4)
EXTRACT_SLICE_8(int64_vec8)
EXTRACT_SLICE_2(int8_vec2)
EXTRACT_SLICE_4(int8_vec4)
EXTRACT_SLICE_8(int8_vec8)
EXTRACT_SLICE_16(int8_vec16)
EXTRACT_SLICE_2(int16_vec2)
EXTRACT_SLICE_4(int16_vec4)
EXTRACT_SLICE_8(int16_vec8)
EXTRACT_SLICE_16(int16_vec16)
#endif  // __clang__

// EXTRACT_SLICE_32 and EXTRACT_SLICE_64 for int8_vec32 / int8_vec64
#define EXTRACT_SLICE_32(type_vec32)                                                                                                              \
    inline type_vec32 extract_slice(type_vec32 a, type_vec32 b, int offset) {                                                                     \
        switch (offset) {                                                                                                                         \
        case 0: return a;                                                                                                                         \
        case 1:  return __builtin_shufflevector(a, b,  1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32); \
        case 2:  return __builtin_shufflevector(a, b,  2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33); \
        case 3:  return __builtin_shufflevector(a, b,  3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34); \
        case 4:  return __builtin_shufflevector(a, b,  4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35); \
        case 5:  return __builtin_shufflevector(a, b,  5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36); \
        case 6:  return __builtin_shufflevector(a, b,  6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37); \
        case 7:  return __builtin_shufflevector(a, b,  7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38); \
        case 8:  return __builtin_shufflevector(a, b,  8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39); \
        case 9:  return __builtin_shufflevector(a, b,  9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40); \
        case 10: return __builtin_shufflevector(a, b, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41); \
        case 11: return __builtin_shufflevector(a, b, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42); \
        case 12: return __builtin_shufflevector(a, b, 12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43); \
        case 13: return __builtin_shufflevector(a, b, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44); \
        case 14: return __builtin_shufflevector(a, b, 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45); \
        case 15: return __builtin_shufflevector(a, b, 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46); \
        case 16: return __builtin_shufflevector(a, b, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47); \
        case 17: return __builtin_shufflevector(a, b, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48); \
        case 18: return __builtin_shufflevector(a, b, 18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49); \
        case 19: return __builtin_shufflevector(a, b, 19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50); \
        case 20: return __builtin_shufflevector(a, b, 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51); \
        case 21: return __builtin_shufflevector(a, b, 21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52); \
        case 22: return __builtin_shufflevector(a, b, 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53); \
        case 23: return __builtin_shufflevector(a, b, 23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54); \
        case 24: return __builtin_shufflevector(a, b, 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55); \
        case 25: return __builtin_shufflevector(a, b, 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56); \
        case 26: return __builtin_shufflevector(a, b, 26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57); \
        case 27: return __builtin_shufflevector(a, b, 27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58); \
        case 28: return __builtin_shufflevector(a, b, 28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59); \
        case 29: return __builtin_shufflevector(a, b, 29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60); \
        case 30: return __builtin_shufflevector(a, b, 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61); \
        case 31: return __builtin_shufflevector(a, b, 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62); \
        case 32: return b;                                                                                                                        \
        };                                                                                                                                        \
        assert(false);                                                                                                                            \
        return type_vec32{};                                                                                                                      \
    }

#define EXTRACT_SLICE_64(type_vec64)                                                                                                                                                                                                  \
    inline type_vec64 extract_slice(type_vec64 a, type_vec64 b, int offset) {                                                                                                                                                         \
        switch (offset) {                                                                                                                                                                                                             \
        case 0: return a;                                                                                                                                                                                                             \
        case 1:  return __builtin_shufflevector(a, b,  1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64); \
        case 2:  return __builtin_shufflevector(a, b,  2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65); \
        case 3:  return __builtin_shufflevector(a, b,  3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66); \
        case 4:  return __builtin_shufflevector(a, b,  4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67); \
        case 5:  return __builtin_shufflevector(a, b,  5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68); \
        case 6:  return __builtin_shufflevector(a, b,  6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69); \
        case 7:  return __builtin_shufflevector(a, b,  7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70); \
        case 8:  return __builtin_shufflevector(a, b,  8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71); \
        case 9:  return __builtin_shufflevector(a, b,  9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72); \
        case 10: return __builtin_shufflevector(a, b, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73); \
        case 11: return __builtin_shufflevector(a, b, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74); \
        case 12: return __builtin_shufflevector(a, b, 12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75); \
        case 13: return __builtin_shufflevector(a, b, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76); \
        case 14: return __builtin_shufflevector(a, b, 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77); \
        case 15: return __builtin_shufflevector(a, b, 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78); \
        case 16: return __builtin_shufflevector(a, b, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79); \
        case 17: return __builtin_shufflevector(a, b, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80); \
        case 18: return __builtin_shufflevector(a, b, 18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81); \
        case 19: return __builtin_shufflevector(a, b, 19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82); \
        case 20: return __builtin_shufflevector(a, b, 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83); \
        case 21: return __builtin_shufflevector(a, b, 21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84); \
        case 22: return __builtin_shufflevector(a, b, 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85); \
        case 23: return __builtin_shufflevector(a, b, 23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86); \
        case 24: return __builtin_shufflevector(a, b, 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87); \
        case 25: return __builtin_shufflevector(a, b, 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88); \
        case 26: return __builtin_shufflevector(a, b, 26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89); \
        case 27: return __builtin_shufflevector(a, b, 27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90); \
        case 28: return __builtin_shufflevector(a, b, 28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91); \
        case 29: return __builtin_shufflevector(a, b, 29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92); \
        case 30: return __builtin_shufflevector(a, b, 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93); \
        case 31: return __builtin_shufflevector(a, b, 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94); \
        case 32: return __builtin_shufflevector(a, b, 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95); \
        case 33: return __builtin_shufflevector(a, b, 33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96); \
        case 34: return __builtin_shufflevector(a, b, 34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97); \
        case 35: return __builtin_shufflevector(a, b, 35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98); \
        case 36: return __builtin_shufflevector(a, b, 36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99); \
        case 37: return __builtin_shufflevector(a, b, 37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100); \
        case 38: return __builtin_shufflevector(a, b, 38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101); \
        case 39: return __builtin_shufflevector(a, b, 39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102); \
        case 40: return __builtin_shufflevector(a, b, 40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103); \
        case 41: return __builtin_shufflevector(a, b, 41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104); \
        case 42: return __builtin_shufflevector(a, b, 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105); \
        case 43: return __builtin_shufflevector(a, b, 43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106); \
        case 44: return __builtin_shufflevector(a, b, 44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107); \
        case 45: return __builtin_shufflevector(a, b, 45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108); \
        case 46: return __builtin_shufflevector(a, b, 46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109); \
        case 47: return __builtin_shufflevector(a, b, 47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110); \
        case 48: return __builtin_shufflevector(a, b, 48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111); \
        case 49: return __builtin_shufflevector(a, b, 49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112); \
        case 50: return __builtin_shufflevector(a, b, 50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113); \
        case 51: return __builtin_shufflevector(a, b, 51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114); \
        case 52: return __builtin_shufflevector(a, b, 52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115); \
        case 53: return __builtin_shufflevector(a, b, 53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116); \
        case 54: return __builtin_shufflevector(a, b, 54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117); \
        case 55: return __builtin_shufflevector(a, b, 55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118); \
        case 56: return __builtin_shufflevector(a, b, 56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119); \
        case 57: return __builtin_shufflevector(a, b, 57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120); \
        case 58: return __builtin_shufflevector(a, b, 58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121); \
        case 59: return __builtin_shufflevector(a, b, 59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122); \
        case 60: return __builtin_shufflevector(a, b, 60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123); \
        case 61: return __builtin_shufflevector(a, b, 61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124); \
        case 62: return __builtin_shufflevector(a, b, 62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125); \
        case 63: return __builtin_shufflevector(a, b, 63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126); \
        case 64: return b;                                                                                                                                                                                                            \
        };                                                                                                                                                                                                                            \
        assert(false);                                                                                                                                                                                                                \
        return type_vec64{};                                                                                                                                                                                                          \
    }

#ifdef __clang__
EXTRACT_SLICE_32(int8_vec32)
EXTRACT_SLICE_32(int16_vec32)
EXTRACT_SLICE_32(int32_vec32)
EXTRACT_SLICE_64(int8_vec64)
EXTRACT_SLICE_64(int16_vec64)
#endif  // __clang__

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
EXTRACT_SUB_VECTOR1(double)
EXTRACT_SUB_VECTOR2(double_vec2)
EXTRACT_SUB_VECTOR4(double_vec4)
EXTRACT_SUB_VECTOR8(double_vec8)
EXTRACT_SUB_VECTOR1(int32_t)
EXTRACT_SUB_VECTOR2(int32_vec2)
EXTRACT_SUB_VECTOR4(int32_vec4)
EXTRACT_SUB_VECTOR8(int32_vec8)
EXTRACT_SUB_VECTOR1(int64_t)
EXTRACT_SUB_VECTOR2(int64_vec2)
EXTRACT_SUB_VECTOR4(int64_vec4)

#define EXTRACT_SUB_VECTOR16(type_vec16)                                                                                                                              \
    template <typename vec_type_in, int n>                                                                                                                            \
    struct ExtractSubVector<vec_type_in, type_vec16, n> {                                                                                                             \
        static type_vec16 extract_sub_vector(vec_type_in a) {                                                                                                         \
            return __builtin_shufflevector(a, a, 16*n, 16*n+1, 16*n+2, 16*n+3, 16*n+4, 16*n+5, 16*n+6, 16*n+7,                                                      \
                                                 16*n+8, 16*n+9, 16*n+10, 16*n+11, 16*n+12, 16*n+13, 16*n+14, 16*n+15);                                              \
        }                                                                                                                                                             \
    };

#define EXTRACT_SUB_VECTOR32(type_vec32)                                                                                                                              \
    template <typename vec_type_in, int n>                                                                                                                            \
    struct ExtractSubVector<vec_type_in, type_vec32, n> {                                                                                                             \
        static type_vec32 extract_sub_vector(vec_type_in a) {                                                                                                         \
            return __builtin_shufflevector(a, a, 32*n,    32*n+1,  32*n+2,  32*n+3,  32*n+4,  32*n+5,  32*n+6,  32*n+7,                                              \
                                                 32*n+8,  32*n+9,  32*n+10, 32*n+11, 32*n+12, 32*n+13, 32*n+14, 32*n+15,                                             \
                                                 32*n+16, 32*n+17, 32*n+18, 32*n+19, 32*n+20, 32*n+21, 32*n+22, 32*n+23,                                             \
                                                 32*n+24, 32*n+25, 32*n+26, 32*n+27, 32*n+28, 32*n+29, 32*n+30, 32*n+31);                                            \
        }                                                                                                                                                             \
    };

EXTRACT_SUB_VECTOR1(int8_t)
EXTRACT_SUB_VECTOR2(int8_vec2)
EXTRACT_SUB_VECTOR4(int8_vec4)
EXTRACT_SUB_VECTOR8(int8_vec8)
EXTRACT_SUB_VECTOR16(int8_vec16)
EXTRACT_SUB_VECTOR32(int8_vec32)

EXTRACT_SUB_VECTOR1(int16_t)
EXTRACT_SUB_VECTOR2(int16_vec2)
EXTRACT_SUB_VECTOR4(int16_vec4)
EXTRACT_SUB_VECTOR8(int16_vec8)
EXTRACT_SUB_VECTOR16(int16_vec16)
EXTRACT_SUB_VECTOR32(int16_vec32)

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
            /* fence removed */
            reg[base + i] = vals[i];
        }
    }

    void reset_and_next_row() {}
    void seek_col(int /* col_start */) {}
};

template <int _taps, typename vec_type, const int indices[_taps]>
struct TimeVaryingKernel {
    constexpr static const int *idxs = indices;
    constexpr static int taps = _taps;

    TimeVaryingKernel() = default;

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, const int args[taps]) {
        _Pragma("unroll") for (int i = 0; i < taps; i++) {
            /* fence removed */
            reg[base + i] = reg[args[i]];
        }
    }

    void reset_and_next_row() {}
    void seek_col(int /* col_start */) {}
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
    void seek_col(int /* col_start */) {}
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
    void seek_col(int /* col_start */) {}
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
    void seek_col(int /* col_start */) {}
};

// Context struct to share state between Repeater and Signal2D.
//
// When band_width > 0, each row only writes columns [col_start, col_end) to
// the buffer. Positions 0..col_start-1 are never updated and would retain the
// PREVIOUS row's values, which would then be read as the current row's values
// by the next row — a correctness bug. reset_and_next_row() zeroes those
// positions after every row so the buffer always represents M[row][j] = 0
// for j < col_start (the banded-DP "not reached" sentinel). This zeroing is
// O(col_start) ≈ O(band_width) per row and is compiled away when band_width == 0.
//
// epi_multiple: the ratio epi/lanes (= ele_per_iter_multiple in CodeGen). col_start
// must be aligned to epi = lanes * epi_multiple to avoid buffer overflow when the
// last vector group in a row extends past phys_w. Default 1 = align to lanes only.
template <typename vec_type, int n_rows, int band_width = 0, int epi_multiple = 1>
struct RepeaterContext {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    static constexpr int epi = lanes * epi_multiple;
    int width;
    elt_type *buffer;
    int current_row;       // ring-buffer slot index
    int logical_row;       // monotonic row counter used to compute col_start
    elt_type zero_val;     // semiring zero; used to fill skipped positions
    // When you write a value, the old value gets evicted, yet
    // Repeater still needs to return it.
    vec_type overwritten_value;

    RepeaterContext() = delete;
    RepeaterContext(int row_size, elt_type zero_value)
        : width(row_size), current_row(0), logical_row(0), zero_val(zero_value) {
        // Allocate extra elements beyond (n_rows-1)*width so the kernel's
        // vec_type look-ahead reads within Signal2D and SConvolve don't overflow.
        // We add 2*epi (= 2*lanes*epi_multiple) extra elements as a safe margin.
        const int alloc_size = (n_rows - 1) * width + 2 * epi;
        buffer = new (std::align_val_t(32)) elt_type[alloc_size];
        for (int i = 0; i < alloc_size; i++) {
            buffer[i] = zero_value;
        }
    }

    // Block-aware get: read from the row at offset `row_offset` relative to the
    // logical "current row" of block-row `block_row`.
    vec_type get_at(int row_offset, int col, int block_row, bool after_write) {
        int row_idx = (current_row + block_row - row_offset + 2 * (n_rows - 1)) % (n_rows - 1);
        if (row_offset == n_rows - 1 && after_write) {
            return overwritten_value;
        }
        vec_type result;
        memcpy(&result, &buffer[row_idx * width + col], sizeof(vec_type));
        return result;
    }

    // Block-aware set: write to the row at offset `block_row` relative to the
    // current row (i.e. the slot that becomes (current_row + block_row) mod n_rows).
    void set_at(vec_type value, int col, int block_row) {
        int row_idx = (current_row + block_row) % (n_rows - 1);
        memcpy(&overwritten_value, &buffer[row_idx * width + col], sizeof(vec_type));
        memcpy(&buffer[row_idx * width + col], &value, sizeof(vec_type));
    }

    void reset_and_next_row() {
        if constexpr (band_width > 0) {
            // Zero positions the just-completed row (logical_row) did not write.
            // Align to epi (not just lanes) to match the runner and avoid buffer overflow.
            // Left side always starts at col=0, so no stale-buffer positions exist
            // on the left.  No zeroing needed (the buffer is always written from 0).
            (void)logical_row;  // suppress unused-variable warning
        }
        current_row = (current_row + 1) % (n_rows - 1);
        logical_row++;
    }

    // Read one element from the previous row's buffer (row_offset=1) for diagonal priming.
    elt_type prev_row_scalar(int col) const {
        int row_idx = (current_row + 2 * (n_rows - 1) - 1) % (n_rows - 1);
        return buffer[row_idx * width + col];
    }
    void seek_col(int /* col_start */) {}  // no column state; present for uniform seek dispatch

    // ---- Tiling support (2D / row-strip restart) ----------------------------------
    // Reposition the logical row counter to `row_start` without touching buffer data.
    // Used by block tiling so a kernel restarted at the top of a row-band computes
    // with the correct row phase (matters when combined with load_prev_row below).
    void seek_row(int row_start) {
        logical_row = row_start;
        current_row = ((row_start % (n_rows - 1)) + (n_rows - 1)) % (n_rows - 1);
    }
    // Copy the previous-row slot's columns [col, col+n) into dst[col, col+n).
    // (dst is indexed by absolute column so multiple panels share one buffer.)
    void save_prev_row(elt_type *dst, int col, int n) const {
        int row_idx = (current_row + 2 * (n_rows - 1) - 1) % (n_rows - 1);
        memcpy(dst + col, &buffer[row_idx * width + col], (size_t)n * sizeof(elt_type));
    }
    // Inverse of save_prev_row: preload the previous-row slot from src[col, col+n).
    void load_prev_row(const elt_type *src, int col, int n) {
        int row_idx = (current_row + 2 * (n_rows - 1) - 1) % (n_rows - 1);
        memcpy(&buffer[row_idx * width + col], src + col, (size_t)n * sizeof(elt_type));
    }

    
    RepeaterContext(const RepeaterContext &) = delete;
    RepeaterContext &operator=(const RepeaterContext &) = delete;

    RepeaterContext(RepeaterContext &&other) noexcept
        : width(other.width), buffer(other.buffer), current_row(other.current_row),
          logical_row(other.logical_row), zero_val(other.zero_val) {
        other.buffer = nullptr;
    }

    RepeaterContext &operator=(RepeaterContext &&other) noexcept {
        if (this != &other) {
            delete[] buffer;
            width = other.width;
            current_row = other.current_row;
            logical_row = other.logical_row;
            zero_val = other.zero_val;
            buffer = other.buffer;
            other.buffer = nullptr;
        }
        return *this;
    }

    ~RepeaterContext() {
        delete[] buffer;
    }
};

template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct Signal2D {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int lanes = vec_lanes_of(vec_type{});

    int my_col;

    Signal2D() : my_col(0) {}

    template <typename Reg_t, typename Ctx>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, Ctx &context) {
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            /* fence removed */
            reg[base + i] = context.get_at(i + 1, my_col, block_row, false);
        }
        my_col += lanes;
    }

    void reset_and_next_row() {
        my_col = 0;
    }

    void seek_col(int col_start) { my_col = col_start; }
};

// InputSignal2D: reads from a user-provided 2D array with causal (look-behind) semantics.
// d[0] is the current row, d[1] is the previous row, ..., d[n_rows-1] is n_rows-1 rows back.
// Rows before row 0 are treated as zero (zero-padding).
// Row pointers are kept as elt_type* to avoid alignment assumptions; memcpy
// is used for the actual load so that unaligned input data is handled safely.
//
// When band_width > 0, only columns in [max(0, first_row_of_block - band_width),
// min(stride, first_row_of_block + block_size + band_width)] are visited per row.
// The runner loop is responsible for restricting column iteration; this struct
// initialises row pointers to the block's col_start so the first run() lands there.
// epi_multiple = ele_per_iter / lanes = how many SIMD loads per kernel.run() call.
// col_start must align to epi = lanes * epi_multiple so that the last vector group in
// a row does not spill beyond phys_w and corrupt adjacent rows.
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0, int band_width = 0, int epi_multiple = 1>
struct InputSignal2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    static constexpr int epi = lanes * epi_multiple;

    const elt_type *data;
    int stride;
    int current_row;
    elt_type *zero_row;
    // points into data for valid rows, or zero_row.get() for zero-padding rows
    const elt_type *row_ptrs[n_rows];

    // Returns the column start for the current block (shared by all block_rows).
    // first_row = current_row - block_row (the first logical row of this block).
    //
    // FIX — diagonal carry bug: after reset_and_next_row() all SIMD carry registers
    // are 0.  If the col loop starts at col_start > 0, the diagonal TIKernel carry
    // (which should hold M(i-1, col_start - lanes)) is 0, giving wrong scores.
    //
    // Fix: never restrict the LEFT side.  Always start from col = 0 so the carry is
    // the correct boundary value (0 = before sequence start).  Only col_end is
    // restricted (right/diagonal side).  This is a triangular band — the semantically
    // correct shape for sequence extension (alignment always starts at position 0).
    inline int _block_col_start() const {
        return 0;  // left side always starts at 0; only col_end (right side) is clipped
    }

    InputSignal2D() = delete;
    InputSignal2D(const elt_type *data, int stride, elt_type zero)
        : data(data),
          stride(stride),
          current_row(block_row),
          zero_row(new elt_type[stride + 2 * lanes]) {   // +2*lanes: prevents read-past-end in run()
        _Pragma("unroll") for (int i = 0; i < stride + 2 * lanes; i++) {
            zero_row[i] = zero;
        }
        int cs = _block_col_start();
        // d[i] = row (block_row - i), zero-padded for rows before 0.
        // Pointers start one step before cs so the first run() += lanes lands at cs.
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int src_row = block_row - i;
            row_ptrs[i] = ((src_row >= 0) ? data + (size_t)src_row * stride + cs : zero_row) - lanes;
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
        // Each call advances by exactly one row. The blocked runner calls
        // reset_and_next_row exactly `block_size` times per block, so all the
        // per-block-row InputSignal2D instances advance by `block_size` total
        // (matching the way RepeaterContext::current_row is bumped).
        current_row++;
        int cs = _block_col_start();
        // d[i] = row (current_row - i), zero-padded for rows before 0
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int src_row = current_row - i;
            row_ptrs[i] = ((src_row >= 0) ? data + (size_t)src_row * stride + cs : zero_row) - lanes;
        }
    }

    void seek_col(int col_start) {
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int src_row = current_row - i;
            row_ptrs[i] = ((src_row >= 0) ? data + (size_t)src_row * stride : zero_row)
                          + col_start - lanes;
        }
    }

    // Tiling support: reposition the row counter (block tiling restarts mid-matrix).
    // A subsequent seek_col() recomputes row_ptrs from this current_row.
    void seek_row(int row_start) { current_row = row_start; }
};

// ProfileSignal2D: a drop-in InputSignal2D replacement that reads a row from a
// small per-symbol PROFILE instead of a full O(rows x cols) matrix. For row r it
// uses profile[sym[r]] (one of `alphabet` precomputed reference score-vectors),
// the parasail-style "query profile" approach. This lets the alignment kernels
// avoid materializing the O(N^2) substitution matrix: the driver builds only the
// O(alphabet*cols) profile + the O(rows) symbol-index table `sym`.
//
// band_width > 0 mirrors InputSignal2D banding: row pointers are initialised to
// the block's col_start so the column loop can start there instead of column 0.
// epi_multiple matches InputSignal2D's parameter; col_start is aligned to epi.
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0, int band_width = 0, int epi_multiple = 1>
struct ProfileSignal2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    static constexpr int epi = lanes * epi_multiple;

    const elt_type *profile;   // alphabet rows of length `stride`
    const int *sym;            // sym[r] selects which profile row logical row r uses
    int stride;
    int current_row;
    elt_type *zero_row;
    const elt_type *row_ptrs[n_rows];

    // Same fix as InputSignal2D: no left restriction.
    inline int _block_col_start() const {
        return 0;
    }

    ProfileSignal2D() = delete;
    ProfileSignal2D(const elt_type *profile, const int *sym, int stride, elt_type zero)
        : profile(profile), sym(sym), stride(stride),
          current_row(block_row), zero_row(new elt_type[stride + 2 * lanes]) {  // +2*lanes: prevents read-past-end
        _Pragma("unroll") for (int i = 0; i < stride + 2 * lanes; i++) zero_row[i] = zero;
        int cs = _block_col_start();
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int src_row = block_row - i;
            row_ptrs[i] = ((src_row >= 0) ? profile + (size_t)sym[src_row] * stride + cs : zero_row) - lanes;
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
        int cs = _block_col_start();
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int src_row = current_row - i;
            row_ptrs[i] = ((src_row >= 0) ? profile + (size_t)sym[src_row] * stride + cs : zero_row) - lanes;
        }
    }

    void seek_col(int col_start) {
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int src_row = current_row - i;
            const elt_type *base = (src_row >= 0)
                ? profile + (size_t)sym[src_row] * stride
                : zero_row;
            row_ptrs[i] = base + col_start - lanes;
        }
    }

    // Tiling support: reposition the row counter (block tiling restarts mid-matrix).
    // A subsequent seek_col() recomputes row_ptrs from this current_row.
    void seek_row(int row_start) { current_row = row_start; }
};

template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct Repeater {
    using elt_type = typename ElementType<vec_type>::type;
    constexpr static int vec_lanes = vec_lanes_of(vec_type{});

    int my_col;

    Repeater() : my_col(0) {}

    template <typename Reg_t, typename Ctx>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int inner, Ctx &context) {
        vec_type curr_output = reg[inner];
        // Skip the cache write for the first (block_size - n_rows) block-rows,
        // since their values are dead before any future block reads them.
        if constexpr (block_row > block_size - n_rows) {
            context.set_at(curr_output, my_col, block_row);
        }
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            reg[base + i] = context.get_at(i, my_col, block_row, true);
        }
        my_col += vec_lanes;
    }

    void reset_and_next_row() {
        my_col = 0;
    }

    void seek_col(int col_start) { my_col = col_start; }
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
                /* fence removed */                                                                  \
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
        void seek_col(int /* col_start */) {}                                                          \
        void prime_diagonal_carry(elt_type val) {                                                      \
            reinterpret_cast<elt_type *>(&prev_input[buffer_size - 1])[vec_lanes - 1] = val;           \
        }                                                                                              \
        /* Tiling support: checkpoint/restore the diagonal shift carry at a column   */                \
        /* boundary so a tiled driver can restart the recurrence mid-matrix.         */                \
        void save_state(char *&dst) const {                                                            \
            memcpy(dst, prev_input, sizeof(prev_input)); dst += sizeof(prev_input);                    \
        }                                                                                              \
        void restore_state(const char *&src) {                                                         \
            memcpy(prev_input, src, sizeof(prev_input)); src += sizeof(prev_input);                    \
        }                                                                                              \
    };                                                                                                 \
    // template <typename vec_type>                                                                           \
    // auto make_s_convolve##suffix(int base, int kernel, int signal) {                                       \
    //     return SConvolve##suffix<vec_type>(base, kernel, signal);                                          \
    // }

SConvolveOp(, 0, [](auto a, auto b) { return a * b; }, [](auto acc, auto x) { return acc + x; });
// SConvolveOp(, 0, [](auto a, auto b) { return safe_mul(a, b); }, [](auto acc, auto x) { return acc + x; });
SConvolveOp(TropMin, std::numeric_limits<elt_type>::max() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SConvolveOp(TropMax, std::numeric_limits<elt_type>::lowest() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });
// Saturating variants for int8: use min/2 sentinel and saturating add to prevent overflow
SConvolveOp(TropMinSat, std::numeric_limits<elt_type>::max() / 2, [](auto a, auto b) { return __builtin_elementwise_add_sat(a, b); }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SConvolveOp(TropMaxSat, std::numeric_limits<elt_type>::lowest() / 2, [](auto a, auto b) { return __builtin_elementwise_add_sat(a, b); }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

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
                /* fence removed */                                                                  \
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
        void seek_col(int /* col_start */) {}                                                          \
        /* Tiling support: checkpoint/restore the left-dependency (in-row) carry at a */                \
        /* column boundary so a tiled driver can restart the recurrence mid-matrix.  */                \
        void prime_left_carry(elt_type val) {                                                          \
            reinterpret_cast<elt_type *>(&prev_output[buffer_size - 1])[vec_lanes - 1] = val;          \
        }                                                                                              \
        void save_state(char *&dst) const {                                                            \
            memcpy(dst, prev_output, sizeof(prev_output)); dst += sizeof(prev_output);                 \
        }                                                                                              \
        void restore_state(const char *&src) {                                                         \
            memcpy(prev_output, src, sizeof(prev_output)); src += sizeof(prev_output);                 \
        }                                                                                              \
    };

SRecurseOp(, 0, [](auto a, auto b) { return a * b; }, [](auto acc, auto x) { return acc + x; });
// SRecurseOp(, 0, [](auto a, auto b) { return safe_mul(a, b); }, [](auto acc, auto x) { return acc + x; });
SRecurseOp(TropMin, std::numeric_limits<elt_type>::max() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SRecurseOp(TropMax, std::numeric_limits<elt_type>::lowest() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });
SRecurseOp(TropMinSat, std::numeric_limits<elt_type>::max() / 2, [](auto a, auto b) { return __builtin_elementwise_add_sat(a, b); }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SRecurseOp(TropMaxSat, std::numeric_limits<elt_type>::lowest() / 2, [](auto a, auto b) { return __builtin_elementwise_add_sat(a, b); }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

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
    void seek_col(int /* col_start */) {}                                                                 \
    /* Tiling support: checkpoint/restore all per-component left carries.        */                      \
    void save_state(char *&dst) const {                                                                  \
        memcpy(dst, prev_output, sizeof(prev_output)); dst += sizeof(prev_output);                       \
    }                                                                                                     \
    void restore_state(const char *&src) {                                                               \
        memcpy(prev_output, src, sizeof(prev_output)); src += sizeof(prev_output);                       \
    }                                                                                                     \
};


        // for (int i = 0; i < n_vec; i++) { for (int j = 0; j < comp_buf_sizes_arr[i]; j++) { fprintf(stderr, "%f ", prev_output[comp_buf_offsets_arr[i] + j]); }}                                                                                                  \
        // printf("\n");                                                                                         \

SRecurseVOp(, 0, [](auto a, auto b) { return a * b; }, [](auto acc, auto x) { return acc + x; });
SRecurseVOp(TropMin, std::numeric_limits<elt_type>::max() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SRecurseVOp(TropMax, std::numeric_limits<elt_type>::lowest() / 8, [](auto a, auto b) { return a + b; }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });
SRecurseVOp(TropMinSat, std::numeric_limits<elt_type>::max() / 2, [](auto a, auto b) { return __builtin_elementwise_add_sat(a, b); }, [](auto acc, auto x) { return __builtin_elementwise_min(acc, x); });
SRecurseVOp(TropMaxSat, std::numeric_limits<elt_type>::lowest() / 2, [](auto a, auto b) { return __builtin_elementwise_add_sat(a, b); }, [](auto acc, auto x) { return __builtin_elementwise_max(acc, x); });

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
    void seek_col(int /* col_start */) {}
};

// SelectMax##suffix: argmax/argmin selection over n candidate registers.
//
// Template parameters:
//   vec_type      – SIMD vector element type (output stores the winning label)
//   n             – number of candidate value signals
//   labels[n]     – integer label emitted for each candidate (compile-time)
//
// run(reg, base, srcs[n]):
//   – reads candidate values from reg[srcs[0..n-1]]
//   – scans candidates left→right keeping a running best value and its label
//   – `better_op(cand, best)` is the strict comparison that decides whether the
//     new candidate beats the running best; strict comparison means earlier
//     candidates win ties (deterministic argmax: "keep latest best, or take i")
//   – `reduce_op` folds the running best (max for TropMax, min for TropMin)
//   – writes the winning label (splat across lanes) to reg[base]
#define SelectMaxOp(suffix, better_op, reduce_op)                                                  \
    template <typename vec_type, int n, const int labels[n]>                                       \
    struct SelectMax##suffix {                                                                     \
        using elt_type = typename ElementType<vec_type>::type;                                     \
        SelectMax##suffix() = default;                                                             \
        template <typename Reg_t>                                                                  \
        inline __attribute__((always_inline)) void run(Reg_t &reg, int base, const int srcs[n]) {  \
            vec_type best = reg[srcs[0]];                                                          \
            vec_type idx = (elt_type)labels[0];                                                    \
            _Pragma("unroll") for (int k = 1; k < n; k++) {                                        \
                vec_type cand = reg[srcs[k]];                                                      \
                auto mask = better_op(cand, best);                                                 \
                vec_type lbl = (elt_type)labels[k];                                                \
                idx = mask ? lbl : idx;                                                            \
                best = reduce_op(best, cand);                                                      \
            }                                                                                      \
            reg[base] = idx;                                                                       \
        }                                                                                          \
        void reset_and_next_row() {}                                                                    \
        void seek_col(int /* col_start */) {}                                                          \
    };

SelectMaxOp(TropMax, [](auto cand, auto best) { return cand > best; }, [](auto a, auto b) { return __builtin_elementwise_max(a, b); });
SelectMaxOp(TropMin, [](auto cand, auto best) { return cand < best; }, [](auto a, auto b) { return __builtin_elementwise_min(a, b); });
SelectMaxOp(TropMaxSat, [](auto cand, auto best) { return cand > best; }, [](auto a, auto b) { return __builtin_elementwise_max(a, b); });
SelectMaxOp(TropMinSat, [](auto cand, auto best) { return cand < best; }, [](auto a, auto b) { return __builtin_elementwise_min(a, b); });

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
    void seek_col(int /* col_start */) {}
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
            /* fence removed */
            out[i] = reg_in[inner_base + i];
        }
        memcpy(&reg_out[base], out, sizeof(vec_type_out));
    }

    void reset_and_next_row() {}
    void seek_col(int /* col_start */) {}
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
    void seek_col(int /* col_start */) {}
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
    void seek_col(int /* col_start */) {}
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
    void seek_col(int /* col_start */) {}
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
    void seek_col(int /* col_start */) {}
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
        void seek_col(int /* col_start */) {}                                                  \
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
// Saturating variants for int8 tropical ops
BinOp(SAddTropMaxSat, [](auto left, auto right) { return __builtin_elementwise_max(left, right); });
BinOp(SAddTropMinSat, [](auto left, auto right) { return __builtin_elementwise_min(left, right); });
BinOp(PointwiseMulTropMaxSat, [](auto left, auto right) { return __builtin_elementwise_add_sat(left, right); });
BinOp(PointwiseMulTropMinSat, [](auto left, auto right) { return __builtin_elementwise_add_sat(left, right); });

template <typename vec_type>
struct SNeg {
    SNeg() = default;

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base, int s1) {
        reg[base] = -reg[s1];
    }

    void reset_and_next_row() {}
    void seek_col(int /* col_start */) {}
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

// Mov: copy one register to another. Used by blocked codegen to forward
// in-block previous-row outputs from earlier block-rows' inner registers
// into the current block-row's Repeater/Signal2D output positions.
template <typename vec_type>
struct Mov {
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int dst, int src) {
        reg[dst] = reg[src];
    }
    void reset_and_next_row() {}
    void seek_col(int /* col_start */) {}
};

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
    void seek_col(int /* col_start */) {}
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
    void seek_col(int /* col_start */) {}
};

// Primary templates — specializations below for each concrete vec_type.
template <typename vec_type> struct Unzip2 { void reset_and_next_row() {} void seek_col(int /* col_start */) {} };
template <typename vec_type> struct Zip2   { void reset_and_next_row() {} void seek_col(int /* col_start */) {} };

#define VECREC_DEFINE_UNZIP2_ZIP2_K1(vec_type) \
template <> struct Unzip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_even, int out_odd, int in_lo, int in_hi) { \
        auto lo = reg[in_lo]; auto hi = reg[in_hi]; \
        reg[out_even] = lo; \
        reg[out_odd]  = hi; \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
}; \
template <> struct Zip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_lo, int out_hi, int in_even, int in_odd) { \
        auto e = reg[in_even]; auto o = reg[in_odd]; \
        reg[out_lo] = e; \
        reg[out_hi] = o; \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
};

VECREC_DEFINE_UNZIP2_ZIP2_K1(float_vec1)
VECREC_DEFINE_UNZIP2_ZIP2_K1(double_vec1)
VECREC_DEFINE_UNZIP2_ZIP2_K1(int8_vec1)
VECREC_DEFINE_UNZIP2_ZIP2_K1(int16_vec1)
VECREC_DEFINE_UNZIP2_ZIP2_K1(int32_vec1)
VECREC_DEFINE_UNZIP2_ZIP2_K1(int64_vec1)
#undef VECREC_DEFINE_UNZIP2_ZIP2_K1

// K=2 specializations (float_vec2, int32_vec2, int64_vec2)
// Unzip2: in = [e0,o0](lo) ++ [e1,o1](hi) -> even=[e0,e1], odd=[o0,o1]
// Zip2:   even=[e0,e1], odd=[o0,o1] -> lo=[e0,o0], hi=[e1,o1]
#define VECREC_DEFINE_UNZIP2_ZIP2_K2(vec_type) \
template <> struct Unzip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_even, int out_odd, int in_lo, int in_hi) { \
        auto lo = reg[in_lo]; auto hi = reg[in_hi]; \
        reg[out_even] = __builtin_shufflevector(lo, hi, 0, 2); \
        reg[out_odd]  = __builtin_shufflevector(lo, hi, 1, 3); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
}; \
template <> struct Zip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_lo, int out_hi, int in_even, int in_odd) { \
        auto e = reg[in_even]; auto o = reg[in_odd]; \
        reg[out_lo] = __builtin_shufflevector(e, o, 0, 2); \
        reg[out_hi] = __builtin_shufflevector(e, o, 1, 3); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
};

VECREC_DEFINE_UNZIP2_ZIP2_K2(float_vec2)
VECREC_DEFINE_UNZIP2_ZIP2_K2(double_vec2)
VECREC_DEFINE_UNZIP2_ZIP2_K2(int8_vec2)
VECREC_DEFINE_UNZIP2_ZIP2_K2(int16_vec2)
VECREC_DEFINE_UNZIP2_ZIP2_K2(int32_vec2)
VECREC_DEFINE_UNZIP2_ZIP2_K2(int64_vec2)
#undef VECREC_DEFINE_UNZIP2_ZIP2_K2

// K=4 specializations (float_vec4, int32_vec4, int64_vec4)
// Unzip2: in=[e0,o0,e1,o1](lo)++[e2,o2,e3,o3](hi) -> even=[e0,e1,e2,e3], odd=[o0,o1,o2,o3]
// Zip2:   even=[e0..e3], odd=[o0..o3] -> lo=[e0,o0,e1,o1], hi=[e2,o2,e3,o3]
#define VECREC_DEFINE_UNZIP2_ZIP2_K4(vec_type) \
template <> struct Unzip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_even, int out_odd, int in_lo, int in_hi) { \
        auto lo = reg[in_lo]; auto hi = reg[in_hi]; \
        reg[out_even] = __builtin_shufflevector(lo, hi, 0, 2, 4, 6); \
        reg[out_odd]  = __builtin_shufflevector(lo, hi, 1, 3, 5, 7); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
}; \
template <> struct Zip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_lo, int out_hi, int in_even, int in_odd) { \
        auto e = reg[in_even]; auto o = reg[in_odd]; \
        reg[out_lo] = __builtin_shufflevector(e, o, 0, 4, 1, 5); \
        reg[out_hi] = __builtin_shufflevector(e, o, 2, 6, 3, 7); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
};

VECREC_DEFINE_UNZIP2_ZIP2_K4(float_vec4)
VECREC_DEFINE_UNZIP2_ZIP2_K4(double_vec4)
VECREC_DEFINE_UNZIP2_ZIP2_K4(int8_vec4)
VECREC_DEFINE_UNZIP2_ZIP2_K4(int16_vec4)
VECREC_DEFINE_UNZIP2_ZIP2_K4(int32_vec4)
VECREC_DEFINE_UNZIP2_ZIP2_K4(int64_vec4)
#undef VECREC_DEFINE_UNZIP2_ZIP2_K4

// K=8 specializations (float_vec8, int32_vec8, int64_vec8)
// Unzip2: even indices 0,2,4,6,8,10,12,14; odd 1,3,5,7,9,11,13,15
// Zip2:   lo=[e0,o0,e1,o1,e2,o2,e3,o3] odd=[e4,o4,e5,o5,e6,o6,e7,o7]
#define VECREC_DEFINE_UNZIP2_ZIP2_K8(vec_type) \
template <> struct Unzip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_even, int out_odd, int in_lo, int in_hi) { \
        auto lo = reg[in_lo]; auto hi = reg[in_hi]; \
        reg[out_even] = __builtin_shufflevector(lo, hi, 0, 2, 4, 6, 8, 10, 12, 14); \
        reg[out_odd]  = __builtin_shufflevector(lo, hi, 1, 3, 5, 7, 9, 11, 13, 15); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
}; \
template <> struct Zip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_lo, int out_hi, int in_even, int in_odd) { \
        auto e = reg[in_even]; auto o = reg[in_odd]; \
        reg[out_lo] = __builtin_shufflevector(e, o, 0, 8, 1, 9, 2, 10, 3, 11); \
        reg[out_hi] = __builtin_shufflevector(e, o, 4, 12, 5, 13, 6, 14, 7, 15); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
};

VECREC_DEFINE_UNZIP2_ZIP2_K8(float_vec8)
VECREC_DEFINE_UNZIP2_ZIP2_K8(double_vec8)
VECREC_DEFINE_UNZIP2_ZIP2_K8(int8_vec8)
VECREC_DEFINE_UNZIP2_ZIP2_K8(int16_vec8)
VECREC_DEFINE_UNZIP2_ZIP2_K8(int32_vec8)
VECREC_DEFINE_UNZIP2_ZIP2_K8(int64_vec8)
#undef VECREC_DEFINE_UNZIP2_ZIP2_K8

// K=16 specializations (float_vec16, int32_vec16)
// Unzip2: even 0,2,...,30; odd 1,3,...,31
// Zip2:   lo=[e0,o0,...,e7,o7], hi=[e8,o8,...,e15,o15]
#define VECREC_DEFINE_UNZIP2_ZIP2_K16(vec_type) \
template <> struct Unzip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_even, int out_odd, int in_lo, int in_hi) { \
        auto lo = reg[in_lo]; auto hi = reg[in_hi]; \
        reg[out_even] = __builtin_shufflevector(lo, hi, \
            0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30); \
        reg[out_odd]  = __builtin_shufflevector(lo, hi, \
            1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
}; \
template <> struct Zip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_lo, int out_hi, int in_even, int in_odd) { \
        auto e = reg[in_even]; auto o = reg[in_odd]; \
        reg[out_lo] = __builtin_shufflevector(e, o, \
            0, 16,  1, 17,  2, 18,  3, 19,  4, 20,  5, 21,  6, 22,  7, 23); \
        reg[out_hi] = __builtin_shufflevector(e, o, \
            8, 24,  9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
};

VECREC_DEFINE_UNZIP2_ZIP2_K16(float_vec16)
VECREC_DEFINE_UNZIP2_ZIP2_K16(double_vec16)
VECREC_DEFINE_UNZIP2_ZIP2_K16(int8_vec16)
VECREC_DEFINE_UNZIP2_ZIP2_K16(int16_vec16)
VECREC_DEFINE_UNZIP2_ZIP2_K16(int32_vec16)
#undef VECREC_DEFINE_UNZIP2_ZIP2_K16

// K=32 specializations (int8_vec32)
// Unzip2: even 0,2,...,62; odd 1,3,...,63
// Zip2:   lo=[e0,o0,...,e15,o15], hi=[e16,o16,...,e31,o31]
#define VECREC_DEFINE_UNZIP2_ZIP2_K32(vec_type) \
template <> struct Unzip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_even, int out_odd, int in_lo, int in_hi) { \
        auto lo = reg[in_lo]; auto hi = reg[in_hi]; \
        reg[out_even] = __builtin_shufflevector(lo, hi, \
             0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, \
            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62); \
        reg[out_odd]  = __builtin_shufflevector(lo, hi, \
             1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, \
            33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
}; \
template <> struct Zip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_lo, int out_hi, int in_even, int in_odd) { \
        auto e = reg[in_even]; auto o = reg[in_odd]; \
        reg[out_lo] = __builtin_shufflevector(e, o, \
             0, 32,  1, 33,  2, 34,  3, 35,  4, 36,  5, 37,  6, 38,  7, 39, \
             8, 40,  9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47); \
        reg[out_hi] = __builtin_shufflevector(e, o, \
            16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, \
            24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
};

VECREC_DEFINE_UNZIP2_ZIP2_K32(int8_vec32)
VECREC_DEFINE_UNZIP2_ZIP2_K32(int16_vec32)
#undef VECREC_DEFINE_UNZIP2_ZIP2_K32

// K=64 specializations (int8_vec64)
// Unzip2: even 0,2,...,126; odd 1,3,...,127
// Zip2:   lo=[e0,o0,...,e31,o31], hi=[e32,o32,...,e63,o63]
#define VECREC_DEFINE_UNZIP2_ZIP2_K64(vec_type) \
template <> struct Unzip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_even, int out_odd, int in_lo, int in_hi) { \
        auto lo = reg[in_lo]; auto hi = reg[in_hi]; \
        reg[out_even] = __builtin_shufflevector(lo, hi, \
              0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,  28,  30, \
             32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62, \
             64,  66,  68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94, \
             96,  98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126); \
        reg[out_odd]  = __builtin_shufflevector(lo, hi, \
              1,   3,   5,   7,   9,  11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31, \
             33,  35,  37,  39,  41,  43,  45,  47,  49,  51,  53,  55,  57,  59,  61,  63, \
             65,  67,  69,  71,  73,  75,  77,  79,  81,  83,  85,  87,  89,  91,  93,  95, \
             97,  99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
}; \
template <> struct Zip2<vec_type> { \
    template <typename Reg_t> \
    inline __attribute__((always_inline)) void \
    run(Reg_t &reg, int out_lo, int out_hi, int in_even, int in_odd) { \
        auto e = reg[in_even]; auto o = reg[in_odd]; \
        reg[out_lo] = __builtin_shufflevector(e, o, \
              0,  64,   1,  65,   2,  66,   3,  67,   4,  68,   5,  69,   6,  70,   7,  71, \
              8,  72,   9,  73,  10,  74,  11,  75,  12,  76,  13,  77,  14,  78,  15,  79, \
             16,  80,  17,  81,  18,  82,  19,  83,  20,  84,  21,  85,  22,  86,  23,  87, \
             24,  88,  25,  89,  26,  90,  27,  91,  28,  92,  29,  93,  30,  94,  31,  95); \
        reg[out_hi] = __builtin_shufflevector(e, o, \
             32,  96,  33,  97,  34,  98,  35,  99,  36, 100,  37, 101,  38, 102,  39, 103, \
             40, 104,  41, 105,  42, 106,  43, 107,  44, 108,  45, 109,  46, 110,  47, 111, \
             48, 112,  49, 113,  50, 114,  51, 115,  52, 116,  53, 117,  54, 118,  55, 119, \
             56, 120,  57, 121,  58, 122,  59, 123,  60, 124,  61, 125,  62, 126,  63, 127); \
    } \
    void reset_and_next_row() {}                                                                           \
        void seek_col(int /* col_start */) {}                                                          \
};

VECREC_DEFINE_UNZIP2_ZIP2_K64(int8_vec64)
#undef VECREC_DEFINE_UNZIP2_ZIP2_K64

#endif  // VECREC_COMMON_H_ALREADY_INCLUDED

