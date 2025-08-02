#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

include(CheckCXXSourceCompiles)

# Check for x86 AVX512
set(CMAKE_REQUIRED_FLAGS "-mavx512f")
check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m512 x = _mm512_setzero_ps();
    __m256i h = _mm512_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    (void)h;
    return 0;
}" HAVE_X86_AVX512)
unset(CMAKE_REQUIRED_FLAGS)

if (HAVE_X86_AVX512)
    message(STATUS "AVX512F SIMD support detected")
    set(SIMD_FLAGS "-mavx512f")
    add_definitions(-DKNN_HAVE_AVX512)
endif()

# Check for x86 AVX2 + F16C (only if AVX512 not available)
if (NOT HAVE_X86_AVX512)
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mf16c")
    check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m128 x = _mm_setzero_ps();
        __m128i h = _mm_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        (void)h;
        return 0;
    }" HAVE_X86_F16C)
    unset(CMAKE_REQUIRED_FLAGS)

    if (HAVE_X86_F16C)
        message(STATUS "AVX2 + F16C SIMD support detected")
        set(SIMD_FLAGS "-mavx2 -mf16c")
        add_definitions(-DKNN_HAVE_F16C)
    endif()
endif()

# Check for ARM NEON FP16
check_cxx_source_compiles("
#include <arm_neon.h>
int main() {
    float32x4_t v = vdupq_n_f32(1.0f);
    float16x4_t h = vcvt_f16_f32(v);
    (void)h;
    return 0;
}" HAVE_ARM_FP16)

if (HAVE_ARM_FP16)
    message(STATUS "ARM NEON FP16 SIMD support detected")
    set(SIMD_FLAGS "-march=armv8.4-a+fp16")
    add_definitions(-DKNN_HAVE_ARM_FP16)
endif()

# Fallback
if (NOT HAVE_X86_AVX512 AND NOT HAVE_X86_F16C AND NOT HAVE_ARM_FP16)
    message(WARNING "No SIMD support detected. Falling back to default encoding/decoding in Java.")
    set(SIMD_FLAGS "")
endif()

# Always use these optimization flags for all SIMD variants
set(FP16_SIMD_FLAGS "-O3 -fPIC")
