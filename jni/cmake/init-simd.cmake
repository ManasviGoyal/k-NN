# SPDX-License-Identifier: Apache-2.0
# Copyright OpenSearch Contributors

include(CheckCXXSourceCompiles)

# ------------------ Handle user-overrides or set default ------------------
if(NOT DEFINED AVX2_ENABLED)
    set(AVX2_ENABLED true)
endif()

if(NOT DEFINED AVX512_ENABLED)
    set(AVX512_ENABLED true)
endif()

if(NOT DEFINED AVX512_SPR_ENABLED)
    execute_process(
        COMMAND bash -c "lscpu | grep -q 'GenuineIntel' && lscpu | grep -i 'avx512_fp16' | grep -i 'avx512_bf16' | grep -i 'avx512_vpopcntdq'"
        OUTPUT_VARIABLE SPR_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT "${SPR_FLAGS}" STREQUAL "")
        set(AVX512_SPR_ENABLED true)
    else()
        set(AVX512_SPR_ENABLED false)
    endif()
endif()

# ------------------ SIMD Detection ----------------------------------

set(SIMD_OPT_LEVEL "")
set(SIMD_FLAGS "")

# AVX512 detection (if enabled)
if(AVX512_ENABLED)
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

    if(HAVE_X86_AVX512)
        message(STATUS "AVX512F SIMD support detected")
        set(SIMD_OPT_LEVEL "avx512")
        set(SIMD_FLAGS -mavx512f)
        add_definitions(-DKNN_HAVE_AVX512)
    endif()
endif()

# AVX2 + F16C detection (if enabled and AVX512 wasn't selected)
if(AVX2_ENABLED AND (SIMD_OPT_LEVEL STREQUAL ""))
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

    if(HAVE_X86_F16C)
        message(STATUS "AVX2 + F16C SIMD support detected")
        set(SIMD_OPT_LEVEL "avx2")
        set(SIMD_FLAGS -mavx2 -mf16c)
        add_definitions(-DKNN_HAVE_F16C)
    endif()
endif()

# ARM NEON (if on aarch64 and nothing selected yet)
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64" AND (SIMD_OPT_LEVEL STREQUAL ""))
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main() {
            float32x4_t v = vdupq_n_f32(1.0f);
            float16x4_t h = vcvt_f16_f32(v);
            (void)h;
            return 0;
        }" HAVE_ARM_FP16)

    if(HAVE_ARM_FP16)
        message(STATUS "ARM NEON FP16 SIMD support detected")
        set(SIMD_OPT_LEVEL "neon")
        set(SIMD_FLAGS -march=armv8.4-a+fp16)
        add_definitions(-DKNN_HAVE_ARM_FP16)
    endif()
endif()

# Fallback — if nothing selected
if(SIMD_OPT_LEVEL STREQUAL "")
    message(WARNING "No SIMD support detected or all SIMD options disabled. Falling back to default encoding/decoding in Java.")
    set(SIMD_OPT_LEVEL "generic")
    set(SIMD_FLAGS "")
endif()

# Common optimization flags
set(FP16_SIMD_FLAGS "-O3" "-fPIC")

# Suffix for library name (used in CMakeLists.txt)
if(SIMD_OPT_LEVEL STREQUAL "avx512")
    set(SIMD_LIB_SUFFIX "_avx512")
elseif(SIMD_OPT_LEVEL STREQUAL "avx2")
    set(SIMD_LIB_SUFFIX "_avx2")
else()
    set(SIMD_LIB_SUFFIX "")
endif()
