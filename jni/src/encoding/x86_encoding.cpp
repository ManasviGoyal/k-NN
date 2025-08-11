/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#include <jni.h>
#if defined(__x86_64__) && (defined(KNN_HAVE_AVX512) || defined(KNN_HAVE_AVX2_F16C))
#include <immintrin.h>
#endif

#include <cstdint>

#include "jni_util.h"
#include "encoding/encoding.h"

// Returns JNI_TRUE to indicate that SIMD support is enabled at compile time for x86 architecture.
jboolean knn_jni::encoding::isSIMDSupported() {
    return JNI_TRUE;
}

/*
 * This function implements architecture-specific SIMD optimizations for converting FP32 values to FP16.
 * using x86_64 vector intrinsics. The conversion path is selected at compile time via preprocessor macros.
 * All of these intrinsics and instruction sets are documented in the official Intel Intrinsics Guide:
 * https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 */
jboolean knn_jni::encoding::convertFP32ToFP16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env, jfloatArray fp32Array, jbyteArray fp16Array, jint count) {
    // Return early if there's nothing to convert
    if (count <= 0) return JNI_TRUE;

    // Obtain direct in-place access to the Java byte[] backing store for FP16 results.
    // GetPrimitiveArrayCritical typically pins the Java heap object and returns its real address
    // (no copy on most JVMs, since they support pinning).
    // If pinning isn’t possible (rare on modern OpenJDK), it will fall back to copying,
    // so always check for NULL.
    // While pinned, GC compaction is disabled, so keep this critical region very short
    // to avoid stalling other threads or garbage collection.
    jfloat* src_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    jbyte* dst_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    // When 'release_arrays' goes out of scope, its lambda will run to ensure that the
    // critical arrays are always released properly even if there's an error or early return
    knn_jni::JNIReleaseElements release_arrays{[=]() {
        // Release the destination FP16 array, mode 0 means changes are written back
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, dst_bytes, 0);
        // Release the source FP32 array, JNI_ABORT means we don't write changes back (read-only)
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, src_f32, JNI_ABORT);
    }};

    // Ensure the FP16 destination buffer is 2-byte aligned.
    // The JVM will almost always give you an 8 or at least 4-byte-aligned buffer for any primitive array,
    // so the base pointer (dst_bytes or src_bytes) is naturally aligned to alignof(uint16_t)==2.
    // A misalignment here is therefore extremely rare, but this guard prevents undefined behavior or crashes.
    if ((reinterpret_cast<uintptr_t>(dst_bytes) % alignof(uint16_t)) != 0) {
        // release_arrays will cleanup the arrays automatically
        return JNI_FALSE;
    }

    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);

    size_t i = 0;
#if defined(KNN_HAVE_AVX512_SPR)
    for (; i + 32 <= count; i += 32) {
        // Load two __m512 (16 FP32 values each) to process 32 elements total
        __m512 v0 = _mm512_loadu_ps(&src[i]);
        __m512 v1 = _mm512_loadu_ps(&src[i + 16]);

        // Convert to two __m512h halves (each holds 16 FP16 values)
        // _mm512_cvtps_ph is only available on Intel Sapphire Rapids and newer CPUs with AVX512FP16.
        __m512h h0 = _mm512_cvtps_ph(v0);
        __m512h h1 = _mm512_cvtps_ph(v1);

        // Store each half (cast down to __m256i for correct size)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), _mm256_castph512_ph256(h0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i + 16]), _mm256_castph512_ph256(h1));
    }
#elif defined(KNN_HAVE_AVX512)
    for (; i + 16 <= count; i += 16) {
        // Load 16 float values (16 x 32 bits = 512 bits) into a __m512 register.
        // AVX512 registers are 512 bits wide, so they can hold 16 float32 values at once.
        __m512 v = _mm512_loadu_ps(&src[i]);
        // Convert the 16 FP32 values to FP16 and pack them into a 256-bit register (__m256i).
        // This uses round-to-nearest-even and disables floating-point exceptions.
        __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Store the 16 packed FP16 values (256 bits) to memory.
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), h);
    }
#elif defined(KNN_HAVE_AVX2_F16C)
    for (; i + 8 <= count; i += 8) {
        // Load 8 float values (8 x 32 bits = 256 bits) into a __m256 register.
        // F16C uses 256-bit registers to convert 8 FP32 values to FP16 in parallel.
        __m256 v = _mm256_loadu_ps(&src[i]);
        // Convert 8 FP32 values to FP16 and pack them into a 128-bit register (__m128i).
        // This uses round-to-nearest-even and disables floating-point exceptions.
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Stores 8 packed FP16 values (128 bits) into memory at &dst[i] using unaligned access.
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i]), h);
    }
#endif
    // Scalar fallback using F16C for remaining elements.
    // This path is taken if any elements remain after vectorized processing.
    // Converts one FP32 float at a time to FP16.
    for (; i < count; ++i) {
        // Load scalar float into the lowest part of __m128 register.
        __m128 sv = _mm_set_ss(src[i]);
        // Convert FP32 to FP16
        __m128i hv = _mm_cvtps_ph(sv, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Extract the lowest 16 bits as uint16_t
        dst[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(hv));
    }

    // Arrays are released automatically by the RAII release_arrays lambda
    return JNI_TRUE;
}