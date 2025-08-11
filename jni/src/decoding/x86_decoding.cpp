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
#include "decoding/decoding.h"

// Returns JNI_TRUE to indicate that SIMD support is enabled at compile time for x86 architecture.
jboolean knn_jni::decoding::isSIMDSupported() {
    return JNI_TRUE;
}

/*
 * This function implements architecture-specific SIMD optimizations for converting FP16 values to FP32.
 * using x86_64 vector intrinsics. The conversion path is selected at compile time via preprocessor macros.
 * All of these intrinsics and instruction sets are documented in the official Intel Intrinsics Guide:
 * https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 */
jboolean knn_jni::decoding::convertFP16ToFP32(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env, jbyteArray fp16Array, jfloatArray fp32Array, jint count, jint offset) {
    // Return early if there's nothing to convert
    if (count <= 0) return JNI_TRUE;

    // Pin the destination Java float[] and get raw access to its memory
    jfloat* dst_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    // Pin the source Java byte[] and get raw access to its memory
    jbyte* src_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    // When 'release_arrays' goes out of scope, its lambda will run to ensure that the
    // critical arrays are always released properly even if there's an error or early return
    knn_jni::JNIReleaseElements release_arrays{[=]() {
        // Release the destination FP32 array, mode 0 means changes are written back
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, dst_f32, 0);
        // Release the source FP16 array, JNI_ABORT means we don't write changes back (read-only)
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, src_bytes, JNI_ABORT);
    }};

    // Ensure that the starting address is aligned to 2 bytes (required for correct uint16_t interpretation)
    if ((reinterpret_cast<uintptr_t>(src_bytes + offset) % alignof(uint16_t)) != 0) {
        return JNI_FALSE;  // release_arrays will still run its cleanup here
    }

    float* dst = reinterpret_cast<float*>(dst_f32);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bytes + offset);

    size_t i = 0;
#if defined(KNN_HAVE_AVX512_SPR)
    for (; i + 32 <= count; i += 32) {
        // Prefetch 128 FP16 elements (256 bytes) ahead into L1 cache.
        // Each AVX512_SPR iteration processes 32 FP16 values (64 bytes), so this prefetch is 4 iterations ahead.
        // This prefetches 4 full cache lines (assuming 64-byte cache line size)
        if (i + 128 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 128]), _MM_HINT_T0);
        }

        // Load and convert first 16 FP16 values
        __m256h h0 = _mm256_loadu_ph(&src[i]);
        __m512 v0 = _mm512_cvtph_ps(h0);

        // Load and convert next 16 FP16 values
        __m256h h1 = _mm256_loadu_ph(&src[i + 16]);
        __m512 v1 = _mm512_cvtph_ps(h1);

        // Store 32 FP32 values to memory
        _mm512_storeu_ps(&dst[i], v0);
        _mm512_storeu_ps(&dst[i + 16], v1);
    }
#elif defined(KNN_HAVE_AVX512)
    for (; i + 16 <= count; i += 16) {
        // Prefetch 64 FP16 values (128 bytes) ahead into L1 cache.
        // Each loop iteration processes 16 FP16 values (32 bytes), so this prefetch is 4 iterations ahead.
        // This gives the CPU time to load 2 cache lines before the data is accessed.
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        // Load 16 FP16 values (stored as 16-bit integers) into a 256-bit AVX register
        __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        // Convert 16 FP16 values to 16 FP32 values using AVX512 hardware support
        __m512 v = _mm512_cvtph_ps(h);
        // Store the 16 converted FP32 values into destination array
        _mm512_storeu_ps(&dst[i], v);
    }
#elif defined(KNN_HAVE_AVX2_F16C)
    for (; i + 8 <= count; i += 8) {
        // Prefetch 64 FP16 values (128 bytes) ahead into L1 cache.
        // Each AVX2 iteration processes 8 FP16 values (16 bytes), so this is 8 iterations ahead.
        // This prefetches 2 cache lines of upcoming data, ideal for avoiding stalls during vectorized loads.
        // It’s tuned for stride-1 access patterns in large arrays.
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        // Load 8 FP16 values (stored as 16-bit integers) into a 128-bit AVX register
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[i]));
        // Convert 8 FP16 values to 8 FP32 values using AVX2 + F16C instructions
        __m256 v = _mm256_cvtph_ps(h);
         // Store the 8 converted FP32 values into destination array
        _mm256_storeu_ps(&dst[i], v);
    }
#endif
    // Scalar fallback using F16C for remaining elements.
    // This path is taken if any elements remain after vectorized processing.
    // Converts one FP16 float at a time to FP32.
    for (; i < count; ++i) {
        // Load a single FP16 value into the lower 16 bits of an XMM register
        __m128i h = _mm_cvtsi32_si128(src[i]);
        // Convert the FP16 to a single-precision float (__m128)
        __m128 v = _mm_cvtph_ps(h);
        // Extract the lowest 32-bit float from __m128 and store it
        dst[i] = _mm_cvtss_f32(v);
    }

    // Arrays are released automatically by the RAII release_arrays lambda
    return JNI_TRUE;
}