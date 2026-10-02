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
#include <cstdint>
#include <cstring>

#include "jni_util.h"
#include "simd/fp16_codec/fp16_codec.h"

namespace knn_jni::simd::fp16_codec {

jboolean isSIMDSupported() {
    return JNI_FALSE;
}

jboolean encodeFp32ToFp16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env,
                           jfloatArray fp32Array, jbyteArray fp16Array, jint count) {
    return JNI_FALSE;
}

// Unlike the JNI entry points below, this one has a real implementation even on a build without
// SIMD: callers that already hold native memory (the SIMD search context's query buffer) need a
// working conversion regardless of whether the CPU offers a vector one. Written as a plain loop so
// the compiler can still vectorise it where the hardware allows.
void decodeFp16ToFp32Raw(const uint16_t* src, float* dst, size_t count) {
    for (size_t i = 0 ; i < count ; ++i) {
        const uint16_t half = src[i];
        const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
        const uint32_t exponent = (half >> 10) & 0x1Fu;
        const uint32_t mantissa = half & 0x3FFu;

        uint32_t bits;
        if (exponent == 0) {
            if (mantissa == 0) {
                // Signed zero.
                bits = sign;
            } else {
                // Subnormal half: renormalise into a normal float by shifting the mantissa up until
                // its implicit leading bit appears, charging each shift to the exponent.
                uint32_t shift = 0;
                uint32_t normalised = mantissa;
                while ((normalised & 0x400u) == 0) {
                    normalised <<= 1;
                    ++shift;
                }
                normalised &= 0x3FFu;
                bits = sign | ((127u - 14u - shift) << 23) | (normalised << 13);
            }
        } else if (exponent == 0x1Fu) {
            // Infinity or NaN; the mantissa carries the NaN payload.
            bits = sign | 0x7F800000u | (mantissa << 13);
        } else {
            // Normal half: rebias the exponent from 15 to 127.
            bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
        }

        std::memcpy(&dst[i], &bits, sizeof(float));
    }
}

jboolean decodeFp16ToFp32(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env,
                           jbyteArray fp16Array, jint offset, jfloatArray fp32Array, jint count) {
    return JNI_FALSE;
}

}  // namespace knn_jni::simd::fp16_codec
