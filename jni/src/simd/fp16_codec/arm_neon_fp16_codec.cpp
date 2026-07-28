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
#include <arm_neon.h>
#include <cstdint>

#include "jni_util.h"
#include "simd/fp16_codec/fp16_codec.h"

namespace knn_jni::simd::fp16_codec {

jboolean isSIMDSupported() {
    return JNI_TRUE;
}

jboolean encodeFp32ToFp16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env,
                           jfloatArray fp32Array, jbyteArray fp16Array, jint count) {
    if (count <= 0) return JNI_TRUE;

    jfloat* src_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    jbyte* dst_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    knn_jni::JNIReleaseElements release_arrays{[=]() {
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, dst_bytes, 0);
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, src_f32, JNI_ABORT);
    }};

    if ((reinterpret_cast<uintptr_t>(dst_bytes) % alignof(uint16_t)) != 0) {
        return JNI_FALSE;
    }

    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);

    int i = 0;

    // NEON: process 16 elements per iteration (2x unrolled)
    for (; i + 16 <= count; i += 16) {
        float32x4_t v0 = vld1q_f32(&src[i]);
        float32x4_t v1 = vld1q_f32(&src[i + 4]);
        float32x4_t v2 = vld1q_f32(&src[i + 8]);
        float32x4_t v3 = vld1q_f32(&src[i + 12]);
        float16x4_t h0 = vcvt_f16_f32(v0);
        float16x4_t h1 = vcvt_f16_f32(v1);
        float16x4_t h2 = vcvt_f16_f32(v2);
        float16x4_t h3 = vcvt_f16_f32(v3);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i]), h0);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i + 4]), h1);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i + 8]), h2);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i + 12]), h3);
    }

    // NEON tail: process 8 elements
    for (; i + 8 <= count; i += 8) {
        float32x4_t v0 = vld1q_f32(&src[i]);
        float32x4_t v1 = vld1q_f32(&src[i + 4]);
        float16x4_t h0 = vcvt_f16_f32(v0);
        float16x4_t h1 = vcvt_f16_f32(v1);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i]), h0);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i + 4]), h1);
    }

    // NEON tail: process 4 elements
    if (i + 4 <= count) {
        float32x4_t v0 = vld1q_f32(&src[i]);
        float16x4_t h0 = vcvt_f16_f32(v0);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i]), h0);
        i += 4;
    }

    // Scalar fallback for remaining elements
    for (; i < count; ++i) {
        reinterpret_cast<__fp16*>(dst)[i] = static_cast<__fp16>(src[i]);
    }

    return JNI_TRUE;
}

}  // namespace knn_jni::simd::fp16_codec
