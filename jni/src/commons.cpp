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
#include <vector>

#include "jni_util.h"
#include "commons.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

jlong knn_jni::commons::storeVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ, jboolean appendJ) {
    std::vector<float> *vect;
    if ((long) memoryAddressJ == 0) {
        vect = new std::vector<float>();
        vect->reserve((long)initialCapacityJ);
    } else {
        vect = reinterpret_cast<std::vector<float>*>(memoryAddressJ);
    }

    if (appendJ == JNI_FALSE) {
        vect->clear();
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaFloatArray(env, dataJ);
    jniUtil->Convert2dJavaObjectArrayAndStoreToFloatVector(env, dataJ, dim, vect);

    return (jlong) vect;
}

jlong knn_jni::commons::storeBinaryVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ, jboolean appendJ) {
    std::vector<uint8_t> *vect;
    if ((long) memoryAddressJ == 0) {
        vect = new std::vector<uint8_t>();
        vect->reserve((long)initialCapacityJ);
    } else {
        vect = reinterpret_cast<std::vector<uint8_t>*>(memoryAddressJ);
    }

    if (appendJ == JNI_FALSE) {
        vect->clear();
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaByteArray(env, dataJ);
    jniUtil->Convert2dJavaObjectArrayAndStoreToBinaryVector(env, dataJ, dim, vect);

    return (jlong) vect;
}

jlong knn_jni::commons::storeByteVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ, jboolean appendJ) {
    std::vector<int8_t> *vect;
    if (memoryAddressJ == 0) {
        vect = new std::vector<int8_t>();
        vect->reserve(static_cast<long>(initialCapacityJ));
    } else {
        vect = reinterpret_cast<std::vector<int8_t>*>(memoryAddressJ);
    }

    if (appendJ == JNI_FALSE) {
            vect->clear();
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaByteArray(env, dataJ);
    jniUtil->Convert2dJavaObjectArrayAndStoreToByteVector(env, dataJ, dim, vect);

    return (jlong) vect;
}

void knn_jni::commons::freeVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<float>*>(memoryAddressJ);
        delete vect;
    }
}

void knn_jni::commons::freeBinaryVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<uint8_t>*>(memoryAddressJ);
        delete vect;
    }
}

void knn_jni::commons::freeByteVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<int8_t>*>(memoryAddressJ);
        delete vect;
    }
}

int knn_jni::commons::getIntegerMethodParameter(JNIEnv * env, knn_jni::JNIUtilInterface * jniUtil, std::unordered_map<std::string, jobject> methodParams, std::string methodParam, int defaultValue) {
    if (methodParams.empty()) {
        return defaultValue;
    }
    auto efSearchIt = methodParams.find(methodParam);
    if (efSearchIt != methodParams.end()) {
        return jniUtil->ConvertJavaObjectToCppInteger(env, methodParams[methodParam]);
    }

    return defaultValue;
}

void knn_jni::commons::convertFP16ToFP32(knn_jni::JNIUtilInterface* jniUtil,
                                         JNIEnv* env,
                                         jbyteArray fp16Array,
                                         jfloatArray fp32Array,
                                         jint count,
                                         jint offset) {
    if (count <= 0) return;

    jbyte * fp16_bytes = (jbyte*) env->GetPrimitiveArrayCritical(fp16Array, nullptr);
    const __fp16* src = (const __fp16*) (fp16_bytes + offset);
    // jbyte*   fp16_bytes  = (jbyte*) env->GetPrimitiveArrayCritical(fp16Array, nullptr);
    jfloat*  fp32_floats = (jfloat*) env->GetPrimitiveArrayCritical(fp32Array, nullptr);

    // const __fp16* src = (const __fp16*) fp16_bytes;
    float* dst = fp32_floats;

    int vec_count = (count / 4) * 4;

    for (int i = 0; i < vec_count; i += 4) {
        float16x4_t h = vld1_f16(src + i);
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(dst + i, f);
    }

    for (int i = vec_count; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }

    env->ReleasePrimitiveArrayCritical(fp16Array,  fp16_bytes,  JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(fp32Array, fp32_floats, 0);
}

void knn_jni::commons::convertFP32ToFP16(knn_jni::JNIUtilInterface *jniUtil,
                                         JNIEnv* env,
                                         jfloatArray fp32Array,
                                         jbyteArray fp16Array,
                                         jint count) {
    if (count <= 0) return;

    jfloat* src_f32   = (jfloat*) env->GetPrimitiveArrayCritical(fp32Array, nullptr);
    jbyte*  dst_bytes = (jbyte*)  env->GetPrimitiveArrayCritical(fp16Array, nullptr);

    const float* src = (const float*) src_f32;
    __fp16*      dst = (__fp16*)      dst_bytes;

    int i = 0;
    // SIMD
    for (; i + 4 <= count; i += 4) {
        float32x4_t v_f32 = vld1q_f32(src + i);
        float16x4_t v_f16 = vcvt_f16_f32(v_f32);
        vst1_f16(dst + i, v_f16);
    }
    // Tail
    for (; i < count; ++i) {
        float16x4_t tmp = vcvt_f16_f32(vdupq_n_f32(src[i]));
        __fp16      h   = vget_lane_f16(tmp, 0);
        dst[i] = h;
    }

    env->ReleasePrimitiveArrayCritical(fp32Array,  src_f32,   JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(fp16Array, dst_bytes, 0);
}