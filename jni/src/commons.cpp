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

void knn_jni::commons::convertFP32ToFP16(knn_jni::JNIUtilInterface* jniUtil,
                                         JNIEnv* env,
                                         jfloatArray fp32Array,
                                         jbyteArray fp16Array,
                                         jint count) {
    if (count <= 0) return;

    jfloat* src_f32 = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(fp32Array, nullptr));
    jbyte* dst_bytes = reinterpret_cast<jbyte*>(env->GetPrimitiveArrayCritical(fp16Array, nullptr));
    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);

    int i = 0;

#if defined(__aarch64__)
    // ARM NEON 8-wide unrolled loop
    for (; i + 8 <= count; i += 8) {
        float32x4_t v0 = vld1q_f32(src + i);
        float32x4_t v1 = vld1q_f32(src + i + 4);
        float16x4_t h0 = vcvt_f16_f32(v0);
        float16x4_t h1 = vcvt_f16_f32(v1);
        vst1_f16(reinterpret_cast<__fp16*>(dst + i), h0);
        vst1_f16(reinterpret_cast<__fp16*>(dst + i + 4), h1);
    }
    // Tail - scalar cast
    for (; i < count; ++i) {
        dst[i] = static_cast<uint16_t>(__fp16(src[i]));
    }

#elif defined(__x86_64__)
  #if defined(__AVX512F__)
    for (; i + 16 <= count; i += 16) {
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        __m512 v = _mm512_loadu_ps(&src[i]);
        __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), h);
    }
  #elif defined(__AVX2__) && defined(__F16C__)
    for (; i + 16 <= count; i += 16) {
        if (i + 32 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 32]), _MM_HINT_T0);
        }
        __m256 v0 = _mm256_loadu_ps(&src[i]);
        __m256 v1 = _mm256_loadu_ps(&src[i + 8]);

        __m128i h0 = _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h1 = _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i]), h0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i + 8]), h1);
    }

  #else
    #error "x86_64 must support AVX512F or AVX2+F16C"
  #endif
    for (; i < count; ++i) {
        __m128 sv = _mm_set_ss(src[i]);
        __m128i hv = _mm_cvtps_ph(sv, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        dst[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(hv));
    }
#else
    #error "Only aarch64 or x86_64 supported"
#endif

    env->ReleasePrimitiveArrayCritical(fp16Array, dst_bytes, 0);
    env->ReleasePrimitiveArrayCritical(fp32Array, src_f32, JNI_ABORT);
}

void knn_jni::commons::convertFP16ToFP32(knn_jni::JNIUtilInterface* jniUtil,
                                         JNIEnv* env,
                                         jbyteArray fp16Array,
                                         jfloatArray fp32Array,
                                         jint count,
                                         jint offset) {
    if (count <= 0) return;

    jfloat* dst_f32 = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(fp32Array, nullptr));
    jbyte* src_bytes = reinterpret_cast<jbyte*>(env->GetPrimitiveArrayCritical(fp16Array, nullptr));
    float* dst = reinterpret_cast<float*>(dst_f32);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bytes + offset);

    int i = 0;

#if defined(__aarch64__)
    // ARM NEON 8-wide unrolled loop
    for (; i + 8 <= count; i += 8) {
        __builtin_prefetch(&src[i + 32], 0, 1);
        __builtin_prefetch(&dst[i + 32], 1, 1);
        float16x4_t h0 = vld1_f16(reinterpret_cast<const __fp16*>(&src[i]));
        float16x4_t h1 = vld1_f16(reinterpret_cast<const __fp16*>(&src[i + 4]));
        float32x4_t v0 = vcvt_f32_f16(h0);
        float32x4_t v1 = vcvt_f32_f16(h1);
        vst1q_f32(dst + i, v0);
        vst1q_f32(dst + i + 4, v1);
    }

    // Tail - scalar cast
    for (; i < count; ++i) {
        dst[i] = static_cast<float>(reinterpret_cast<const __fp16&>(src[i]));
    }

#elif defined(__x86_64__)
  #if defined(__AVX512F__)
    for (; i + 16 <= count; i += 16) {
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        __m512 v = _mm512_cvtph_ps(h);
        _mm512_storeu_ps(&dst[i], v);
    }
  #elif defined(__AVX2__) && defined(__F16C__)
    for (; i + 16 <= count; i += 16) {
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }

        __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[i]));
        __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[i + 8]));

        __m256 v0 = _mm256_cvtph_ps(h0);
        __m256 v1 = _mm256_cvtph_ps(h1);

        _mm256_storeu_ps(&dst[i], v0);
        _mm256_storeu_ps(&dst[i + 8], v1);
    }
  #else
    #error "x86_64 must support AVX512F or AVX2+F16C"
  #endif
    for (; i < count; ++i) {
        __m128i h = _mm_cvtsi32_si128(src[i]);
        __m128 v = _mm_cvtph_ps(h);
        dst[i] = _mm_cvtss_f32(v);
    }
#else
    #error "Only aarch64 or x86_64 supported"
#endif

    env->ReleasePrimitiveArrayCritical(fp32Array, dst_f32, 0);
    env->ReleasePrimitiveArrayCritical(fp16Array, src_bytes, JNI_ABORT);
}