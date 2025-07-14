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
#include <string>
#include <cmath>

#include <arm_neon.h>
#include <vector>

#include "jni_util.h"
#include "commons.h"

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

jfloatArray knn_jni::commons::bytesToFloatArray(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env, jbyteArray halfFloatBytes) {
    jsize byteLength = jniUtil->GetJavaBytesArrayLength(env, halfFloatBytes);
    if (byteLength % 2 != 0) {
        jniUtil->ThrowJavaException(env, "java/lang/IllegalArgumentException", "Byte array length must be even (2 bytes per FP16 value)");
        return nullptr;
    }

    jsize numElements = byteLength / 2;
    if (numElements == 0) {
        jniUtil->ThrowJavaException(env, "java/lang/IllegalArgumentException", "Byte array must contain at least one FP16 element");
        return nullptr;
    }

    const jbyte* rawBytes = jniUtil->GetByteArrayElements(env, halfFloatBytes, nullptr);
    if (!rawBytes) {
        jniUtil->ThrowJavaException(env, "java/lang/RuntimeException", "Unable to access byte array");
        return nullptr;
    }

    const uint8_t* inputBytes = reinterpret_cast<const uint8_t*>(rawBytes);
    std::vector<float> outputFloats(numElements);

    auto decodeFP16 = [](uint16_t h) -> float {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp  = (h & 0x7C00) >> 10;
        uint32_t mant = (h & 0x03FF);
        uint32_t f;

        if (exp == 0x1F) {
            f = sign | 0x7F800000 | (mant << 13);  // Inf/NaN
        } else if (exp != 0) {
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);  // Normalized
        } else if (mant != 0) {
            // Subnormal number
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FF;
            f = sign | ((127 - 15 - exp + 1) << 23) | (mant << 13);
        } else {
            f = sign;  // Zero
        }

        union { uint32_t u; float f; } u = { f };
        return u.f;
    };

    // Convert from Java big-endian byte pairs (no need to byte-swap manually)
    for (jsize i = 0; i < numElements; ++i) {
        uint16_t h = (static_cast<uint8_t>(inputBytes[2 * i]) << 8) |
                      static_cast<uint8_t>(inputBytes[2 * i + 1]);

        outputFloats[i] = decodeFP16(h);

        // Optional sanity check
        if (std::isnan(outputFloats[i]) || std::isinf(outputFloats[i])) {
            jniUtil->ThrowJavaException(env, "java/lang/IllegalArgumentException", "Invalid FP16 value decoded (NaN or Inf)");
            jniUtil->ReleaseByteArrayElements(env, halfFloatBytes, const_cast<jbyte*>(rawBytes), JNI_ABORT);
            return nullptr;
        }
    }

    jniUtil->ReleaseByteArrayElements(env, halfFloatBytes, const_cast<jbyte*>(rawBytes), JNI_ABORT);

    jfloatArray result = env->NewFloatArray(numElements);
    if (!result) {
        jniUtil->ThrowJavaException(env, "java/lang/RuntimeException", "Failed to allocate float array");
        return nullptr;
    }

    jniUtil->SetFloatArrayRegion(env, result, 0, numElements, outputFloats.data());
    return result;
}

void knn_jni::commons::convertFP16ToFP32(knn_jni::JNIUtilInterface *jniUtil,
                                        JNIEnv* env,
                                        jbyteArray fp16Array,
                                        jfloatArray fp32Array,
                                        jint count) {
    if (count <= 0) return;

    // Pin Java arrays
    jbyte* fp16_bytes = reinterpret_cast<jbyte*>(
            env->GetPrimitiveArrayCritical(fp16Array, nullptr));
    jfloat* fp32_floats = reinterpret_cast<jfloat*>(
            env->GetPrimitiveArrayCritical(fp32Array, nullptr));

    const __fp16* src = reinterpret_cast<const __fp16*>(fp16_bytes);
    float* dst = fp32_floats;

    int vec_count = (count / 4) * 4;

    for (int i = 0; i < count; i += 4) {
        float16x4_t h = vld1_f16(src + i);
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(dst + i, f);
    }

    // Handle any remaining 1–3 elements scalar‐wise
    for (int i = vec_count; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }

    env->ReleasePrimitiveArrayCritical(fp16Array, fp16_bytes, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(fp32Array, fp32_floats, 0);
}

void knn_jni::commons::convertFP32ToFP16(knn_jni::JNIUtilInterface *jniUtil,
                                         JNIEnv* env,
                                         jfloatArray fp32Array,
                                         jbyteArray fp16Array,
                                         jint count) {
    if (count <= 0) return;

    // Pin Java arrays
    jfloat* src_f32   = reinterpret_cast<jfloat*>(
        env->GetPrimitiveArrayCritical(fp32Array, nullptr));
    jbyte* dst_bytes = reinterpret_cast<jbyte*>(
        env->GetPrimitiveArrayCritical(fp16Array, nullptr));

    const float* src = reinterpret_cast<const float*>(src_f32);
    __fp16* dst = reinterpret_cast<__fp16*>(dst_bytes);

    int i = 0;
    // SIMD encode 4 floats at a time
    for (; i + 4 <= count; i += 4) {
        float32x4_t v_f32 = vld1q_f32(src + i);
        float16x4_t v_f16 = vcvt_f16_f32(v_f32);
        vst1_f16(dst + i, v_f16);  // store 4 half-words natively
    }
    // tail: 1–3 scalars
    for (; i < count; ++i) {
        float16x4_t tmp = vcvt_f16_f32(vdup_n_f32(src[i]));
        // extract lane 0
        __fp16 h = vget_lane_f16(tmp, 0);
        dst[i] = h;
    }

    env->ReleasePrimitiveArrayCritical(fp32Array, src_f32, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(fp16Array, dst_bytes, 0);
}
