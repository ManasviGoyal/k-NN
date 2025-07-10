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
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FF;
            f = sign | ((127 - 15 - exp + 1) << 23) | (mant << 13);  // Subnormal
        } else {
            f = sign;  // Zero
        }

        union { uint32_t u; float f; } u = { f };
        return u.f;
    };

    for (jsize i = 0; i < numElements; ++i) {
        // Java uses big-endian byte order for ByteBuffer
        uint16_t h = (static_cast<uint16_t>(inputBytes[2 * i]) << 8) |
                     static_cast<uint8_t>(inputBytes[2 * i + 1]);

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        // Convert to host endianness if needed
        h = (h >> 8) | (h << 8);
#endif

        outputFloats[i] = decodeFP16(h);
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
