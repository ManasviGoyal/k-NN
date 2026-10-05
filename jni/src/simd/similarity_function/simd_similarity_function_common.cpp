/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <iostream>

#include "parameter_utils.h"
#include "platform_defs.h"
#include "simd/similarity_function/similarity_function.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "simd/fp16_codec/fp16_codec.h"
#include "jni_util.h"

using knn_jni::simd::similarity_function::SimdVectorSearchContext;
using knn_jni::simd::similarity_function::SimilarityFunction;

// Since Windows CI is failing to recognize std::aligned_malloc, it will make CI pass with macro.
#if defined(_WIN32)
    #include <malloc.h>
#endif

inline void* allocate_aligned_memory(size_t alignment, size_t size) {
#if defined(_WIN32)
    // Works for MSVC and MinGW alike
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void free_aligned_memory(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

//
// SimdVectorSearchContext
//
void SimdVectorSearchContext::getVectorPointersInBulk(uint8_t* vectors[], int32_t* internalVectorIds, int32_t numVectors) {
    if (LIKELY(mmapPages.size() == 1)) {
        // Fast case, there's only one mmap area.
        auto base = mmapPages[0];
        for (int32_t i = 0 ; i < numVectors ; ++i) {
            const uint64_t offset = oneVectorByteSize * internalVectorIds[i];
            if (LIKELY(offset < mmapPageSizes[0])) {
                vectors[i] = reinterpret_cast<uint8_t*>(base) + offset;
            } else {
                throw std::runtime_error(std::string("Offset [") + std::to_string(offset)
                + "] exceeds the chunk size [" + std::to_string(mmapPageSizes[0]) + "].");
            }
        }
        return;
    }

    // There are multiple mapped regions.
    if (mmapPages.empty() == false) {
        // A vector that straddles two mmap regions is reassembled by getVectorPointer() into
        // tmpBuffer via tmpBuffer.resize(), which may reallocate the underlying storage. If two
        // (or more) vectors in this same batch straddle, the second resize() would reallocate and
        // dangle the pointer we already stored in vectors[] for the earlier straddling vector,
        // making the kernel read freed/moved memory. Pre-reserve the worst-case headroom for the
        // whole batch up front (every vector reassembled needs at most oneVectorByteSize bytes plus
        // one padding byte for even-address alignment) so no resize() inside the loop can reallocate,
        // keeping every pointer returned in this call stable. reserve() preserves existing contents,
        // so the query-correction prefix some kernels keep at the front of tmpBuffer is untouched.
        tmpBuffer.reserve(tmpBuffer.size() + static_cast<size_t>(numVectors) * (static_cast<size_t>(oneVectorByteSize) + 1));

        for (int32_t i = 0 ; i < numVectors ; ++i) {
            vectors[i] = getVectorPointer(internalVectorIds[i]);
        }
        return;
    }  // End if

    throw std::runtime_error("Search context has not been initialized, mmapPages was empty.");
}

uint8_t* SimdVectorSearchContext::getVectorPointer(const int32_t internalVectorId) {
    if (LIKELY(mmapPages.size() == 1)) {
        // Fast case, there's only one mmap area.
        return reinterpret_cast<uint8_t*>(mmapPages[0]) + (oneVectorByteSize * internalVectorId);
    }

    if (mmapPages.empty() == false) {
        // Acquire offsets
        const uint64_t startOffset = oneVectorByteSize * internalVectorId;
        const uint64_t endOffsetInclusive = startOffset + oneVectorByteSize - 1;

        // Find region having the vector.
        uint64_t regionStartOffset = 0;
        for (int32_t j = 0 ; j < mmapPageSizes.size() ; ++j) {
            // Note that mmapPageSizes[j] is the endOffset (exclusive) of a region.
            // Therefore, in turn, mmapPageSizes[j - 1] is the starting offset of mmapPageSizes[j] where
            // j > 0, if j == 0, 0 would be the start offset.
            if (startOffset < mmapPageSizes[j]) {
                // Found the first region having the vector.
                // At the worst case, one vector can span across two mapped regions.

                const uint64_t relativeOffsetInFirstRegion = (startOffset - regionStartOffset);

                if (endOffsetInclusive < mmapPageSizes[j]) {
                    // Nice! This region has the entire vector intact.
                    return reinterpret_cast<uint8_t*>(mmapPages[j]) + relativeOffsetInFirstRegion;
                } else {
                    // Prevent seg-fault, this should not happen but it's better to throw an exception than
                    // halting a process.
                    if (UNLIKELY((j + 1) >= mmapPageSizes.size() || (j + 1) >= mmapPages.size())) {
                        throw std::runtime_error(
                        std::string("One vector[vid=") + std::to_string(internalVectorId)
                        + "] straddle two regions(" + std::to_string(j) + "th and " + std::to_string(j + 1)
                        + "th), but there was no next region. We had " + std::to_string(mmapPageSizes.size()) + " regions.");
                    }

                    // No luck, one vector spans across two mapped regions.
                    // We need to copy vectors into a temp buffer having continuous space.
                    // Make sure the vector to have an even address.
                    const int32_t padding = tmpBuffer.size() & 1;
                    const int32_t copyDestIndex = tmpBuffer.size() + padding;
                    tmpBuffer.resize(tmpBuffer.size() + padding + oneVectorByteSize);

                    // Copy the first part
                    const int32_t firstPartSize = mmapPageSizes[j] - startOffset;
                    std::memcpy(&tmpBuffer[copyDestIndex],
                                reinterpret_cast<uint8_t*>(mmapPages[j]) + relativeOffsetInFirstRegion, firstPartSize);

                    // Copy the second part
                    const int32_t secondPartSize = oneVectorByteSize - firstPartSize;
                    const uint64_t nextRegionSize = mmapPageSizes[j + 1] - mmapPageSizes[j];
                    if (UNLIKELY(secondPartSize > nextRegionSize)) {
                        throw std::runtime_error(
                            std::string("One vector[vid=") + std::to_string(internalVectorId)
                            + "] straddle two regions(" + std::to_string(j) + "th and " + std::to_string(j + 1)
                            + "th), but the second part of the vector size=" + std::to_string(secondPartSize)
                            + " exceeds the second region size=" + std::to_string(nextRegionSize));
                    }
                    std::memcpy(&tmpBuffer[copyDestIndex + firstPartSize],
                                reinterpret_cast<uint8_t*>(mmapPages[j + 1]), secondPartSize);

                    // Set the pointer pointing temp buffer.
                    return &tmpBuffer[copyDestIndex];
                }  // End if
            }  // End if

            // mmapPageSizes[j] is the starting offset of mmapPageSizes[j + 1]
            regionStartOffset = mmapPageSizes[j];
        }  // End for

        // Should not happen, region must be found
        std::string errorMsg = std::string("Mapped region for vector(vid=") + std::to_string(internalVectorId) + ") was not found. ";
        errorMsg += "#mmapPageSizes=" + std::to_string(mmapPageSizes.size()) + ", [";
        for (auto pageSize : mmapPageSizes) {
            errorMsg += std::to_string(pageSize) + ", ";
        }
        errorMsg += "], #mmapPages=" + std::to_string(mmapPages.size()) + ", [";
        for (auto pagePtr : mmapPages) {
            errorMsg += std::to_string(reinterpret_cast<uint64_t>(pagePtr)) + ", ";
        }
        errorMsg += "]";
        throw std::runtime_error(std::move(errorMsg));
    }  // End if

    throw std::runtime_error("Search context has not been initialized, mmapPages was empty.");
}

SimdVectorSearchContext::~SimdVectorSearchContext() {
    if (queryVectorSimdAligned) {
        free_aligned_memory(queryVectorSimdAligned);
    }
}

// Thread static local SimdVectorSearchContext
thread_local SimdVectorSearchContext THREAD_LOCAL_SIMD_VEC_SRCH_CTX {};



//
// SimilarityFunction
//
SimdVectorSearchContext* SimilarityFunction::saveSearchContext(
           uint8_t* queryPtr,
           int32_t queryByteSize,
           int32_t dimension,
           int64_t* mmapAddressAndSize,
           int32_t numAddressAndSize,
           int32_t nativeFunctionTypeOrd) {
    // Free tmp buffer
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.tmpBuffer = {};

    // Allocate query vector space
    if (THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorByteSize < queryByteSize) {
        // We need to allocate or re-allocate the space.
        // Allocating 64 bytes aligned memory.
        // Since 16000 dimension is the maximum, therefore at most 62.6KB will be allocated per thread.
        const auto roundedUpQueryByteSize = ((queryByteSize + 63) / 64) * 64;
        void* alignedPtr = allocate_aligned_memory(64, roundedUpQueryByteSize);
        if (alignedPtr == nullptr) {
            throw std::runtime_error(
            std::string("Failed to allocate space for SIMD aligned query vector with size=")
            + std::to_string(queryByteSize));
        }

        // Free up previously allocated space
        if (THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned) {
            free_aligned_memory(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned);
        }

        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned = alignedPtr;
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorByteSize = queryByteSize;
    }

    // Copy query bytes. A null queryPtr means the caller fills the buffer itself after this returns
    // (see saveSearchContextFromOrdinal); set_query() below only records the buffer's address, so
    // filling it later is equivalent.
    if (queryPtr != nullptr) {
        std::memcpy(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned, queryPtr, queryByteSize);
    }

    // FP16 and BF16 share the same setup: they are 2-byte-per-component quantized
    // formats whose per-vector similarity is offloaded to a Faiss SQDistanceComputer.
    // Only the similarity function, the Faiss quantizer type and the metric differ.
    auto setupFaissQuantizedFunction = [&](NativeSimilarityFunctionType functionType,
                                           faiss::ScalarQuantizer::QuantizerType quantizerType,
                                           faiss::MetricType metric) {
        // Set similarity function to offload similarity calculation
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.similarityFunction = selectSimilarityFunction(functionType);

        // FP16/BF16 vector bytes = 2bytes * dimension
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.oneVectorByteSize = 2 * dimension;

        // The distance computer depends only on (dimension, quantizer type, metric), and those are
        // fixed for a field, so rebuild it only when this thread last served a different one. Search
        // saves the context once per query and would not care, but HNSW graph build re-saves it once
        // per graph node - rebuilding there costs a heap allocation plus a free per node, which at
        // millions of nodes dominates the merge. Only the query below actually changes per call.
        // `dimension` and `nativeFunctionTypeOrd` on the context are still the previous call's values
        // here; they are assigned after this lambda returns.
        const bool reusable = THREAD_LOCAL_SIMD_VEC_SRCH_CTX.faissFunction != nullptr
            && THREAD_LOCAL_SIMD_VEC_SRCH_CTX.dimension == dimension
            && THREAD_LOCAL_SIMD_VEC_SRCH_CTX.nativeFunctionTypeOrd == static_cast<int32_t>(functionType);
        if (reusable == false) {
            THREAD_LOCAL_SIMD_VEC_SRCH_CTX.faissFunction.reset(
                faiss::ScalarQuantizer {static_cast<size_t>(dimension), quantizerType}
                                       .get_distance_computer(metric));
        }

        // Always re-assign: the query contents change on every call, and queryVectorSimdAligned
        // itself moves whenever it is grown for a larger dimension.
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.faissFunction->set_query(
            reinterpret_cast<float*>(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned));
    };

    // Set similarity function
    if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT)) {
        setupFaissQuantizedFunction(NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT,
                                    faiss::ScalarQuantizer::QuantizerType::QT_fp16,
                                    faiss::MetricType::METRIC_INNER_PRODUCT);
    } else if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::FP16_L2)) {
        setupFaissQuantizedFunction(NativeSimilarityFunctionType::FP16_L2,
                                    faiss::ScalarQuantizer::QuantizerType::QT_fp16,
                                    faiss::MetricType::METRIC_L2);
    } else if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::FP16_COSINE)) {
        // FP16_COSINE shares the on-disk FP16 layout with FP16_MAX_INNER_PRODUCT. Vectors are
        // L2-normalized at index time, so cosine equals the inner product, and the kernel emits
        // a (1 + dot) / 2 Lucene score directly via cosineTransform. On platforms with native
        // AVX-512-FP16 support FP16_COSINE binds to AVX512NativeFP16IP (true FP16 FMA, 32
        // elements per 512-bit register); elsewhere it falls back to the FP32-converted IP
        // kernel used by FP16_MAX_INNER_PRODUCT. The single-vector calculateSimilarity path
        // goes through BaseSimilarityFunction, which calls the Faiss IP distance computer
        // (set up here with METRIC_INNER_PRODUCT) and applies cosineTransform on top of the raw
        // IP result.
        setupFaissQuantizedFunction(NativeSimilarityFunctionType::FP16_COSINE,
                                    faiss::ScalarQuantizer::QuantizerType::QT_fp16,
                                    faiss::MetricType::METRIC_INNER_PRODUCT);
    } else if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::BF16_MAXIMUM_INNER_PRODUCT)) {
        setupFaissQuantizedFunction(NativeSimilarityFunctionType::BF16_MAXIMUM_INNER_PRODUCT,
                                    faiss::ScalarQuantizer::QuantizerType::QT_bf16,
                                    faiss::MetricType::METRIC_INNER_PRODUCT);
    } else if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::BF16_L2)) {
        setupFaissQuantizedFunction(NativeSimilarityFunctionType::BF16_L2,
                                    faiss::ScalarQuantizer::QuantizerType::QT_bf16,
                                    faiss::MetricType::METRIC_L2);
    } else if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::SQ_IP)
               || nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::SQ_L2)
               || nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::SQ_COSINE)) {
         // SQ_COSINE shares on-disk layout and IP math with SQ_IP; only the score-transform kernel differs.
         // Set similarity function to offload similarity calculation
         THREAD_LOCAL_SIMD_VEC_SRCH_CTX.similarityFunction =
             selectSimilarityFunction(static_cast<NativeSimilarityFunctionType>(nativeFunctionTypeOrd));

         // oneVectorByteSize = quantized vector bytes + 3 floats + 1 int (correction factors)
         // On-disk layout: [binaryCode | lowerInterval(float) | upperInterval(float) | additionalCorrection(float) | quantizedComponentSum(int)]
         // Lucene's docPackedLength for SINGLE_BIT_QUERY_NIBBLE: (dim + 7) / 8
         THREAD_LOCAL_SIMD_VEC_SRCH_CTX.oneVectorByteSize =
             (dimension + 7) / 8 + 3 * sizeof(float) + sizeof(int32_t);
    } else {
        throw std::runtime_error(
            std::string("Invalid native similarity function type was given, nativeFunctionTypeOrd=")
            + std::to_string(nativeFunctionTypeOrd));
    }

    // Assign native function ord number
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.nativeFunctionTypeOrd = nativeFunctionTypeOrd;

    // Set dimension
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.dimension = dimension;

    // Set mmap pages
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.clear();
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.clear();
    for (int32_t i = 0 ; i < numAddressAndSize ; i += 2) {
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.emplace_back(reinterpret_cast<void*>(mmapAddressAndSize[i]));
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.emplace_back(mmapAddressAndSize[i + 1]);
    }

    // Build prefix sum table. This table will be used to locate the mapped page with a logical offset.
    // For example, let's say the size list was [100, 100, 100] meaning each mmap page had 100 bytes.
    // Then the resulting prefix sum table would be [100, 200, 300]. Then, we can identify the second mmap page has
    // a vector whose start offset is 150.
    for (int32_t i = 1 ; i < THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.size() ; ++i) {
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes[i] += THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes[i - 1];
    }

    // Return thread_local object
    return &THREAD_LOCAL_SIMD_VEC_SRCH_CTX;
}

namespace {

using knn_jni::simd::similarity_function::NativeSimilarityFunctionType;

// Only FP16 widens two bytes to a float this way; BF16 and SQ store different layouts and have to go
// through the float[] entry point.
void requireFp16FunctionType(const int32_t nativeFunctionTypeOrd, const char* caller) {
    if (nativeFunctionTypeOrd != static_cast<int32_t>(NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT)
        && nativeFunctionTypeOrd != static_cast<int32_t>(NativeSimilarityFunctionType::FP16_L2)
        && nativeFunctionTypeOrd != static_cast<int32_t>(NativeSimilarityFunctionType::FP16_COSINE)) {
        throw std::runtime_error(
            std::string(caller) + " supports FP16 only, nativeFunctionTypeOrd="
            + std::to_string(nativeFunctionTypeOrd));
    }
}

}  // namespace

SimdVectorSearchContext* SimilarityFunction::saveSearchContextFromOrdinal(
           const int32_t internalVectorId,
           const int32_t dimension,
           int64_t* mmapAddressAndSize,
           const int32_t numAddressAndSize,
           const int32_t nativeFunctionTypeOrd) {
    requireFp16FunctionType(nativeFunctionTypeOrd, "saveSearchContextFromOrdinal");

    // Configure everything but the query contents. This also leaves mmapPages and the prefix sum
    // table built, which getVectorPointer() below needs in order to locate the target.
    saveSearchContext(nullptr,
                      dimension * static_cast<int32_t>(sizeof(float)),
                      dimension,
                      mmapAddressAndSize,
                      numAddressAndSize,
                      nativeFunctionTypeOrd);

    // The target is itself a stored vector, so widen its FP16 bytes straight out of the mapped
    // region. HNSW graph build switches target once per graph node; having Java decode to a float[]
    // and ship that across JNI instead costs a decode plus a full query-sized copy every time.
    const uint8_t* vectorPtr = THREAD_LOCAL_SIMD_VEC_SRCH_CTX.getVectorPointer(internalVectorId);
    knn_jni::simd::fp16_codec::decodeFp16ToFp32Raw(
        reinterpret_cast<const uint16_t*>(vectorPtr),
        reinterpret_cast<float*>(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned),
        static_cast<size_t>(dimension));

    return &THREAD_LOCAL_SIMD_VEC_SRCH_CTX;
}

SimdVectorSearchContext* SimilarityFunction::saveSearchContextFromFp16Bytes(
           const uint8_t* fp16Target,
           const int32_t dimension,
           const int32_t nativeFunctionTypeOrd) {
    requireFp16FunctionType(nativeFunctionTypeOrd, "saveSearchContextFromFp16Bytes");

    // No mapped region here, matching what the heap-buffer scoring path passes; the vector chunk is
    // repointed separately by updateVectorChunk when the candidates are scored.
    saveSearchContext(nullptr,
                      dimension * static_cast<int32_t>(sizeof(float)),
                      dimension,
                      nullptr,
                      0,
                      nativeFunctionTypeOrd);

    // Widen the caller's FP16 bytes rather than having Java decode them: half the bytes cross the
    // boundary and the conversion runs on the SIMD path instead of in Java.
    knn_jni::simd::fp16_codec::decodeFp16ToFp32Raw(
        reinterpret_cast<const uint16_t*>(fp16Target),
        reinterpret_cast<float*>(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned),
        static_cast<size_t>(dimension));

    return &THREAD_LOCAL_SIMD_VEC_SRCH_CTX;
}

SimdVectorSearchContext* SimilarityFunction::getSearchContext() {
    return &THREAD_LOCAL_SIMD_VEC_SRCH_CTX;
}

void SimilarityFunction::updateVectorChunk(uint8_t* vectorsPtr, const int64_t vectorsByteSize) {
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.clear();
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.emplace_back(reinterpret_cast<void*>(vectorsPtr));
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.clear();
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.emplace_back(vectorsByteSize);
}

//
// Similarity function base
//
using BulkScoreTransform = void (*)(float*/*scores*/, int32_t/*num scores to transform*/);
using ScoreTransform = float (*)(float/*score*/);

// Metric mode for SQ (Scalar Quantized) similarity functions.
// Controls both intermediate score computation and final score transformation.
enum class SQMetricMode {
    MAX_IP,   // Inner product: score += correction; ipToMaxIpTransform
    L2,       // L2 distance: score = max(0, correction - 2*score); l2Transform
    COSINE    // Cosine (normalized IP): score += correction; cosineTransform
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct BaseSimilarityFunction : SimilarityFunction {
    // Generic single-vector scoring through the Faiss distance computer. Not final: a SIMD variant
    // that has its own kernel for the format overrides this so single-vector scoring does not fall
    // back to Faiss, whose SIMD level depends on how Faiss itself happened to be compiled.
    float calculateSimilarity(SimdVectorSearchContext* srchContext, const int32_t internalVectorId) override {
        // Prepare distance calculation
        auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorId));
        knn_jni::util::ParameterCheck::require_non_null(vector, "vector from getVectorPointer");
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        // Calculate distance
        const float score = func->query_to_code(vector);

        // Transform score value if it needs to
        return ScoreTransformFunc(score);
    }
};
