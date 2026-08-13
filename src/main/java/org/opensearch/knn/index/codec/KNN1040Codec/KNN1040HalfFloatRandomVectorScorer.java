/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.io.IOException;

/**
 * Scores FP16 vectors via native SIMD, reading raw FP16 bytes from the segment's
 * non-mmap-backed {@link org.apache.lucene.store.IndexInput} slice.
 *
 * The search context (query buffer + similarity function) is saved once, in the constructor, for
 * the scorer's lifetime — same as the mmap-backed
 * {@link org.opensearch.knn.memoryoptsearch.faiss.NativeRandomVectorScorer}. The difference is
 * address stability: {@code NativeRandomVectorScorer} points the saved context at a stable mmap
 * address once and never touches it again, whereas the vector bytes here are only pinned for the
 * duration of a single native call, so {@link #score} and {@link #bulkScore} must repoint the saved
 * context at a fresh chunk every call via {@link SimdVectorComputeService#scoreSimilarityInBulkFromBytes}
 * — that call skips re-copying the query and re-selecting the similarity function, updating only the
 * vector chunk location before scoring.
 */
class KNN1040HalfFloatRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
    private final KNN1040HalfFloatFlatVectorsValues values;
    private byte[] vectorBytesBuffer;
    private final float[] singleScoreBuffer = new float[1];
    private final int[] singleVectorId = new int[] { 0 };
    // Positional ids of the vectors packed into vectorBytesBuffer
    private int[] identityIds = new int[0];

    KNN1040HalfFloatRandomVectorScorer(
        KNN1040HalfFloatFlatVectorsValues values,
        float[] target,
        SimdVectorComputeService.SimilarityFunctionType nativeFunctionType
    ) {
        super(values);
        this.values = values;
        this.vectorBytesBuffer = new byte[values.byteSize()];
        SimdVectorComputeService.saveSearchContext(target, new long[0], nativeFunctionType.ordinal());
    }

    @Override
    public float score(int node) throws IOException {
        values.readRawVectorBytes(node, vectorBytesBuffer, 0);
        SimdVectorComputeService.scoreSimilarityInBulkFromBytes(vectorBytesBuffer, 1, singleVectorId, singleScoreBuffer);
        return singleScoreBuffer[0];
    }

    @Override
    public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
        int byteSize = values.byteSize();
        int requiredBytes = numNodes * byteSize;
        if (vectorBytesBuffer.length < requiredBytes) {
            vectorBytesBuffer = new byte[requiredBytes];
        }
        for (int i = 0; i < numNodes; i++) {
            values.readRawVectorBytes(nodes[i], vectorBytesBuffer, i * byteSize);
        }
        growIdentityIds(numNodes);
        return SimdVectorComputeService.scoreSimilarityInBulkFromBytes(vectorBytesBuffer, numNodes, identityIds, scores);
    }

    private void growIdentityIds(int numNodes) {
        int previousLength = identityIds.length;
        if (previousLength >= numNodes) {
            return;
        }
        identityIds = new int[numNodes];
        for (int i = 0; i < numNodes; i++) {
            identityIds[i] = i;
        }
    }
}
