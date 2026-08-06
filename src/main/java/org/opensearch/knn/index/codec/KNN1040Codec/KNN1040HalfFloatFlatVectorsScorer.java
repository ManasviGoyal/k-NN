/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.io.IOException;

/**
 * Scores FP16 vectors via native SIMD, reading raw (undecoded) FP16 bytes from the segment's
 * non-mmap-backed {@link org.apache.lucene.store.IndexInput} slice.
 *
 * <p>The search context is saved once per native call. Unlike
 * {@link org.opensearch.knn.memoryoptsearch.faiss.NativeRandomVectorScorer}, which saves it once
 * for the scorer's lifetime against a stable mmap address, the pinned {@code byte[]} address here
 * is valid only within a single call, so the save cannot be hoisted. Prefer {@link #bulkScore},
 * which amortizes the setup across a batch; {@link #score(int)} pays it for one vector.
 */
class KNN1040HalfFloatFlatVectorsScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
    private final KNN1040HalfFloatFlatVectorsValues values;
    private final float[] target;
    private final int nativeFunctionTypeOrd;
    private byte[] vectorBytesBuffer;
    private final float[] singleScoreBuffer = new float[1];

    KNN1040HalfFloatFlatVectorsScorer(
        KNN1040HalfFloatFlatVectorsValues values,
        float[] target,
        SimdVectorComputeService.SimilarityFunctionType nativeFunctionType
    ) {
        super(values);
        this.values = values;
        this.target = target;
        this.nativeFunctionTypeOrd = nativeFunctionType.ordinal();
        this.vectorBytesBuffer = new byte[values.byteSize()];
    }

    @Override
    public float score(int node) throws IOException {
        values.readRawVectorBytes(node, vectorBytesBuffer, 0);
        SimdVectorComputeService.scoreSimilarityInBulkFromBytes(
            target,
            vectorBytesBuffer,
            values.dimension(),
            nativeFunctionTypeOrd,
            1,
            singleScoreBuffer
        );
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
        return SimdVectorComputeService.scoreSimilarityInBulkFromBytes(
            target,
            vectorBytesBuffer,
            values.dimension(),
            nativeFunctionTypeOrd,
            numNodes,
            scores
        );
    }
}
