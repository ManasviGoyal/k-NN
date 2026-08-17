/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.io.IOException;

/**
 * Wraps a {@link FlatVectorsScorer} so HNSW graph construction (flush's incremental build and merge's
 * graph rebuild) always gets {@link DefaultFlatVectorScorer}, regardless of what {@code delegate} would
 * otherwise pick. Merge in particular must never see {@code delegate} directly with our own
 * {@link KNN1040HalfFloatFlatVectorsValues}: that's Lucene's *optimized* scorer chain, which detects
 * {@code HasIndexSlice} and reads the raw slice assuming 4 bytes/dimension (float32), corrupting/crashing
 * on this FP16 (2 bytes/dimension) data - the same danger
 * {@link KNN1040HalfFloatFlatVectorsValues#selectFallbackScorer} already avoids for search.
 * {@code DefaultFlatVectorScorer} only ever calls {@code FloatVectorValues#vectorValue(ord)} - our own
 * correct FP16 decode - and never does {@code HasIndexSlice}/memory-segment detection, so it's always
 * safe here.
 *
 * <p>A native SIMD scorer supplier was tried for merge and measured ~10x slower than this plain decode
 * path: {@code saveSearchContext} rebuilds a whole Faiss distance computer on every
 * {@code setScoringOrdinal} call (once per graph node), which dominates the fewer JNI crossings it saves.
 *
 * <p>{@link #getRandomVectorScorer} overloads are untouched pass-throughs to {@code delegate} - that's
 * search's mmap-tier path (see {@link KNN1040HalfFloatFlatVectorsValues#selectScorer}), which always
 * calls this with {@link org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues}, not our own
 * FP16 values, so the crash risk above doesn't apply there.
 */
public class KNN1040HalfFloatVectorScorer implements FlatVectorsScorer {
    private final FlatVectorsScorer delegate;

    public KNN1040HalfFloatVectorScorer(FlatVectorsScorer delegate) {
        this.delegate = delegate;
    }

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues
    ) throws IOException {
        return DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(similarityFunction, vectorValues);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        return delegate.getRandomVectorScorer(similarityFunction, vectorValues, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        byte[] target
    ) throws IOException {
        return delegate.getRandomVectorScorer(similarityFunction, vectorValues, target);
    }

    @Override
    public String toString() {
        return "KNN1040HalfFloatVectorScorer(delegate=" + delegate + ")";
    }

    /**
     * Scores FP16 vectors via native SIMD, reading raw FP16 bytes directly from the segment's
     * {@link org.apache.lucene.store.IndexInput} slice - the decode-free path search's
     * {@link KNN1040HalfFloatFlatVectorsValues#selectFallbackScorer} uses for its native-no-mmap tier.
     */
    static class HalfFloatRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        private final KNN1040HalfFloatFlatVectorsValues values;
        private byte[] vectorBytesBuffer;
        private final float[] singleScoreBuffer = new float[1];
        private final int[] singleVectorId = new int[] { 0 };
        private int[] identityIds = new int[0];

        HalfFloatRandomVectorScorer(
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
            SimdVectorComputeService.scoreSimilarityInBulkFromFp16Bytes(vectorBytesBuffer, 1, singleVectorId, singleScoreBuffer);
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
            return SimdVectorComputeService.scoreSimilarityInBulkFromFp16Bytes(vectorBytesBuffer, numNodes, identityIds, scores);
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
}
