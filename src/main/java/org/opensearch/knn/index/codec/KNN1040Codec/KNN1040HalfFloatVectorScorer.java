/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.jni.SimdFp16;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;

import java.io.IOException;

/**
 * Wraps a {@link FlatVectorsScorer} to give HNSW graph construction (flush's incremental build and
 * merge's graph rebuild) a decode-free scorer supplier for FP16 values, matching the decode-free path
 * search already gets via {@link KNN1040HalfFloatFlatVectorsValues#selectFallbackScorer}. Without
 * this, {@code getRandomVectorScorerSupplier} has no SIMD-aware path and decodes on every graph-edge
 * comparison instead of once per vector.
 *
 * <p>Delegates everything else unchanged, so it's safe to wrap any existing scorer chain - this class
 * only touches the one method, and only activates for our own {@link KNN1040HalfFloatFlatVectorsValues}.
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
        FloatVectorValues bottomValues = WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues);
        // MMapFloatVectorValues doesn't extend WrappedFloatVectorValues, so the unwrap above stops at
        // it rather than reaching the FP16 values it wraps - unwrap that one extra layer here.
        if (bottomValues instanceof MMapFloatVectorValues mmapValues) {
            bottomValues = mmapValues.getDelegate();
        }
        if (bottomValues instanceof KNN1040HalfFloatFlatVectorsValues halfFloatValues) {
            final SimdVectorComputeService.SimilarityFunctionType nativeType = NativeEngines990KnnVectorsScorer.getNativeFunctionType(
                similarityFunction
            );
            if (nativeType != null && SimdFp16.isSIMDSupported()) {
                return new HalfFloatRandomVectorScorerSupplier(halfFloatValues, nativeType);
            }
            // No native FP16 kernel for this similarity function (e.g. COSINE - see
            // NativeEngines990KnnVectorsScorer#getNativeFunctionType), or SIMD isn't available. Must
            // still never hand `halfFloatValues` to `delegate` below: that's Lucene's *optimized*
            // scorer chain, which detects HasIndexSlice and reads the raw slice assuming 4
            // bytes/dimension (float32), corrupting/crashing on this FP16 (2 bytes/dimension) data -
            // the same danger `KNN1040HalfFloatFlatVectorsValues#selectFallbackScorer` already avoids
            // for search. DefaultFlatVectorScorer is Lucene's plain, non-accelerated scorer: it only
            // ever calls FloatVectorValues#vectorValue(ord) - our own correct FP16 decode - and never
            // does any HasIndexSlice/memory-segment detection, so it's safe here even though the
            // values also implement HasIndexSlice for the (separate, guarded) mmap-native path above.
            return DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(similarityFunction, halfFloatValues);
        }
        return delegate.getRandomVectorScorerSupplier(similarityFunction, vectorValues);
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
     * Builds {@link UpdateableRandomVectorScorer}s for HNSW graph construction that read raw FP16
     * bytes directly for candidate comparisons via {@link HalfFloatRandomVectorScorer} - the
     * same decode-free path search already uses. The "current" graph node set via
     * {@link UpdateableRandomVectorScorer#setScoringOrdinal} is decoded once (via
     * {@link KNN1040HalfFloatFlatVectorsValues#vectorValue}) to build the native search context;
     * every subsequent candidate comparison against it stays fully byte-based - one decode per graph
     * node instead of one per graph edge, versus the generic Lucene fallback this replaces.
     */
    private static final class HalfFloatRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
        private final KNN1040HalfFloatFlatVectorsValues values;
        private final KNN1040HalfFloatFlatVectorsValues targetValues;
        private final SimdVectorComputeService.SimilarityFunctionType nativeType;

        HalfFloatRandomVectorScorerSupplier(KNN1040HalfFloatFlatVectorsValues values, SimdVectorComputeService.SimilarityFunctionType nativeType)
            throws IOException {
            this.values = values;
            this.targetValues = values.copy();
            this.nativeType = nativeType;
        }

        @Override
        public UpdateableRandomVectorScorer scorer() {
            return new UpdateableRandomVectorScorer.AbstractUpdateableRandomVectorScorer(values) {
                private HalfFloatRandomVectorScorer delegate;

                @Override
                public void setScoringOrdinal(int node) throws IOException {
                    delegate = new HalfFloatRandomVectorScorer(values, targetValues.vectorValue(node), nativeType);
                }

                @Override
                public float score(int node) throws IOException {
                    return delegate.score(node);
                }

                @Override
                public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
                    return delegate.bulkScore(nodes, scores, numNodes);
                }
            };
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return new HalfFloatRandomVectorScorerSupplier(values.copy(), nativeType);
        }
    }

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
    static class HalfFloatRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        private final KNN1040HalfFloatFlatVectorsValues values;
        private byte[] vectorBytesBuffer;
        private final float[] singleScoreBuffer = new float[1];
        private final int[] singleVectorId = new int[] { 0 };
        // Positional ids of the vectors packed into vectorBytesBuffer
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
}
