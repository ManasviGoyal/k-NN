/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;

import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Regression guard for merge-time HNSW graph construction over real, valid FP16 data, using Lucene's
 * own {@link HnswGraphBuilder} directly - no OpenSearch cluster needed, and safe to let the real
 * native SIMD decode run (unlike mocking {@code SimdVectorComputeService}, which crashed with a real
 * SIGSEGV in {@code KNN1040HalfFloatVectorScorerTests} when given fake addresses).
 *
 * <p>{@link KNN1040HalfFloatVectorScorer#getRandomVectorScorerSupplier} previously routed merge through
 * a native SIMD scorer supplier that was measured ~10x slower than {@link DefaultFlatVectorScorer} -
 * {@code saveSearchContext} rebuilt a whole Faiss distance computer on every graph node instead of
 * reusing one across candidates. The timing assertion here guards against that regressing back in.
 */
@Log4j2
public class KNN1040HalfFloatMergeCallCountTests extends KNNTestCase {

    private static final int NUM_VECTORS = 300;
    private static final int DIMENSION = 64;
    private static final int M = 16;
    private static final int BEAM_WIDTH = 100;
    // Generous headroom over the ~50ms this build actually takes, to absorb CI/JIT variance while
    // still catching a regression back to the native tier's ~400ms+.
    private static final double MAX_AVG_MILLIS = 200.0;

    @SneakyThrows
    public void testMergeGraphBuild_usesDefaultFlatVectorScorer_andStaysFast() {
        final Directory dir = new ByteBuffersDirectory();
        writeRandomHalfFloatVectors(dir, "vectors", NUM_VECTORS, DIMENSION);

        final CallCounts counts = buildGraphAndCountCalls(dir);

        log.info(
            "[fp16-merge-call-count] bulkScoreCalls={} totalCandidatesScored={} setScoringOrdinalCalls={} avgMillis={} minMillis={}",
            counts.bulkScoreCalls.get(),
            counts.totalCandidatesScored.get(),
            counts.setScoringOrdinalCalls.get(),
            counts.avgMillis,
            counts.minMillis
        );

        assertTrue("graph build should have scored candidates", counts.totalCandidatesScored.get() > 0);
        assertTrue("graph build should have set a scoring ordinal at least once", counts.setScoringOrdinalCalls.get() > 0);
        assertTrue(
            "merge graph build averaged " + counts.avgMillis + "ms, expected under " + MAX_AVG_MILLIS + "ms",
            counts.avgMillis < MAX_AVG_MILLIS
        );

        dir.close();
    }

    private static final int WARMUP_ITERATIONS = 1;
    private static final int TIMED_ITERATIONS = 3;

    @SneakyThrows
    private CallCounts buildGraphAndCountCalls(Directory dir) {
        CallCounts counts = null;
        final long[] elapsedNanosPerRun = new long[TIMED_ITERATIONS];

        for (int iteration = 0; iteration < WARMUP_ITERATIONS + TIMED_ITERATIONS; iteration++) {
            final IndexInput slice = dir.openInput("vectors", IOContext.DEFAULT);
            final KNN1040HalfFloatFlatVectorsValues values = new KNN1040HalfFloatFlatVectorsValues(
                DIMENSION,
                NUM_VECTORS,
                slice,
                null,
                DefaultFlatVectorScorer.INSTANCE,
                VectorSimilarityFunction.EUCLIDEAN
            );

            counts = new CallCounts();
            final KNN1040HalfFloatVectorScorer flatVectorsScorer = new KNN1040HalfFloatVectorScorer(DefaultFlatVectorScorer.INSTANCE);

            final RandomVectorScorerSupplier realSupplier = flatVectorsScorer.getRandomVectorScorerSupplier(
                VectorSimilarityFunction.EUCLIDEAN,
                values
            );
            final RandomVectorScorerSupplier countingSupplier = new CountingSupplier(realSupplier, counts, values);

            final long start = System.nanoTime();
            HnswGraphBuilder.create(countingSupplier, M, BEAM_WIDTH, 42).build(NUM_VECTORS);
            final long elapsed = System.nanoTime() - start;
            slice.close();

            if (iteration >= WARMUP_ITERATIONS) {
                elapsedNanosPerRun[iteration - WARMUP_ITERATIONS] = elapsed;
            }
        }

        long totalNanos = 0;
        long minNanos = Long.MAX_VALUE;
        for (long nanos : elapsedNanosPerRun) {
            totalNanos += nanos;
            minNanos = Math.min(minNanos, nanos);
        }
        counts.avgMillis = (totalNanos / (double) TIMED_ITERATIONS) / 1_000_000.0;
        counts.minMillis = minNanos / 1_000_000.0;
        return counts;
    }

    private static void writeRandomHalfFloatVectors(Directory dir, String name, int numVectors, int dimension) throws Exception {
        final Random random = new Random(1234);
        final IndexOutput out = dir.createOutput(name, IOContext.DEFAULT);
        final byte[] buffer = new byte[dimension * 2];
        for (int i = 0; i < numVectors; i++) {
            final float[] vector = new float[dimension];
            for (int d = 0; d < dimension; d++) {
                vector[d] = random.nextFloat() * 2 - 1;
            }
            KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, buffer, dimension);
            out.writeBytes(buffer, buffer.length);
        }
        out.close();
    }

    private static final class CallCounts {
        final AtomicLong bulkScoreCalls = new AtomicLong();
        final AtomicLong totalCandidatesScored = new AtomicLong();
        final AtomicLong setScoringOrdinalCalls = new AtomicLong();
        double avgMillis;
        double minMillis;
    }

    /**
     * Wraps the real {@link RandomVectorScorerSupplier}, counting scorer-method invocations without
     * mocking or altering any call itself.
     */
    private static final class CountingSupplier implements RandomVectorScorerSupplier {
        private final RandomVectorScorerSupplier delegate;
        private final CallCounts counts;
        private final KNN1040HalfFloatFlatVectorsValues values;

        CountingSupplier(RandomVectorScorerSupplier delegate, CallCounts counts, KNN1040HalfFloatFlatVectorsValues values) {
            this.delegate = delegate;
            this.counts = counts;
            this.values = values;
        }

        @Override
        public UpdateableRandomVectorScorer scorer() throws java.io.IOException {
            return new CountingScorer(delegate.scorer(), counts, values);
        }

        @Override
        public RandomVectorScorerSupplier copy() throws java.io.IOException {
            return new CountingSupplier(delegate.copy(), counts, values);
        }
    }

    private static final class CountingScorer extends UpdateableRandomVectorScorer.AbstractUpdateableRandomVectorScorer {
        private final UpdateableRandomVectorScorer delegate;
        private final CallCounts counts;

        CountingScorer(UpdateableRandomVectorScorer delegate, CallCounts counts, KNN1040HalfFloatFlatVectorsValues values) {
            super(values);
            this.delegate = delegate;
            this.counts = counts;
        }

        @Override
        public void setScoringOrdinal(int node) throws java.io.IOException {
            counts.setScoringOrdinalCalls.incrementAndGet();
            delegate.setScoringOrdinal(node);
        }

        @Override
        public float score(int node) throws java.io.IOException {
            counts.bulkScoreCalls.incrementAndGet();
            counts.totalCandidatesScored.incrementAndGet();
            return delegate.score(node);
        }

        @Override
        public float bulkScore(int[] nodes, float[] scores, int numNodes) throws java.io.IOException {
            counts.bulkScoreCalls.incrementAndGet();
            counts.totalCandidatesScored.addAndGet(numNodes);
            return delegate.bulkScore(nodes, scores, numNodes);
        }
    }
}
