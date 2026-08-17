/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
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
 * Characterizes how many times each merge-time scorer tier actually crosses into native code
 * (JNI) while building a real HNSW graph, using Lucene's own {@link HnswGraphBuilder} directly
 * against real, valid FP16 data - not mocked or faked, so it's safe to let the real native SIMD
 * library run (unlike mocking {@code SimdVectorComputeService} itself, which crashed with a real
 * SIGSEGV in {@code KNN1040HalfFloatVectorScorerTests} when given fake addresses).
 *
 * <p>Every {@code score()}/{@code bulkScore()} call on our native tier's scorer is exactly one
 * JNI crossing (bulk batches all their candidates into a single native call). Every candidate
 * scored via {@link DefaultFlatVectorScorer}'s fallback triggers its own
 * {@code KNN1040HalfFloatFlatVectorsValues#vectorValue} call, which is its own native SIMD decode
 * crossing - no batching, since {@code DefaultFlatVectorScorer.FloatScoringSupplier} never
 * overrides {@code bulkScore} (confirmed by reading Lucene 10.5.0's actual source). This test
 * counts both directly, rather than reasoning about which should be lower.
 */
@Log4j2
public class KNN1040HalfFloatMergeCallCountTests extends KNNTestCase {

    private static final int NUM_VECTORS = 300;
    private static final int DIMENSION = 64;
    private static final int M = 16;
    private static final int BEAM_WIDTH = 100;

    @SneakyThrows
    public void testCallCounts_nativeTierVsDefaultFlatVectorScorer() {
        final Directory dir = new ByteBuffersDirectory();
        writeRandomHalfFloatVectors(dir, "vectors", NUM_VECTORS, DIMENSION);

        log.info("[fp16-merge-call-count] SimdFp16.isSIMDSupported()={}", org.opensearch.knn.jni.SimdFp16.isSIMDSupported());
        final CallCounts nativeCounts = buildGraphAndCountCalls(dir, true);
        final CallCounts fallbackCounts = buildGraphAndCountCalls(dir, false);

        log.info(
            "[fp16-merge-call-count] native tier: bulkScoreCalls={} totalCandidatesScored={} "
                + "setScoringOrdinalCalls={} decodeCalls={} avgMillis={} minMillis={}",
            nativeCounts.bulkScoreCalls.get(),
            nativeCounts.totalCandidatesScored.get(),
            nativeCounts.setScoringOrdinalCalls.get(),
            nativeCounts.decodeCalls.get(),
            nativeCounts.avgMillis,
            nativeCounts.minMillis
        );
        log.info(
            "[fp16-merge-call-count] DefaultFlatVectorScorer fallback: bulkScoreCalls={} totalCandidatesScored={} "
                + "setScoringOrdinalCalls={} decodeCalls={} avgMillis={} minMillis={}",
            fallbackCounts.bulkScoreCalls.get(),
            fallbackCounts.totalCandidatesScored.get(),
            fallbackCounts.setScoringOrdinalCalls.get(),
            fallbackCounts.decodeCalls.get(),
            fallbackCounts.avgMillis,
            fallbackCounts.minMillis
        );

        // Both tiers must score the same number of candidates overall - only how many native
        // crossings that took should differ.
        assertEquals(fallbackCounts.totalCandidatesScored.get(), nativeCounts.totalCandidatesScored.get());

        dir.close();
    }

    private static final int WARMUP_ITERATIONS = 1;
    private static final int TIMED_ITERATIONS = 3;

    @SneakyThrows
    private CallCounts buildGraphAndCountCalls(Directory dir, boolean useNativeTier) {
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
            final FlatVectorsScorer flatVectorsScorer = useNativeTier
                ? new KNN1040HalfFloatVectorScorer(DefaultFlatVectorScorer.INSTANCE)
                : DefaultFlatVectorScorer.INSTANCE;

            final RandomVectorScorerSupplier realSupplier = flatVectorsScorer.getRandomVectorScorerSupplier(
                VectorSimilarityFunction.EUCLIDEAN,
                values
            );
            final RandomVectorScorerSupplier countingSupplier = new CountingSupplier(realSupplier, counts, values);

            KNN1040HalfFloatFlatVectorsValues.DIAGNOSTIC_DECODE_CALLS.set(0);
            final long start = System.nanoTime();
            HnswGraphBuilder.create(countingSupplier, M, BEAM_WIDTH, 42).build(NUM_VECTORS);
            final long elapsed = System.nanoTime() - start;
            counts.decodeCalls.set(KNN1040HalfFloatFlatVectorsValues.DIAGNOSTIC_DECODE_CALLS.get());
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
        final AtomicLong decodeCalls = new AtomicLong();
        double avgMillis;
        double minMillis;
    }

    /**
     * Wraps a real {@link RandomVectorScorerSupplier}, counting scorer-method invocations without
     * mocking or altering any native call itself - every delegate call is real.
     *
     * <p>{@code score()}/{@code setScoringOrdinal()} calls double as decode-crossing counts: for
     * {@link DefaultFlatVectorScorer}, {@code score(node)} calls
     * {@code KNN1040HalfFloatFlatVectorsValues#vectorValue} on the candidate directly (one native
     * decode per candidate, confirmed by reading Lucene's actual source - it never overrides
     * {@code bulkScore}), and {@code setScoringOrdinal} decodes the "current" node the same way.
     * For our native tier, only {@code setScoringOrdinal} decodes anything (the "current" node);
     * candidates are read as raw bytes and never decoded in Java at all.
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
