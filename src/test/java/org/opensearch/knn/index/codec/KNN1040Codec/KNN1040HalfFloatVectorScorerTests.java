/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies {@link KNN1040HalfFloatVectorScorer#getRandomVectorScorerSupplier} always uses
 * {@link DefaultFlatVectorScorer} - a native SIMD supplier was tried for merge and measured ~10x
 * slower (see class javadoc), so merge must never route to {@code delegate} regardless of similarity
 * function or SIMD availability. Also verifies the two {@code getRandomVectorScorer} overloads remain
 * untouched pass-throughs to {@code delegate} - that's search's mmap-tier path (see
 * {@link KNN1040HalfFloatFlatVectorsValues#selectScorer}), which must keep using the real native chain.
 */
public class KNN1040HalfFloatVectorScorerTests extends KNNTestCase {

    @SneakyThrows
    public void testGetRandomVectorScorerSupplier_alwaysUsesDefaultFlatVectorScorer() {
        assertAlwaysUsesDefaultTier(VectorSimilarityFunction.EUCLIDEAN);
        assertAlwaysUsesDefaultTier(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
        assertAlwaysUsesDefaultTier(VectorSimilarityFunction.COSINE);
        assertAlwaysUsesDefaultTier(VectorSimilarityFunction.DOT_PRODUCT);
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_float_delegatesUnchanged() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);

        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        final MMapFloatVectorValues mmapValues = new MMapFloatVectorValues(mockValues, new long[] { 1L, 2L });
        final float[] target = new float[] { 1f, 2f, 3f };
        final RandomVectorScorer expected = mock(RandomVectorScorer.class);
        when(mockDelegate.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, mmapValues, target)).thenReturn(expected);

        RandomVectorScorer actual = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, mmapValues, target);
        assertSame(expected, actual);
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_byte_delegatesUnchanged() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);

        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        final byte[] target = new byte[] { 1, 2, 3 };
        final RandomVectorScorer expected = mock(RandomVectorScorer.class);
        when(mockDelegate.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, mockValues, target)).thenReturn(expected);

        RandomVectorScorer actual = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, mockValues, target);
        assertSame(expected, actual);
    }

    @SneakyThrows
    private void assertAlwaysUsesDefaultTier(VectorSimilarityFunction similarityFunction) {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040HalfFloatVectorScorer scorer = new KNN1040HalfFloatVectorScorer(mockDelegate);

        final KNN1040HalfFloatFlatVectorsValues mockValues = mock(KNN1040HalfFloatFlatVectorsValues.class);
        // Only reached by DefaultFlatVectorScorer, which validates the encoding.
        when(mockValues.getEncoding()).thenReturn(VectorEncoding.FLOAT32);

        RandomVectorScorerSupplier result = scorer.getRandomVectorScorerSupplier(similarityFunction, mockValues);
        RandomVectorScorerSupplier expected = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(
            similarityFunction,
            mockValues
        );
        assertEquals(expected.getClass(), result.getClass());
        verify(mockDelegate, never()).getRandomVectorScorerSupplier(any(), any());
    }
}
