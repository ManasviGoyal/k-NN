/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;

/**
 * A {@link FlatVectorsFormat} wrapper for half-precision (FP16) float vectors that stores each dimension as
 * an IEEE 754 half-float (2 bytes) and uses SIMD-accelerated native scoring during search.
 *
 * <p>This format is designed for exact (brute-force) search over half_float vector fields. Vectors
 * are encoded from FP32 to FP16 at index time and stored in a {@code .vec} file. At search time,
 * the reader overrides {@code search()} to perform an exhaustive batched scan using the native SIMD
 * scorer, which widens FP16 data to FP32 in-register for asymmetric distance computation against
 * the FP32 query vector.
 *
 * <p>The scorer chain is:
 * <pre>
 *   PrefetchableFlatVectorScorer
 *     └─ NativeEngines990KnnVectorsScorer
 *          └─ NativeRandomVectorScorer (SIMD via JNI)
 * </pre>
 */
public class KNN1040HalfFloatFlatVectorsFormat extends FlatVectorsFormat {

    static final String NAME = "KNN1040HalfFloatFlatVectorsFormat";
    static final String META_CODEC_NAME = "KNN1040HalfFloatFlatVectorsFormatMeta";
    static final String VECTOR_DATA_CODEC_NAME = "KNN1040HalfFloatFlatVectorsFormatData";
    static final String META_EXTENSION = "vemf";
    static final String VECTOR_DATA_EXTENSION = "vec";
    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = VERSION_START;
    static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

    private static final FlatVectorsScorer KNN_1040_HALF_FLOAT_FLAT_VECTORS_SCORER = new PrefetchableFlatVectorScorer(
        new NativeEngines990KnnVectorsScorer(FlatVectorsScorerProvider.getLucene99FlatVectorsScorer())
    );

    public KNN1040HalfFloatFlatVectorsFormat() {
        super(NAME);
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new KNN1040HalfFloatFlatVectorsWriter(state, KNN_1040_HALF_FLOAT_FLAT_VECTORS_SCORER);
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new KNN1040HalfFloatFlatVectorsReader(state, KNN_1040_HALF_FLOAT_FLAT_VECTORS_SCORER);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return String.format("%s(scorer=%s)", getClass().getSimpleName(), KNN_1040_HALF_FLOAT_FLAT_VECTORS_SCORER);
    }
}
