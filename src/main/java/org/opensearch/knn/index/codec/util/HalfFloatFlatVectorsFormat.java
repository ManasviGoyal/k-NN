/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120BinaryVectorScorer;

/**
 * A FlatVectorsFormat wrapper that writes half-precision (FP16) vectors.
 */

/**
 * Custom FlatVectorsFormat implementation to support half-float vectors. This class is mostly identical to
 * {@link org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat}, however we use the custom {@link KNN9120BinaryVectorScorer}
 * to perform hamming bit scoring.
 */
public final class HalfFloatFlatVectorsFormat extends FlatVectorsFormat {

    static final String NAME = "HalfFloatFlatVectorsFormat";
    static final String META_CODEC_NAME = "Lucene99FlatVectorsFormatMeta";
    static final String VECTOR_DATA_CODEC_NAME = "Lucene99FlatVectorsFormatData";
    static final String META_EXTENSION = "vemf";
    static final String VECTOR_DATA_EXTENSION = "vec";

    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;

    static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
    private final FlatVectorsScorer vectorsScorer;

    public HalfFloatFlatVectorsFormat() {
        this(new DefaultFlatVectorScorer());
    }

    public HalfFloatFlatVectorsFormat(FlatVectorsScorer vectorsScorer) {
        super(NAME);
        this.vectorsScorer = vectorsScorer;
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new HalfFloatFlatVectorsWriter(state, vectorsScorer);
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new HalfFloatFlatVectorsReader(state, vectorsScorer);
    }

    @Override
    public String toString() {
        return NAME + "(vectorsScorer=" + vectorsScorer + ")";
    }
}
