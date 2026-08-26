/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;

/**
 * FP16 variant of {@link Faiss1040ScalarQuantizedKnnVectorsFormat}, used when a {@code sq, bits:1}
 * Faiss HNSW field is declared {@code half_float}.
 *
 * <p>Separate class on purpose: Lucene's {@code PerFieldKnnVectorsFormat} reconstructs formats by
 * name via a no-arg constructor when reading a segment back, independent of whatever live instance
 * wrote it. Sharing a name with the FLOAT variant would silently pick the FP32 delegate on every
 * read, corrupting the segment. See {@code KNN1040HnswHalfFloatScalarQuantizedVectorsFormat} for
 * the same pattern used elsewhere in this codebase.
 * {@link Faiss1040ScalarQuantizedKnnVectorsFormat#getName()} is overridden to return the simple
 * class name, so this subclass gets a distinct SPI name for free.
 */
public class Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat extends Faiss1040ScalarQuantizedKnnVectorsFormat {

    public Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat() {
        super(
            KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE,
            new NativeIndexBuildStrategyFactory(),
            VectorDataType.HALF_FLOAT
        );
    }

    public Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat(
        final int approximateThreshold,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(approximateThreshold, nativeIndexBuildStrategyFactory, VectorDataType.HALF_FLOAT);
    }
}
