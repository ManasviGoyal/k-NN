/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;

/**
 * FP16 variant of {@link NativeEngines990KnnVectorsFormat}, used for {@code half_float} fields
 * (flat encoder) and, as an internal storage optimization, for FLOAT fields quantized via
 * {@code sq, bits:16}.
 *
 * <p>Separate class on purpose: Lucene's {@code PerFieldKnnVectorsFormat} reconstructs formats by
 * name via a no-arg constructor when reading a segment back, independent of whatever live instance
 * wrote it. Sharing a name with the FLOAT variant would silently pick the FP32 delegate on every
 * read, corrupting the segment. See {@code KNN1040HnswHalfFloatScalarQuantizedVectorsFormat} for
 * the same pattern used elsewhere in this codebase. {@link NativeEngines990KnnVectorsFormat#getName()}
 * is overridden to return the simple class name, so this subclass gets a distinct SPI name for free.
 */
public class NativeEngines990HalfFloatKnnVectorsFormat extends NativeEngines990KnnVectorsFormat {

    public NativeEngines990HalfFloatKnnVectorsFormat() {
        super(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE, new NativeIndexBuildStrategyFactory(), true);
    }

    public NativeEngines990HalfFloatKnnVectorsFormat(
        int approximateThreshold,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(approximateThreshold, nativeIndexBuildStrategyFactory, true);
    }
}
