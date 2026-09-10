/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;

import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040PerFieldKnnVectorsFormat.resolveFlatScalarEncoding;

/**
 * A compression level names a different SQ width for each data type, because levels are measured against
 * the pre-quantization width: FLOAT starts from 32 bits, half_float from 16. Getting this wrong is silent -
 * the level resolves, the mapping echoes it back, and search returns correct results against valid codes of
 * the wrong width - so it is pinned here rather than left to an integration test. Stored size is not a
 * usable substitute: _stats/store both lags mid-write and counts files pending deletion, so it reads low or
 * roughly double depending on timing.
 */
public class KNN1040FlatScalarEncodingTests extends KNNTestCase {

    private void assertDocBits(int expectedDocBits, CompressionLevel compressionLevel, VectorDataType vectorDataType) {
        assertEquals(
            vectorDataType.getValue() + " " + compressionLevel.getName() + " should store " + expectedDocBits + "-bit codes",
            expectedDocBits,
            ScalarEncodingResolver.docBits(resolveFlatScalarEncoding(compressionLevel, vectorDataType))
        );
    }

    // FLOAT: 32 bits down to 4, 2, 1.
    public void testFloatLevelsMapToWidthsAgainst32Bits() {
        assertDocBits(4, CompressionLevel.x8, VectorDataType.FLOAT);
        assertDocBits(2, CompressionLevel.x16, VectorDataType.FLOAT);
        assertDocBits(1, CompressionLevel.x32, VectorDataType.FLOAT);
    }

    // HALF_FLOAT: 16 bits down to 4, 2, 1 - every level one rung below FLOAT's for the same width.
    public void testHalfFloatLevelsMapToWidthsAgainst16Bits() {
        assertDocBits(4, CompressionLevel.x4, VectorDataType.HALF_FLOAT);
        assertDocBits(2, CompressionLevel.x8, VectorDataType.HALF_FLOAT);
        assertDocBits(1, CompressionLevel.x16, VectorDataType.HALF_FLOAT);
    }

    /**
     * The regression this guards against: reusing the FLOAT table for half_float. That maps x8 to 4-bit and
     * x16 to 2-bit, so half_float would store double the intended width at both levels.
     */
    public void testSameLevelMeansDifferentWidthPerDataType() {
        assertNotEquals(
            "x8 must not resolve to the same width for both data types",
            ScalarEncodingResolver.docBits(resolveFlatScalarEncoding(CompressionLevel.x8, VectorDataType.FLOAT)),
            ScalarEncodingResolver.docBits(resolveFlatScalarEncoding(CompressionLevel.x8, VectorDataType.HALF_FLOAT))
        );
        assertNotEquals(
            "x16 must not resolve to the same width for both data types",
            ScalarEncodingResolver.docBits(resolveFlatScalarEncoding(CompressionLevel.x16, VectorDataType.FLOAT)),
            ScalarEncodingResolver.docBits(resolveFlatScalarEncoding(CompressionLevel.x16, VectorDataType.HALF_FLOAT))
        );
    }

    /** x1 has no quantized width; callers resolve it to the raw FP16 format before reaching the helper. */
    public void testHalfFloatX1FallsBackToNarrowestWidth() {
        assertDocBits(1, CompressionLevel.x1, VectorDataType.HALF_FLOAT);
    }
}
