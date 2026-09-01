/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.index.engine.KNNEngine;

import java.nio.file.Files;
import java.util.Locale;

import static org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

public class KNN1040ScalarQuantizedVectorsFormatTests extends KNNTestCase {

    public void testToString_containsEncodingAndScorerInfo() {
        KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        String str = format.toString();
        assertTrue("toString should contain class name", str.contains("KNN1040ScalarQuantizedVectorsFormat"));
        assertTrue("toString should contain encoding", str.contains(SINGLE_BIT_QUERY_NIBBLE.name()));
        assertTrue("toString should contain scorer", str.contains("ScalarQuantizedVectorScorer"));
    }

    public void testDefaultConstructor_usesSingleBitQueryNibble() {
        KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        assertTrue(format.toString().contains(SINGLE_BIT_QUERY_NIBBLE.name()));
    }

    public void testGetMaxDimensions_returnsLuceneMax() {
        KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetName_returnsClassName() {
        KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        assertEquals("KNN1040ScalarQuantizedVectorsFormat", format.getName());
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_returnsPrefetchableScorer() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);

        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            SegmentReadState readState = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());

            try (FlatVectorsReader reader = format.fieldsReader(readState)) {
                RandomVectorScorer scorer = reader.getRandomVectorScorer(
                    KNN1040ScalarQuantizedTestUtils.FIELD_NAME,
                    KNN1040ScalarQuantizedTestUtils.randomVector(KNN1040ScalarQuantizedTestUtils.DIMENSION, random())
                );

                assertNotNull("RandomVectorScorer should not be null", scorer);
                assertTrue(
                    "Scorer should be PrefetchableRandomVectorScorer (backed by KNN1040ScalarQuantizedVectorScorer), "
                        + "but was: "
                        + scorer.getClass().getSimpleName(),
                    scorer instanceof PrefetchableRandomVectorScorer
                );
                assertEquals("maxOrd should match number of vectors", KNN1040ScalarQuantizedTestUtils.NUM_VECTORS, scorer.maxOrd());
            }
        }
    }

    public void testConstructor_allEncodings() {
        for (ScalarEncoding encoding : ScalarEncoding.values()) {
            KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat(encoding);
            assertNotNull(format);
            assertTrue(format.toString().contains(encoding.name()));
        }
    }

    public void testConstructor_withHalfFloatRawVectorDataType_doesNotThrow() {
        KNN1040HalfFloatScalarQuantizedVectorsFormat format = new KNN1040HalfFloatScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        assertNotNull(format);
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetName_floatAndHalfFloatVariantsAreDistinct() {
        KNN1040ScalarQuantizedVectorsFormat floatFormat = new KNN1040ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        KNN1040HalfFloatScalarQuantizedVectorsFormat halfFloatFormat = new KNN1040HalfFloatScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE
        );
        assertNotEquals(
            "FLOAT and HALF_FLOAT SQ 1-bit formats must have different names or read-time SPI " + "reconstruction cannot tell them apart",
            floatFormat.getName(),
            halfFloatFormat.getName()
        );
    }

    @SneakyThrows
    public void testSpiReconstruction_withHalfFloatRawDelegate_readsBackCorrectly() {
        final KNN1040HalfFloatScalarQuantizedVectorsFormat writeFormat = new KNN1040HalfFloatScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE
        );

        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            SegmentReadState readState = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, writeFormat, random());

            KnnVectorsFormat readFormat = KnnVectorsFormat.forName(writeFormat.getName());
            assertTrue(
                "forName(" + writeFormat.getName() + ") should resolve to the half_float SQ format",
                readFormat instanceof KNN1040HalfFloatScalarQuantizedVectorsFormat
            );

            try (FlatVectorsReader reader = ((KNN1040HalfFloatScalarQuantizedVectorsFormat) readFormat).fieldsReader(readState)) {
                RandomVectorScorer scorer = reader.getRandomVectorScorer(
                    KNN1040ScalarQuantizedTestUtils.FIELD_NAME,
                    KNN1040ScalarQuantizedTestUtils.randomVector(KNN1040ScalarQuantizedTestUtils.DIMENSION, random())
                );
                assertNotNull("RandomVectorScorer should not be null after SPI-based reconstruction", scorer);
                assertEquals(KNN1040ScalarQuantizedTestUtils.NUM_VECTORS, scorer.maxOrd());
                for (int ord = 0; ord < scorer.maxOrd(); ord++) {
                    assertFalse("score should not be NaN for ord " + ord, Float.isNaN(scorer.score(ord)));
                }
            }
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_withHalfFloatRawDelegate_returnsPrefetchableScorer() {
        final KNN1040HalfFloatScalarQuantizedVectorsFormat format = new KNN1040HalfFloatScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE
        );

        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            SegmentReadState readState = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());

            try (FlatVectorsReader reader = format.fieldsReader(readState)) {
                RandomVectorScorer scorer = reader.getRandomVectorScorer(
                    KNN1040ScalarQuantizedTestUtils.FIELD_NAME,
                    KNN1040ScalarQuantizedTestUtils.randomVector(KNN1040ScalarQuantizedTestUtils.DIMENSION, random())
                );

                assertNotNull("RandomVectorScorer should not be null with a half_float raw delegate", scorer);
                assertTrue(
                    "Scorer should be PrefetchableRandomVectorScorer (backed by KNN1040ScalarQuantizedVectorScorer), "
                        + "but was: "
                        + scorer.getClass().getSimpleName(),
                    scorer instanceof PrefetchableRandomVectorScorer
                );
                assertEquals("maxOrd should match number of vectors", KNN1040ScalarQuantizedTestUtils.NUM_VECTORS, scorer.maxOrd());

                for (int ord = 0; ord < scorer.maxOrd(); ord++) {
                    float score = scorer.score(ord);
                    assertFalse("score should not be NaN for ord " + ord, Float.isNaN(score));
                }
            }
        }
    }

    @SneakyThrows
    public void testVecFile_withHalfFloatRawDelegate_isSmallerThanFloatRawDelegate() {
        final long floatVecBytes = writeAndGetVecFileSize(new KNN1040ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE));
        final long halfFloatVecBytes = writeAndGetVecFileSize(new KNN1040HalfFloatScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE));

        assertTrue(
            String.format(
                Locale.ROOT,
                "half_float .vec (%d bytes) should be smaller than FLOAT .vec (%d bytes)",
                halfFloatVecBytes,
                floatVecBytes
            ),
            halfFloatVecBytes < floatVecBytes
        );

        assertTrue(
            "half_float .vec should be roughly half the FLOAT .vec size, not just smaller",
            halfFloatVecBytes < (floatVecBytes / 2) + 4096
        );
    }

    @SneakyThrows
    private long writeAndGetVecFileSize(KNN1040ScalarQuantizedVectorsFormat format) {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            long total = 0L;
            for (String file : dir.getDirectory().toFile().list()) {
                if (file.endsWith(".vec")) {
                    total += Files.size(dir.getDirectory().resolve(file));
                }
            }
            return total;
        }
    }
}
