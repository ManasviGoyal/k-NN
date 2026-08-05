/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import org.apache.lucene.index.Sorter;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class KNN1040HalfFloatFlatVectorsFormatTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_vector";
    private static final int DIMENSION = 8;
    private static final int NUM_VECTORS = 10;

    @SneakyThrows
    public void testFormatName() {
        KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
        assertEquals("KNN1040HalfFloatFlatVectorsFormat", format.getClass().getSimpleName());
    }

    @SneakyThrows
    public void testGetMaxDimensions() {
        KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("test-field"));
    }

    @SneakyThrows
    public void testWriteAndRead_roundTrip() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertNotNull(values);
                assertEquals(DIMENSION, values.dimension());
                assertEquals(NUM_VECTORS, values.size());

                for (int i = 0; i < NUM_VECTORS; i++) {
                    float[] actual = values.vectorValue(i);
                    assertNotNull(actual);
                    assertEquals(DIMENSION, actual.length);
                    for (int d = 0; d < DIMENSION; d++) {
                        float expected = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                        assertEquals("Vector " + i + " dim " + d, expected, actual[d], 0.0f);
                    }
                }
            }
        }
    }

    @SneakyThrows
    public void testWriteAndRead_cachedVectorValue() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(5, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                float[] first = values.vectorValue(2);
                float[] second = values.vectorValue(2);
                assertSame(first, second);
            }
        }
    }

    @SneakyThrows
    public void testWriteAndRead_vectorByteLength() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(3, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertEquals(DIMENSION * Short.BYTES, values.getVectorByteLength());
            }
        }
    }

    @SneakyThrows
    public void testGetByteVectorValues_throws() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(3, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                expectThrows(UnsupportedOperationException.class, () -> reader.getByteVectorValues(FIELD_NAME));
            }
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_unknownField_returnsNull() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(3, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                assertNull(reader.getFloatVectorValues("nonexistent"));
            }
        }
    }

    private SegmentReadState writeVectors(Directory dir, float[][] vectors) throws Exception {
        FieldInfo fieldInfo = createFieldInfo();
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        SegmentInfo segmentInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            vectors.length,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            new HashMap<>(),
            null
        );
        SegmentWriteState writeState = new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
        try (FlatVectorsWriter writer = format.fieldsWriter(writeState)) {
            @SuppressWarnings("unchecked")
            FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);

            for (int i = 0; i < vectors.length; i++) {
                fieldWriter.addValue(i, vectors[i]);
            }

            writer.flush(vectors.length, null);
            writer.finish();
        }

        return new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
    }

    @SneakyThrows
    public void testWriteWithSortMap_writesReorderedData() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = {
                { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
                { 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f } };

            FieldInfo fieldInfo = createFieldInfo();
            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

            SegmentInfo segmentInfo = new SegmentInfo(
                dir,
                Version.LATEST,
                Version.LATEST,
                "_0",
                2,
                false,
                false,
                null,
                Collections.emptyMap(),
                StringHelper.randomId(),
                new HashMap<>(),
                null
            );
            SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                dir,
                segmentInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT
            );

            Sorter.DocMap sortMap = new Sorter.DocMap() {
                @Override
                public int oldToNew(int docID) {
                    return docID == 0 ? 1 : 0;
                }

                @Override
                public int newToOld(int docID) {
                    return docID == 0 ? 1 : 0;
                }

                @Override
                public int size() {
                    return 2;
                }
            };

            KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
            try (FlatVectorsWriter writer = format.fieldsWriter(writeState)) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
                fieldWriter.addValue(0, vectors[0]);
                fieldWriter.addValue(1, vectors[1]);
                writer.flush(2, sortMap);
                writer.finish();
            }

            // Verify writer completes without error and creates files
            SegmentReadState readState = new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
            try (FlatVectorsReader reader = format.fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertEquals(2, values.size());
                assertNotNull(values.vectorValue(0));
                assertNotNull(values.vectorValue(1));
            }
        }
    }

    private FieldInfo createFieldInfo() {
        return new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false
        );
    }

    @SneakyThrows
    public void testScorer_nonMmapDirectory_matchesExpectedSimilarity() {
        // ByteBuffersDirectory does not back onto mmap'd memory, so getFloatVectorValues() returns
        // the plain (non-MMapFloatVectorValues-wrapped) HalfFloatVectorValues. This uses the
        // scorer() fallback path (used when native mmap addresses are unavailable)
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertFalse("Expected non-mmap-backed values for this test", values instanceof MMapFloatVectorValues);

                float[] query = generateVectors(1, DIMENSION)[0];
                VectorScorer scorer = values.scorer(query);
                assertNotNull(scorer);

                for (int i = 0; i < NUM_VECTORS; i++) {
                    assertTrue(scorer.iterator().nextDoc() != DocIdSetIterator.NO_MORE_DOCS);
                    float[] decoded = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                    }
                    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, decoded);
                    assertEquals("Vector " + i, expected, scorer.score(), 1e-3);
                }
            }
        }
    }

    @SneakyThrows
    public void testScorerBulk_nonMmapDirectory_matchesExpectedSimilarity() {
        // Exercises VectorScorer.bulk(), which is what the reader's exhaustive search() path uses.
        // With mmap unavailable, this goes through HalfFloatBytesRandomVectorScorer.bulkScore() when SIMD
        // is supported, or the pure-Java fallback (via the default Bulk implementation) otherwise.
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertFalse("Expected non-mmap-backed values for this test", values instanceof MMapFloatVectorValues);

                float[] query = generateVectors(1, DIMENSION)[0];
                VectorScorer scorer = values.scorer(query);
                assertNotNull(scorer);

                VectorScorer.Bulk bulk = scorer.bulk(null);
                DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
                float maxScore = bulk.nextDocsAndScores(NUM_VECTORS, null, buffer);
                assertEquals(NUM_VECTORS, buffer.size);

                float expectedMax = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < NUM_VECTORS; i++) {
                    float[] decoded = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        decoded[d] = Float.float16ToFloat(Float.floatToFloat16(vectors[i][d]));
                    }
                    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, decoded);
                    assertEquals("Vector " + i, expected, buffer.features[i], 1e-3);
                    expectedMax = Math.max(expectedMax, expected);
                }
                assertEquals(expectedMax, maxScore, 1e-3);
            }
        }
    }

    @SneakyThrows
    public void testFloatVectorValues_implementsHasIndexSlice() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = generateVectors(NUM_VECTORS, DIMENSION);
            SegmentReadState readState = writeVectors(dir, vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsFormat().fieldsReader(readState)) {
                FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertNotNull(values);

                // MMapFloatVectorValues wraps HalfFloatVectorValues which implements HasIndexSlice
                // This enables prefetch in PrefetchableFlatVectorScorer
                assertTrue(
                    "FloatVectorValues should implement HasIndexSlice for prefetch support",
                    values instanceof HasIndexSlice
                );
                HasIndexSlice hasSlice = (HasIndexSlice) values;
                assertNotNull("getSlice() should return non-null IndexInput for mmap-backed values", hasSlice.getSlice());
            }
        }
    }

    private float[][] generateVectors(int count, int dimension) {
        float[][] vectors = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = (random().nextFloat() * 2 - 1) * 10;
            }
        }
        return vectors;
    }
}
