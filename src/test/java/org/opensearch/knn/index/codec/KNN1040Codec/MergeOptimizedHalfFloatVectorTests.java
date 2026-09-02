/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tests {@link MergeOptimizedHalfFloatVector}, whose job is to hand the writer FP16 bytes without
 * decoding them. Correctness alone can't tell the fast path from the fallback - both produce the
 * same bytes, since the FP16/FP32 round trip is lossless - so
 * {@link #testMerge_readsRawBytesAndNeverDecodes_whenSourceIsHalfFloat} spies on the source values
 * and asserts the decode method is never called.
 */
public class MergeOptimizedHalfFloatVectorTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_vector";
    private static final int DIMENSION = 8;

    /**
     * The point of the optimization: merging FP16-backed segments reads raw bytes and never decodes.
     * Spying on the source values makes that observable - {@code vectorValue} is the only decoding
     * path, so zero invocations means zero conversion.
     */
    @SneakyThrows
    public void testMerge_readsRawBytesAndNeverDecodes_whenSourceIsHalfFloat() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { randomVector(), randomVector(), randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                KNN1040HalfFloatFlatVectorsValues spiedValues = Mockito.spy(
                    (KNN1040HalfFloatFlatVectorsValues) reader.getFloatVectorValues(FIELD_NAME)
                );

                List<byte[]> merged = collectMergedBytes(mergeStateOver(dir, identityDocMaps(1), spiedValues));

                assertVectorsEqual(vectors, merged);
                Mockito.verify(spiedValues, Mockito.never()).vectorValue(Mockito.anyInt());
                Mockito.verify(spiedValues, Mockito.times(vectors.length))
                    .readRawVectorBytes(Mockito.anyInt(), Mockito.any(), Mockito.anyInt());
            }
        }
    }

    /**
     * Segments predating the {@code half_float} data type store FP32 and have no bytes to copy, so
     * those subs decode and re-encode instead. Mixing one with an FP16 segment checks the choice is
     * made per sub rather than once for the whole merge.
     */
    @SneakyThrows
    public void testMerge_fallsBackToEncoding_forNonHalfFloatSource() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] halfFloatVectors = { randomVector(), randomVector() };
            float[][] plainVectors = { randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", halfFloatVectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                FloatVectorValues halfFloatValues = reader.getFloatVectorValues(FIELD_NAME);
                FloatVectorValues plainValues = plainFloatVectorValues(plainVectors);

                // The plain sub's docs land after the FP16 sub's in the merged segment.
                MergeState.DocMap[] docMaps = { docID -> docID, docID -> halfFloatVectors.length + docID };
                List<byte[]> merged = collectMergedBytes(mergeStateOver(dir, docMaps, halfFloatValues, plainValues));

                assertVectorsEqual(new float[][] { halfFloatVectors[0], halfFloatVectors[1], plainVectors[0] }, merged);
            }
        }
    }

    /**
     * Readers hand back FP16 values wrapped for mmap-backed segments, so the unwrap has to see
     * through {@link MMapFloatVectorValues} to reach the raw bytes underneath - otherwise every
     * mmap-backed merge, which is the common case, silently takes the slow path.
     */
    @SneakyThrows
    public void testMerge_readsRawBytes_throughMMapWrapper() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { randomVector(), randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                KNN1040HalfFloatFlatVectorsValues spiedValues = Mockito.spy(
                    (KNN1040HalfFloatFlatVectorsValues) reader.getFloatVectorValues(FIELD_NAME)
                );
                // A ByteBuffersDirectory isn't mmap-backed, so wrap explicitly to exercise the layer
                // a real MMapDirectory-backed reader would have added.
                MMapFloatVectorValues wrapped = new MMapFloatVectorValues(spiedValues, new long[] { 1L, 1L });

                List<byte[]> merged = collectMergedBytes(mergeStateOver(dir, identityDocMaps(1), wrapped));

                assertVectorsEqual(vectors, merged);
                Mockito.verify(spiedValues, Mockito.never()).vectorValue(Mockito.anyInt());
            }
        }
    }

    /**
     * Under an index sort, merged order follows the doc maps rather than the order the segments were
     * listed in - the reason this reuses Lucene's {@code DocIDMerger} instead of looping over readers.
     */
    @SneakyThrows
    public void testMerge_ordersVectorsByMappedDocId_underIndexSort() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] firstVectors = { randomVector() };
            float[][] secondVectors = { randomVector() };
            SegmentReadState firstState = writeHalfFloatSegment(dir, "_0", firstVectors);
            SegmentReadState secondState = writeHalfFloatSegment(dir, "_1", secondVectors);

            try (
                FlatVectorsReader firstReader = new KNN1040HalfFloatFlatVectorsReader(firstState, mockScorer());
                FlatVectorsReader secondReader = new KNN1040HalfFloatFlatVectorsReader(secondState, mockScorer())
            ) {
                // The second segment's doc sorts ahead of the first's, as an index sort could arrange.
                MergeState.DocMap[] docMaps = { docID -> docID + 1, docID -> docID };
                MergeState mergeState = mergeStateOver(
                    dir,
                    docMaps,
                    true,
                    firstReader.getFloatVectorValues(FIELD_NAME),
                    secondReader.getFloatVectorValues(FIELD_NAME)
                );

                assertVectorsEqual(new float[][] { secondVectors[0], firstVectors[0] }, collectMergedBytes(mergeState));
            }
        }
    }

    @SneakyThrows
    public void testCurrentVectorBytes_throwsBeforeIteration() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", new float[][] { randomVector() });

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                MergeState mergeState = mergeStateOver(dir, identityDocMaps(1), reader.getFloatVectorValues(FIELD_NAME));
                MergeOptimizedHalfFloatVector values = MergeOptimizedHalfFloatVector.create(createFieldInfo(), mergeState);

                IllegalStateException e = expectThrows(
                    IllegalStateException.class,
                    () -> values.currentVectorBytes(new byte[DIMENSION * Short.BYTES])
                );
                assertTrue(e.getMessage().contains("nextDoc"));
            }
        }
    }

    /** Drives the merged view the way the writer does, collecting each vector's FP16 bytes. */
    private List<byte[]> collectMergedBytes(MergeState mergeState) throws IOException {
        MergeOptimizedHalfFloatVector values = MergeOptimizedHalfFloatVector.create(createFieldInfo(), mergeState);
        List<byte[]> collected = new ArrayList<>();
        KnnVectorValues.DocIndexIterator iterator = values.iterator();
        for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
            byte[] buffer = new byte[DIMENSION * Short.BYTES];
            values.currentVectorBytes(buffer);
            collected.add(buffer);
        }
        return collected;
    }

    /** Compares against the FP16 encoding of the originals, which is what a source segment stores. */
    private void assertVectorsEqual(float[][] expected, List<byte[]> actual) {
        assertEquals("vector count", expected.length, actual.size());
        byte[] expectedBytes = new byte[DIMENSION * Short.BYTES];
        for (int i = 0; i < expected.length; i++) {
            KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(expected[i], expectedBytes, DIMENSION);
            assertArrayEquals("vector " + i, expectedBytes, actual.get(i));
        }
    }

    private MergeState mergeStateOver(Directory dir, MergeState.DocMap[] docMaps, FloatVectorValues... values) throws IOException {
        return mergeStateOver(dir, docMaps, false, values);
    }

    /**
     * @param needsIndexSort selects {@link org.apache.lucene.index.DocIDMerger}'s sorted merger, which
     *                       orders vectors by mapped doc id; without it subs are simply concatenated.
     */
    private MergeState mergeStateOver(Directory dir, MergeState.DocMap[] docMaps, boolean needsIndexSort, FloatVectorValues... values)
        throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { createFieldInfo() });
        KnnVectorsReader[] readers = new KnnVectorsReader[values.length];
        FieldInfos[] sourceFieldInfos = new FieldInfos[values.length];
        Bits[] liveDocs = new Bits[values.length];
        int[] maxDocs = new int[values.length];
        int totalDocs = 0;
        for (int i = 0; i < values.length; i++) {
            KnnVectorsReader reader = Mockito.mock(KnnVectorsReader.class);
            Mockito.when(reader.getFloatVectorValues(FIELD_NAME)).thenReturn(values[i]);
            readers[i] = reader;
            sourceFieldInfos[i] = fieldInfos;
            maxDocs[i] = values[i].size();
            totalDocs += values[i].size();
        }

        return new MergeState(
            docMaps,
            createSegmentInfo(dir, "_merged", totalDocs),
            fieldInfos,
            null,
            null,
            null,
            null,
            sourceFieldInfos,
            liveDocs,
            null,
            null,
            readers,
            maxDocs,
            InfoStream.NO_OUTPUT,
            null,
            needsIndexSort,
            null
        );
    }

    private MergeState.DocMap[] identityDocMaps(int count) {
        MergeState.DocMap[] docMaps = new MergeState.DocMap[count];
        for (int i = 0; i < count; i++) {
            docMaps[i] = docID -> docID;
        }
        return docMaps;
    }

    /** Writes a real FP16 segment so the reader hands back real {@code KNN1040HalfFloatFlatVectorsValues}. */
    @SuppressWarnings("unchecked")
    private SegmentReadState writeHalfFloatSegment(Directory dir, String segmentName, float[][] vectors) throws IOException {
        FieldInfo fieldInfo = createFieldInfo();
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentInfo segmentInfo = createSegmentInfo(dir, segmentName, vectors.length);
        SegmentWriteState writeState = new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        try (FlatVectorsWriter writer = new KNN1040HalfFloatFlatVectorsWriter(writeState, mockScorer())) {
            FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
            for (int docID = 0; docID < vectors.length; docID++) {
                fieldWriter.addValue(docID, vectors[docID]);
            }
            writer.flush(vectors.length, null);
            writer.finish();
        }

        return new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
    }

    /** Stand-in for a source segment that stores FP32, with no FP16 bytes to copy. */
    private FloatVectorValues plainFloatVectorValues(float[][] vectors) {
        return new FloatVectorValues() {
            @Override
            public int dimension() {
                return DIMENSION;
            }

            @Override
            public int size() {
                return vectors.length;
            }

            @Override
            public float[] vectorValue(int ord) {
                return vectors[ord];
            }

            @Override
            public FloatVectorValues copy() {
                return this;
            }

            @Override
            public DocIndexIterator iterator() {
                return createDenseIterator();
            }
        };
    }

    private SegmentInfo createSegmentInfo(Directory dir, String name, int maxDoc) {
        return new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            name,
            maxDoc,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            new HashMap<>(),
            null
        );
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

    private FlatVectorsScorer mockScorer() {
        return Mockito.mock(FlatVectorsScorer.class);
    }

    private float[] randomVector() {
        float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = randomFloat();
        }
        return vector;
    }
}
