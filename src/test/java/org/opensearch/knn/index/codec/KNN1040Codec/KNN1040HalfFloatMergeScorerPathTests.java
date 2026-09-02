/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Checks that the vectors the HNSW graph build reads back during a merge are memory mapped.
 *
 * <p>Why it matters: {@code Lucene99HnswVectorsWriter} reopens the flat vectors it just wrote and
 * builds the merged graph against them, and every scoring shortcut for FP16 hangs off that read being
 * mapped. {@link KNN1040HalfFloatFlatVectorsReader#getFloatVectorValues} only wraps in
 * {@code MMapFloatVectorValues} when an address can be extracted, and
 * {@code KNN1040HalfFloatVectorScorer} keys off that wrapper for both zero-copy candidate scoring and
 * for naming the graph-build target by ordinal instead of decoding it. If the merge-time read were
 * not mapped, all of that would quietly fall back to reading each candidate into a heap buffer -
 * correct, but far slower, and invisible without a test like this.
 */
public class KNN1040HalfFloatMergeScorerPathTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_vector";
    private static final int DIMENSION = 8;
    // Above Lucene's HNSW_GRAPH_THRESHOLD (100), so the merge actually builds a graph rather than
    // taking the tiny-segment path that never reads the vectors back.
    private static final int DOCS_PER_SEGMENT = 60;
    private static final int NUM_SEGMENTS = 3;

    @SneakyThrows
    public void testMerge_readsVectorDataThroughMemoryMappedInput() {
        try (
            Directory mmapDir = new MMapDirectory(createTempDir());
            VectorDataOpenRecordingDirectory dir = new VectorDataOpenRecordingDirectory(mmapDir)
        ) {
            final Codec codec = new Lucene104Codec() {
                @Override
                public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    return new KNN1040HnswHalfFloatVectorsFormat();
                }
            };

            try (IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig().setCodec(codec))) {
                for (int segment = 0; segment < NUM_SEGMENTS; segment++) {
                    for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
                        Document doc = new Document();
                        doc.add(new KnnFloatVectorField(FIELD_NAME, randomVector(), VectorSimilarityFunction.EUCLIDEAN));
                        writer.addDocument(doc);
                    }
                    writer.commit();
                }
                dir.startRecording();
                writer.forceMerge(1);
            }

            assertFalse("the merge should have read vector data back to build the graph, but no .vec was opened", dir.opens.isEmpty());
            for (VectorDataOpen open : dir.opens) {
                assertTrue(
                    "merge opened " + open.name() + " without a memory mapping, so FP16 scoring falls back to heap copies",
                    open.memoryMapped()
                );
            }
        }
    }

    /**
     * The mirror of the test above: on a directory that can't map, the same detection reports no
     * mapping. Without this the positive assertion could pass on a check that never fails.
     */
    @SneakyThrows
    public void testMerge_withoutMemoryMapping_readsVectorDataUnmapped() {
        try (
            Directory heapDir = new ByteBuffersDirectory();
            VectorDataOpenRecordingDirectory dir = new VectorDataOpenRecordingDirectory(heapDir)
        ) {
            mergeOneSegment(dir);

            assertFalse("the merge should still have read vector data back", dir.opens.isEmpty());
            for (VectorDataOpen open : dir.opens) {
                assertFalse("a ByteBuffersDirectory input cannot be memory mapped", open.memoryMapped());
            }
        }
    }

    /** Indexes {@link #NUM_SEGMENTS} segments, then records what force-merging them reads back. */
    private void mergeOneSegment(VectorDataOpenRecordingDirectory dir) throws IOException {
        final Codec codec = new Lucene104Codec() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new KNN1040HnswHalfFloatVectorsFormat();
            }
        };

        try (IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig().setCodec(codec))) {
            for (int segment = 0; segment < NUM_SEGMENTS; segment++) {
                for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, randomVector(), VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }
            dir.startRecording();
            writer.forceMerge(1);
        }
    }

    private float[] randomVector() {
        float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = randomFloat();
        }
        return vector;
    }

    private record VectorDataOpen(String name, boolean memoryMapped) {
    }

    /**
     * Notes every {@code .vec} opened once {@link #startRecording} is called, and whether the input
     * handed back is one an address can be extracted from - the same check
     * {@link KNN1040HalfFloatFlatVectorsReader} makes when deciding whether to wrap in
     * {@code MMapFloatVectorValues}. Recording starts after indexing so only the merge's reads count.
     */
    private static final class VectorDataOpenRecordingDirectory extends FilterDirectory {
        private final List<VectorDataOpen> opens = new ArrayList<>();
        private volatile boolean recording;

        VectorDataOpenRecordingDirectory(Directory in) {
            super(in);
        }

        void startRecording() {
            recording = true;
        }

        @Override
        public IndexInput openInput(String name, IOContext context) throws IOException {
            IndexInput input = super.openInput(name, context);
            if (recording && name.endsWith(".vec")) {
                // The reader extracts from a slice of the field's data block, not from the file input,
                // so slice before checking - a mapped file whose slices weren't extractable would still
                // leave the scorer on the fallback path.
                try (IndexInput slice = input.slice("vector-data", 0, input.length())) {
                    boolean memoryMapped = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(slice, 0, slice.length()) != null;
                    synchronized (opens) {
                        opens.add(new VectorDataOpen(name, memoryMapped));
                    }
                }
            }
            return input;
        }
    }
}
