/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;

import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;

import org.apache.lucene.util.ArrayUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.META_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.META_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT;

/**
 * Writer for half-precision (FP16) flat vector fields. Encodes incoming FP32 vectors to FP16
 * (2 bytes per dimension) and writes them sequentially to a {@code .vec} file, with per-field
 * metadata stored in a {@code .vemf} file.
 *
 * <p>The on-disk layout follows the same structural pattern as Lucene's {@code Lucene99FlatVectorsWriter}:
 * <ul>
 *   <li>{@code .vec} — contiguous FP16-encoded vector data, one field after another</li>
 *   <li>{@code .vemf} — per-field metadata (field number, similarity function, offset, length,
 *       dimension, doc count, ord-to-doc mapping)</li>
 * </ul>
 *
 * <p>Each float dimension is converted to IEEE 754 half-float via {@link Float#floatToFloat16(float)}
 * and stored as 2 bytes in little-endian order.
 */
public class KNN1040HalfFloatFlatVectorsWriter extends FlatVectorsWriter {

    private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(KNN1040HalfFloatFlatVectorsWriter.class);

    private static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

    private final SegmentWriteState segmentWriteState;
    private final IndexOutput meta;
    private final IndexOutput vectorData;
    private final List<FieldData> fields = new ArrayList<>();
    private boolean finished;

    private record FieldData(FlatFieldVectorsWriter<?> fieldWriter, FieldInfo fieldInfo) {
    }

    /**
     * Creates a new writer for FP16 flat vectors.
     *
     * @param state  the segment write state
     * @param scorer the flat vectors scorer used for scoring during indexing
     * @throws IOException if an I/O error occurs while creating output files
     */
    public KNN1040HalfFloatFlatVectorsWriter(SegmentWriteState state, FlatVectorsScorer scorer) throws IOException {
        super(scorer);
        this.segmentWriteState = state;

        boolean success = false;
        try {
            String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
            String vectorDataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION);

            meta = state.directory.createOutput(metaFileName, state.context);
            vectorData = state.directory.createOutput(vectorDataFileName, state.context);

            CodecUtil.writeIndexHeader(meta, META_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            CodecUtil.writeIndexHeader(vectorData, VECTOR_DATA_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        // NOTE: FlatFieldVectorsWriter has no public static create() — we use our own inline impl.
        FlatFieldVectorsWriter<?> fieldWriter = new FloatFieldWriter(fieldInfo);
        fields.add(new FieldData(fieldWriter, fieldInfo));
        return fieldWriter;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        for (FieldData field : fields) {
            if (sortMap == null) {
                writeField(field.fieldWriter(), field.fieldInfo(), maxDoc);
            } else {
                writeSortingField(field.fieldWriter(), field.fieldInfo(), maxDoc, sortMap);
            }
            field.fieldWriter().finish();
        }
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        if (meta != null) {
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }
        if (vectorData != null) {
            CodecUtil.writeFooter(vectorData);
        }
    }

    @Override
    public void mergeOneFlatVectorField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        long vectorDataOffset = vectorData.getFilePointer();
        int dimension = fieldInfo.getVectorDimension();
        byte[] outputBuffer = new byte[dimension * Short.BYTES];

        DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        int docCount = 0;

        for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
            if (mergeState.knnVectorsReaders[i] == null) {
                continue;
            }
            FloatVectorValues vectorValues = mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldInfo.name);
            if (vectorValues == null) {
                continue;
            }
            KnnVectorValues.DocIndexIterator iterator = vectorValues.iterator();
            MergeState.DocMap docMap = mergeState.docMaps[i];
            int doc;
            while ((doc = iterator.nextDoc()) != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS) {
                int newDoc = docMap.get(doc);
                if (newDoc == -1) {
                    continue;
                }
                float[] vector = vectorValues.vectorValue(iterator.index());
                KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, outputBuffer, vector.length);
                vectorData.writeBytes(outputBuffer, 0, dimension * Short.BYTES);
                docsWithFieldSet.add(newDoc);
                docCount++;
            }
        }

        long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
        writeMeta(fieldInfo, vectorDataOffset, vectorDataLength, docCount, docsWithFieldSet);
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        for (FieldData field : fields) {
            total += field.fieldWriter().ramBytesUsed();
        }
        return total;
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorData);
    }

    private void writeField(FlatFieldVectorsWriter<?> fieldWriter, FieldInfo fieldInfo, int maxDoc) throws IOException {
        int dimension = fieldInfo.getVectorDimension();
        byte[] outputBuffer = new byte[dimension * Short.BYTES];

        long vectorDataOffset = vectorData.getFilePointer();
        @SuppressWarnings("unchecked")
        List<float[]> vectors = (List<float[]>) fieldWriter.getVectors();
        DocsWithFieldSet docsWithFieldSet = fieldWriter.getDocsWithFieldSet();

        for (float[] vector : vectors) {
            KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, outputBuffer, vector.length);
            vectorData.writeBytes(outputBuffer, 0, dimension * Short.BYTES);
        }

        long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
        writeMeta(fieldInfo, vectorDataOffset, vectorDataLength, vectors.size(), docsWithFieldSet);
    }

    private void writeSortingField(FlatFieldVectorsWriter<?> fieldWriter, FieldInfo fieldInfo, int maxDoc, Sorter.DocMap sortMap)
        throws IOException {
        int dimension = fieldInfo.getVectorDimension();
        byte[] outputBuffer = new byte[dimension * Short.BYTES];

        long vectorDataOffset = vectorData.getFilePointer();
        @SuppressWarnings("unchecked")
        List<float[]> vectors = (List<float[]>) fieldWriter.getVectors();
        DocsWithFieldSet docsWithFieldSet = fieldWriter.getDocsWithFieldSet();

        int[] ordMap = new int[vectors.size()];
        DocsWithFieldSet sortedDocsWithField = new DocsWithFieldSet();
        int docId;
        int ord = 0;
        DocIdSetIterator iterator = docsWithFieldSet.iterator();
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            int newDocId = sortMap.oldToNew(docId);
            ordMap[ord] = newDocId;
            ord++;
        }

        Integer[] sortedOrds = new Integer[vectors.size()];
        for (int i = 0; i < sortedOrds.length; i++) {
            sortedOrds[i] = i;
        }
        java.util.Arrays.sort(sortedOrds, (a, b) -> Integer.compare(ordMap[a], ordMap[b]));

        for (int i = 0; i < sortedOrds.length; i++) {
            int sortedOrd = sortedOrds[i];
            float[] vector = vectors.get(sortedOrd);
            KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, outputBuffer, vector.length);
            vectorData.writeBytes(outputBuffer, 0, dimension * Short.BYTES);
            sortedDocsWithField.add(ordMap[sortedOrd]);
        }

        long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
        writeMeta(fieldInfo, vectorDataOffset, vectorDataLength, vectors.size(), sortedDocsWithField);
    }

    private void writeMeta(
        FieldInfo fieldInfo,
        long vectorDataOffset,
        long vectorDataLength,
        int docCount,
        DocsWithFieldSet docsWithFieldSet
    ) throws IOException {
        meta.writeInt(fieldInfo.number);
        meta.writeInt(fieldInfo.getVectorSimilarityFunction().ordinal());
        meta.writeVLong(vectorDataOffset);
        meta.writeVLong(vectorDataLength);
        meta.writeVInt(fieldInfo.getVectorDimension());
        meta.writeInt(docCount);
        OrdToDocDISIReaderConfiguration.writeStoredMeta(
            DIRECT_MONOTONIC_BLOCK_SHIFT,
            meta,
            vectorData,
            docCount,
            segmentWriteState.segmentInfo.maxDoc(),
            docsWithFieldSet
        );
    }

    // ─── Per-field writer: stores float[] on heap during indexing ────────────────

    private static class FloatFieldWriter extends FlatFieldVectorsWriter<float[]> {
        private final FieldInfo fieldInfo;
        private final List<float[]> vectors = new ArrayList<>();
        private final DocsWithFieldSet docsWithField = new DocsWithFieldSet();
        private boolean finished;
        private int lastDocID = -1;

        FloatFieldWriter(FieldInfo fieldInfo) {
            this.fieldInfo = fieldInfo;
        }

        @Override
        public void addValue(int docID, float[] vectorValue) throws IOException {
            if (finished) {
                throw new IllegalStateException("already finished");
            }
            if (docID == lastDocID) {
                throw new IllegalArgumentException("VectorValuesField \"" + fieldInfo.name + "\" appears more than once in this document");
            }
            docsWithField.add(docID);
            vectors.add(copyValue(vectorValue));
            lastDocID = docID;
        }

        @Override
        public float[] copyValue(float[] value) {
            return ArrayUtil.copyOfSubArray(value, 0, fieldInfo.getVectorDimension());
        }

        @Override
        public List<float[]> getVectors() {
            return vectors;
        }

        @Override
        public DocsWithFieldSet getDocsWithFieldSet() {
            return docsWithField;
        }

        @Override
        public long ramBytesUsed() {
            if (vectors.isEmpty()) return 0;
            return (long) vectors.size() * fieldInfo.getVectorDimension() * Float.BYTES;
        }

        @Override
        public void finish() throws IOException {
            finished = true;
        }

        @Override
        public boolean isFinished() {
            return finished;
        }
    }
}
