/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import org.apache.lucene.util.packed.DirectMonotonicReader;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.META_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.META_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VERSION_START;

/**
 * Reader for half-precision (FP16) flat vector fields.
 *
 * <p>On segment open, this reader:
 * <ol>
 *   <li>Reads per-field metadata from the {@code .vemf} file (dimension, offset, length, ord-to-doc)</li>
 *   <li>Opens the {@code .vec} data file and creates per-field slices</li>
 *   <li>Attempts mmap address extraction for native SIMD scoring</li>
 * </ol>
 *
 * <p>The returned {@link FloatVectorValues} implements {@link MMapVectorValues} when mmap is available,
 * enabling the scorer chain (NativeEngines990KnnVectorsScorer → NativeRandomVectorScorer) to use
 * SIMD-accelerated FP16 distance computation. When mmap is unavailable, the fallback path decodes
 * FP16 bytes to float[] using {@link Float#float16ToFloat(short)}.
 */
@Log4j2
public class KNN1040HalfFloatFlatVectorsReader extends FlatVectorsReader {

    private static final int BULK_SCORE_BATCH_SIZE = 64;

    private final Map<String, FieldEntry> fields = new HashMap<>();
    private final IndexInput vectorData;
    private final FieldInfos fieldInfos;
    private final FlatVectorsScorer scorer;
    private final IOContext dataContext;

    public KNN1040HalfFloatFlatVectorsReader(SegmentReadState state, FlatVectorsScorer scorer) throws IOException {
        super();
        this.scorer = scorer;
        this.fieldInfos = state.fieldInfos;
        this.dataContext = state.context;

        boolean success = false;
        try {
            // Read metadata
            int versionMeta = readMetadata(state);

            // Open vector data file
            String vectorDataFileName = IndexFileNames.segmentFileName(
                state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION
            );
            vectorData = state.directory.openInput(vectorDataFileName, dataContext);
            int versionData = CodecUtil.checkIndexHeader(
                vectorData, VECTOR_DATA_CODEC_NAME, VERSION_START, VERSION_CURRENT,
                state.segmentInfo.getId(), state.segmentSuffix
            );
            if (versionMeta != versionData) {
                throw new IOException("Version mismatch: meta=" + versionMeta + " data=" + versionData);
            }
            CodecUtil.retrieveChecksum(vectorData);
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    private int readMetadata(SegmentReadState state) throws IOException {
        String metaFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, META_EXTENSION
        );
        int versionMeta;
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
            Throwable priorE = null;
            try {
                versionMeta = CodecUtil.checkIndexHeader(
                    meta, META_CODEC_NAME, VERSION_START, VERSION_CURRENT,
                    state.segmentInfo.getId(), state.segmentSuffix
                );
                readFields(meta);
            } catch (Throwable t) {
                priorE = t;
                throw t;
            } finally {
                CodecUtil.checkFooter(meta, priorE);
            }
        }
        return versionMeta;
    }

    private void readFields(ChecksumIndexInput meta) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
            if (info == null) {
                throw new IOException("Invalid field number: " + fieldNumber);
            }
            int similarityOrd = meta.readInt();
            VectorSimilarityFunction similarity = VectorSimilarityFunction.values()[similarityOrd];
            long vectorDataOffset = meta.readVLong();
            long vectorDataLength = meta.readVLong();
            int dimension = meta.readVInt();
            int size = meta.readInt();
            OrdToDocDISIReaderConfiguration ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, size);

            fields.put(info.name, new FieldEntry(similarity, vectorDataOffset, vectorDataLength, dimension, size, ordToDoc));
        }
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        FieldEntry entry = fields.get(field);
        if (entry == null) {
            return null;
        }
        IndexInput slice = vectorData.slice("fp16-vector-data", entry.vectorDataOffset, entry.vectorDataLength);
        DirectMonotonicReader ordToDocReader = null;
        if (!entry.ordToDoc.isDense()) {
            ordToDocReader = entry.ordToDoc.getDirectMonotonicReader(vectorData);
        }
        FloatVectorValues base = new HalfFloatVectorValues(entry.dimension, entry.size, slice, ordToDocReader, scorer, entry.similarity);
        long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(slice, 0, slice.length());

        if (addressAndSize != null) {
            return new MMapFloatVectorValues(base, addressAndSize);
        }
        return base;
    }

    /**
     * Exhaustive brute-force search over all FP16 vectors.
     * Gets a scorer (which will be NativeRandomVectorScorer if mmap is available),
     * then iterates all ords in batches, collecting into knnCollector.
     */
    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        RandomVectorScorer randomScorer = getRandomVectorScorer(field, target);
        if (randomScorer == null) {
            return;
        }

        int numVectors = randomScorer.maxOrd();
        if (numVectors == 0 || knnCollector.k() == 0) {
            return;
        }

        final Bits acceptedOrds = randomScorer.getAcceptOrds(acceptDocs.bits());
        int[] ords = new int[BULK_SCORE_BATCH_SIZE];
        float[] scores = new float[BULK_SCORE_BATCH_SIZE];
        int numOrds = 0;

        for (int i = 0; i < numVectors; i++) {
            if (acceptedOrds == null || acceptedOrds.get(i)) {
                if (knnCollector.earlyTerminated()) {
                    break;
                }
                ords[numOrds++] = i;
                if (numOrds == BULK_SCORE_BATCH_SIZE) {
                    knnCollector.incVisitedCount(numOrds);
                    if (randomScorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
                        for (int j = 0; j < numOrds; j++) {
                            knnCollector.collect(randomScorer.ordToDoc(ords[j]), scores[j]);
                        }
                    }
                    numOrds = 0;
                }
            }
        }

        // Flush remaining
        if (numOrds > 0) {
            knnCollector.incVisitedCount(numOrds);
            if (randomScorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
                for (int j = 0; j < numOrds; j++) {
                    knnCollector.collect(randomScorer.ordToDoc(ords[j]), scores[j]);
                }
            }
        }
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        throw new UnsupportedOperationException("FP16 format does not support byte vectors");
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        FloatVectorValues vectorValues = getFloatVectorValues(field);
        FieldEntry entry = fields.get(field);
        return scorer.getRandomVectorScorer(entry.similarity, vectorValues, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        throw new UnsupportedOperationException("FP16 format does not support byte vector scoring");
    }

    @Override
    public FlatVectorsScorer getFlatVectorScorer(String field) throws IOException {
        return scorer;
    }

    @Override
    public FlatVectorsReader getMergeInstance() throws IOException {
        vectorData.updateIOContext(dataContext.withHints(DataAccessHint.SEQUENTIAL));
        return this;
    }

    @Override
    public void checkIntegrity() throws IOException {
        CodecUtil.checksumEntireFile(vectorData);
    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(vectorData);
    }

    // ─── Field metadata ─────────────────────────────────────────────────────────

    private record FieldEntry(
        VectorSimilarityFunction similarity,
        long vectorDataOffset,
        long vectorDataLength,
        int dimension,
        int size,
        OrdToDocDISIReaderConfiguration ordToDoc
    ) {
        FieldEntry {
            long expectedBytes = (long) size * dimension * Short.BYTES;
            if (expectedBytes != vectorDataLength) {
                throw new IllegalStateException(
                    "Vector data length " + vectorDataLength
                        + " not matching size=" + size
                        + " * dim=" + dimension
                        + " * byteSize=" + Short.BYTES
                        + " = " + expectedBytes
                );
            }
        }
    }

    // ─── FP16 → FP32 decoding FloatVectorValues ───────────────────────────────────

    /**
     * Decodes FP16 bytes to float[] on access. When wrapped by MMapFloatVectorValues,
     * the SIMD scorer reads FP16 directly from native memory instead of calling vectorValue().
     * When not wrapped (no mmap), the delegate scorer calls vectorValue() for Java-side scoring.
     */
    private static class HalfFloatVectorValues extends FloatVectorValues {
        private final int dimension;
        private final int size;
        private final IndexInput slice;
        private final int byteSize;
        private final float[] value;
        private final DirectMonotonicReader ordToDocReader;
        private final FlatVectorsScorer flatVectorsScorer;
        private final VectorSimilarityFunction similarity;
        private int lastOrd = -1;

        HalfFloatVectorValues(
            int dimension,
            int size,
            IndexInput slice,
            DirectMonotonicReader ordToDocReader,
            FlatVectorsScorer flatVectorsScorer,
            VectorSimilarityFunction similarity
        ) {
            this.dimension = dimension;
            this.size = size;
            this.slice = slice;
            this.byteSize = dimension * Short.BYTES;
            this.value = new float[dimension];
            this.ordToDocReader = ordToDocReader;
            this.flatVectorsScorer = flatVectorsScorer;
            this.similarity = similarity;
        }

        IndexInput getSlice() {
            return slice;
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int getVectorByteLength() {
            return byteSize;
        }

        @Override
        public DocIndexIterator iterator() {
            if (ordToDocReader == null) {
                return createDenseIterator();
            }
            return createSparseIterator();
        }

        @Override
        public int ordToDoc(int ord) {
            if (ordToDocReader == null) {
                return ord;
            }
            return (int) ordToDocReader.get(ord);
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            if (ord == lastOrd) {
                return value;
            }
            lastOrd = ord;
            long offset = (long) ord * byteSize;
            byte[] raw = new byte[byteSize];
            slice.seek(offset);
            slice.readBytes(raw, 0, byteSize);
            ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < dimension; i++) {
                value[i] = Float.float16ToFloat(buf.getShort());
            }
            return value;
        }

        @Override
        public VectorScorer scorer(float[] target) throws IOException {
            if (size() == 0) return null;
            HalfFloatVectorValues copy = new HalfFloatVectorValues(dimension, size, slice.clone(), ordToDocReader, flatVectorsScorer, similarity);
            DocIndexIterator iterator = copy.iterator();
            return new VectorScorer() {
                @Override
                public float score() throws IOException {
                    return similarity.compare(target, copy.vectorValue(iterator.index()));
                }

                @Override
                public DocIdSetIterator iterator() {
                    return iterator;
                }
            };
        }

        @Override
        public FloatVectorValues copy() throws IOException {
            return new HalfFloatVectorValues(dimension, size, slice.clone(), ordToDocReader, flatVectorsScorer, similarity);
        }
    }
}
