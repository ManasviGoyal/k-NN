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
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;

import java.io.IOException;
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
 * <p>When mmap is available and the similarity function has a native SIMD type (L2, MAX_IP), scoring
 * goes through {@code NativeEngines990KnnVectorsScorer} → {@code NativeRandomVectorScorer}. Otherwise
 * (mmap unavailable, or COSINE, which has no native SIMD type today), scoring uses
 * {@link KNN1040HalfFloatFlatVectorsValues#newFallbackScorer}, which either runs native SIMD on raw
 * bytes read through the {@code IndexInput} slice, or decodes FP16 to float32 and compares directly.
 * Both fallback paths deliberately avoid Lucene's own flat vector scorer factory, which would
 * otherwise detect {@code HasIndexSlice} on {@link KNN1040HalfFloatFlatVectorsValues} and read the
 * slice assuming 4 bytes/dimension (float32) -- silently overreading past the buffer on this
 * 2 bytes/dimension data.
 */
@Log4j2
public class KNN1040HalfFloatFlatVectorsReader extends FlatVectorsReader {

    private static final int BULK_SCORE_BATCH_SIZE = 64;
    private static final String VECTOR_VALUES_SLICE = "KNN1040HalfFloatFlatVectorsValuesSlice";

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
            int versionMeta = readMetadata(state);

            String vectorDataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION);
            vectorData = state.directory.openInput(vectorDataFileName, dataContext);
            int versionData = CodecUtil.checkIndexHeader(
                vectorData,
                VECTOR_DATA_CODEC_NAME,
                VERSION_START,
                VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
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
        String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
        int versionMeta;
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
            Throwable priorE = null;
            try {
                versionMeta = CodecUtil.checkIndexHeader(
                    meta,
                    META_CODEC_NAME,
                    VERSION_START,
                    VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
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
            VectorSimilarityFunction similarity = VectorSimilarityFunction.values()[meta.readInt()];
            long vectorDataOffset = meta.readVLong();
            long vectorDataLength = meta.readVLong();
            int dimension = meta.readVInt();
            int size = meta.readInt();
            OrdToDocDISIReaderConfiguration ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, size);

            fields.put(info.name, new FieldEntry(similarity, vectorDataOffset, vectorDataLength, dimension, size, ordToDoc));
        }
    }

    /**
     * Builds a fresh {@link KNN1040HalfFloatFlatVectorsValues} over {@code entry}'s slice of the
     * {@code .vec} file.
     */
    private KNN1040HalfFloatFlatVectorsValues newVectorValues(FieldEntry entry) throws IOException {
        IndexInput slice = vectorData.slice(VECTOR_VALUES_SLICE, entry.vectorDataOffset, entry.vectorDataLength);
        DirectMonotonicReader ordToDocReader = entry.ordToDoc.isDense() ? null : entry.ordToDoc.getDirectMonotonicReader(vectorData);
        return new KNN1040HalfFloatFlatVectorsValues(entry.dimension, entry.size, slice, ordToDocReader, scorer, entry.similarity);
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        FieldEntry entry = fields.get(field);
        if (entry == null) {
            return null;
        }
        KNN1040HalfFloatFlatVectorsValues base = newVectorValues(entry);
        long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(base.getSlice(), 0, base.getSlice().length());
        return addressAndSize != null ? new MMapFloatVectorValues(base, addressAndSize) : base;
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        FieldEntry entry = fields.get(field);
        if (entry == null) {
            return null;
        }
        KNN1040HalfFloatFlatVectorsValues base = newVectorValues(entry);
        if (base.size() == 0) {
            return null;
        }

        long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(base.getSlice(), 0, base.getSlice().length());
        SimdVectorComputeService.SimilarityFunctionType nativeType = NativeEngines990KnnVectorsScorer.getNativeFunctionType(
            entry.similarity
        );
        if (addressAndSize != null && nativeType != null) {
            // Only route through the shared scorer chain (and thus, potentially, Lucene's delegate)
            // once mmap succeeded AND a native SIMD type exists. That chain builds
            // NativeRandomVectorScorer directly in this case and never reaches Lucene's delegate.
            MMapFloatVectorValues mmapValues = new MMapFloatVectorValues(base, addressAndSize);
            return scorer.getRandomVectorScorer(entry.similarity, mmapValues, target);
        }
        // mmap unavailable, or no native SIMD type for this similarity function (COSINE today):
        // never hand `base` to the shared scorer chain here -- it would fall through to Lucene's
        // delegate, which reads KNN1040HalfFloatFlatVectorsValues' HasIndexSlice-exposed slice
        // assuming float32, corrupting scores and eventually reading past the buffer on tail vectors.
        return KNN1040HalfFloatFlatVectorsValues.newFallbackScorer(base, target, entry.similarity);
    }

    /**
     * Exhaustive brute-force search over all FP16 vectors. Gets a scorer (native SIMD when mmap and
     * a native type are both available, otherwise a fallback scorer -- see
     * {@link #getRandomVectorScorer(String, float[])}), then iterates all ords in batches, collecting
     * into knnCollector.
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
                    collectBatch(randomScorer, knnCollector, ords, scores, numOrds);
                    numOrds = 0;
                }
            }
        }

        if (numOrds > 0) {
            collectBatch(randomScorer, knnCollector, ords, scores, numOrds);
        }
    }

    private void collectBatch(RandomVectorScorer scorer, KnnCollector knnCollector, int[] ords, float[] scores, int numOrds)
        throws IOException {
        knnCollector.incVisitedCount(numOrds);
        if (scorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
            for (int j = 0; j < numOrds; j++) {
                knnCollector.collect(scorer.ordToDoc(ords[j]), scores[j]);
            }
        }
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        throw new UnsupportedOperationException("FP16 format does not support byte vectors");
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

    private record FieldEntry(VectorSimilarityFunction similarity, long vectorDataOffset, long vectorDataLength, int dimension, int size,
        OrdToDocDISIReaderConfiguration ordToDoc) {
        FieldEntry {
            long expectedBytes = (long) size * dimension * Short.BYTES;
            if (expectedBytes != vectorDataLength) {
                throw new IllegalStateException(
                    "Vector data length "
                        + vectorDataLength
                        + " not matching size="
                        + size
                        + " * dim="
                        + dimension
                        + " * byteSize="
                        + Short.BYTES
                        + " = "
                        + expectedBytes
                );
            }
        }
    }
}
