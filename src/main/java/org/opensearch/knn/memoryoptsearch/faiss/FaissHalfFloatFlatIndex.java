/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import java.util.Locale;

import lombok.Getter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * A virtual FaissIndex that serves as a proxy for Lucene's FP16 flat vectors reader. Used for the
 * Faiss HNSW cases where flat storage is skipped natively via IO_FLAG_SKIP_STORAGE, because Lucene's
 * own {@code .vec} file already holds an equivalent copy. This class bridges the two by installing itself
 * as the flat storage under {@link FaissHNSWIndex}, providing access to the FP16 reader for scoring.
 */
public class FaissHalfFloatFlatIndex extends FaissIndex {
    static final String FAISS_HALF_FLOAT_FLAT_INDEX = "FaissHalfFloatFlatIndex";

    @Getter
    private final FlatVectorsReader flatVectorsReader;
    @Getter
    private final String fieldName;

    public FaissHalfFloatFlatIndex(final FlatVectorsReader flatVectorsReader, final String fieldName) {
        super(FAISS_HALF_FLOAT_FLAT_INDEX);
        this.flatVectorsReader = flatVectorsReader;
        this.fieldName = fieldName;
    }

    @Override
    protected void doLoad(IndexInput input) {
        // No-op: vectors are served by the Lucene FlatVectorsReader, not loaded from the faiss file.
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        return flatVectorsReader.getFloatVectorValues(fieldName);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        throw new UnsupportedOperationException(
            String.format(Locale.ROOT, "%s does not support byte vector values.", FAISS_HALF_FLOAT_FLAT_INDEX)
        );
    }
}
