/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

/**
 * A FlatVectorsFormat that wraps Lucene99FlatVectorsFormat and adds half-float support.
 */
public class HalfFloatFlatVectorsFormat extends FlatVectorsFormat {
    private final Lucene99FlatVectorsFormat lucene99FlatVectorsFormat;
    private static final String FORMAT_NAME = "HalfFloatFlatVectorsFormat";

    public HalfFloatFlatVectorsFormat() {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()));
    }

    public HalfFloatFlatVectorsFormat(Lucene99FlatVectorsFormat flatVectorsFormat) {
        super(FORMAT_NAME);
        this.lucene99FlatVectorsFormat = flatVectorsFormat;
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return lucene99FlatVectorsFormat.getMaxDimensions(fieldName);
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new HalfFloatFlatVectorsWriter(lucene99FlatVectorsFormat.fieldsWriter(state));
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        // For now, simply delegate to the Lucene99FlatVectorsReader
        return lucene99FlatVectorsFormat.fieldsReader(state);
    }

    @Override
    public String toString() {
        return "HalfFloatFlatVectorsFormat(name=HalfFloatFlatVectorsFormat, flatVectorsFormat=" + lucene99FlatVectorsFormat + ")";
    }
}