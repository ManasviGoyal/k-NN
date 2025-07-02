/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.MergeState;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.apache.lucene.index.ByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;

public class HalfFloatFlatVectorsWriter extends FlatVectorsWriter {
    private final FlatVectorsWriter delegate;
    private final SegmentWriteState state;

    public HalfFloatFlatVectorsWriter(FlatVectorsWriter delegate, SegmentWriteState state) {
        super(delegate.getScorer());
        this.delegate = delegate;
        this.state = state;
    }

    @Override
    public void writeField(FieldInfo fieldInfo, KnnVectorValues values) throws IOException {
        VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        if (vectorDataType == VectorDataType.HALF_FLOAT) {
            // If values are already byte[] (e.g., during merges), write as-is
            if (values instanceof ByteVectorValues) {
                delegate.writeField(fieldInfo, values);
                return;
            }
            // If values are float[], convert to fp16 byte[] and wrap
            if (values instanceof KNNVectorValues) {
                KNNVectorValues<float[]> floatValues = (KNNVectorValues<float[]>) values;
                float[] vector = floatValues.vectorValue();
                byte[] fp16Bytes = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector);
                KNNByteVectorValues byteVectorValues = new KNNByteVectorValues(fp16Bytes, floatValues.docID());
                delegate.writeField(fieldInfo, byteVectorValues);
                return;
            }
            throw new IllegalArgumentException("Unsupported KnnVectorValues type for HALF_FLOAT");
        } else {
            delegate.writeField(fieldInfo, values);
        }
    }

    @Override
    public void finish() throws IOException {
        delegate.finish();
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public void mergeOneFieldToIndex(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        delegate.mergeOneFieldToIndex(fieldInfo, mergeState);
    }
}