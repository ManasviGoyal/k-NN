/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;
import org.apache.lucene.index.ByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.apache.lucene.util.Sorter;

public class HalfFloatFlatVectorsWriter extends FlatVectorsWriter {
    private final FlatVectorsWriter delegate;

    public HalfFloatFlatVectorsWriter(FlatVectorsWriter delegate) {
        super(delegate.getFlatVectorScorer());
        this.delegate = delegate;
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        FlatFieldVectorsWriter<?> baseWriter = delegate.addField(fieldInfo);
        if (vectorDataType == VectorDataType.HALF_FLOAT) {
            // Only wrap the value writing for HALF_FLOAT fields
            return new FlatFieldVectorsWriter<Object>() {
                @Override
                public void addVectorValue(int docID, Object vector) throws IOException {
                    if (vector instanceof float[]) {
                        byte[] fp16Bytes = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray((float[]) vector);
                        baseWriter.addVectorValue(docID, fp16Bytes);
                    } else if (vector instanceof byte[]) {
                        baseWriter.addVectorValue(docID, vector);
                    } else {
                        throw new IllegalArgumentException("Unsupported vector type for HALF_FLOAT: " + vector.getClass());
                    }
                }
                @Override
                public void finish() throws IOException {
                    baseWriter.finish();
                }
                @Override
                public boolean isFinished() {
                    return baseWriter.isFinished();
                }
            };
        } else {
            return baseWriter;
        }
    }

    // Delegate all other methods
    @Override
    public org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(FieldInfo fieldInfo, org.apache.lucene.index.MergeState mergeState) throws IOException {
        return delegate.mergeOneFieldToIndex(fieldInfo, mergeState);
    }

    @Override
    public void finish() throws IOException {
        delegate.finish();
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }
}