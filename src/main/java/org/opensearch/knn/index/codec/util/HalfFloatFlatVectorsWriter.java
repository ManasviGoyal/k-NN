/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;

public class HalfFloatFlatVectorsWriter extends FlatVectorsWriter {

    private final FlatVectorsWriter delegate;

    public HalfFloatFlatVectorsWriter(FlatVectorsWriter delegate) {
        super(delegate.getFlatVectorScorer());
        this.delegate = delegate;
    }

    private static class HalfFloatFieldWriter extends FlatFieldVectorsWriter<byte[]> {
        private final FlatFieldVectorsWriter<byte[]> delegate;

        public HalfFloatFieldWriter(FlatFieldVectorsWriter<byte[]> delegate) {
            this.delegate = delegate;
        }

        public void addValue(int docID, float[] vectorValue) throws IOException {
            byte[] halfFloatBytes = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vectorValue);
            delegate.addValue(docID, halfFloatBytes);
        }

        @Override
        public void addValue(int docID, byte[] vectorValue) throws IOException {
            delegate.addValue(docID, vectorValue);
        }

        @Override
        public void finish() throws IOException {
            delegate.finish();
        }

        @Override
        public boolean isFinished() {
            return delegate.isFinished();
        }

        @Override
        public long ramBytesUsed() {
            return delegate.ramBytesUsed();
        }

        @Override
        public java.util.List<byte[]> getVectors() {
            return delegate.getVectors();
        }
        @Override
        public org.apache.lucene.index.DocsWithFieldSet getDocsWithFieldSet() {
            return delegate.getDocsWithFieldSet();
        }

        @Override
        public byte[] copyValue(byte[] value) {
            return ArrayUtil.copyOfSubArray(value, 0, value.length);
        }
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        return new HalfFloatFieldWriter((FlatFieldVectorsWriter<byte[]>) delegate.addField(fieldInfo));
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        // Implement fp16 writing logic here if needed, or delegate
        delegate.flush(maxDoc, sortMap);
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
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        delegate.mergeOneField(fieldInfo, mergeState);
    }

    @Override
    public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        return delegate.mergeOneFieldToIndex(fieldInfo, mergeState);
    }

    @Override
    public long ramBytesUsed() {
        return delegate.ramBytesUsed();
    }
}