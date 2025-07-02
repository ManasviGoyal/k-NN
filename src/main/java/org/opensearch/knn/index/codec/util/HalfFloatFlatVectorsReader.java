/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;

public class HalfFloatFlatVectorsReader extends FlatVectorsReader {
    private final FlatVectorsReader delegate;
    private final SegmentReadState state;

    public HalfFloatFlatVectorsReader(FlatVectorsReader delegate, SegmentReadState state) {
        super(delegate.getScorer());
        this.delegate = delegate;
        this.state = state;
    }

    @Override
    public KnnVectorValues getVectorValues(String field) throws IOException {
        FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
        VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        KnnVectorValues base = delegate.getVectorValues(field);
        if (vectorDataType == VectorDataType.HALF_FLOAT) {
            // Use the factory method to wrap values in KNNHalfFloatVectorValues
            return KNNVectorValuesFactory.getVectorValues(vectorDataType, base);
        } else {
            return base;
        }
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        delegate.search(field, target, knnCollector, acceptDocs);
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public FlatVectorsReader getRandomVectorScorer(String field, byte[] target) throws IOException {
        return delegate.getRandomVectorScorer(field, target);
    }
}