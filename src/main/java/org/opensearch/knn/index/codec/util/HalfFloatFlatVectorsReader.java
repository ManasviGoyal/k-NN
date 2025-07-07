/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;

public class HalfFloatFlatVectorsReader extends FlatVectorsReader {
    private final FlatVectorsReader delegate;
    private final FieldInfos fieldInfos;

    public HalfFloatFlatVectorsReader(FlatVectorsReader delegate, FieldInfos fieldInfos) {
        super(delegate.getFlatVectorScorer());
        this.delegate = delegate;
        this.fieldInfos = fieldInfos;
    }

    @Override
    public org.apache.lucene.index.ByteVectorValues getByteVectorValues(String field) throws IOException {
        // Delegate to the underlying reader
        return delegate.getByteVectorValues(field);
    }

    @Override
    public KnnVectorValues getVectorValues(String field) throws IOException {
        FieldInfo fieldInfo = fieldInfos.fieldInfo(field);
        VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        KnnVectorValues base = delegate.getVectorValues(field); // Correctly delegate
        if (vectorDataType == VectorDataType.HALF_FLOAT) {
            // Only wrap if base is not already a half-float wrapper
            return KNNVectorValuesFactory.getVectorValues(vectorDataType, base);
        } else {
            return base;
        }
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        delegate.search(field, target, knnCollector, acceptDocs);
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        // Delegate to the underlying reader
        return delegate.getRandomVectorScorer(field, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        // Delegate to the underlying reader
        return delegate.getRandomVectorScorer(field, target);
    }

    @Override
    public FlatVectorsReader getMergeInstance() {
        return new HalfFloatFlatVectorsReader(delegate.getMergeInstance(), fieldInfos);
    }
}