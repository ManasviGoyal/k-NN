/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.util.BytesRef;
import org.junit.Assert;
import org.junit.Test;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;

import java.io.IOException;

public class KNNHalfFloatVectorValuesTest {
    @Test
    public void testGetVector_decodesHalfFloatCorrectly() throws IOException {
        float[] original = new float[] {1.5f, -2.25f, 3.75f, 0.0f};
        byte[] encoded = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(original);
        BinaryDocValues docValues = new TestBinaryDocValues(encoded);
        KNNVectorValuesIterator iterator = new KNNVectorValuesIterator.DocIdsIteratorValues(docValues);
        KNNHalfFloatVectorValues values = new KNNHalfFloatVectorValues(iterator);
        float[] decoded = values.getVector();
        Assert.assertArrayEquals(original, decoded, 1e-3f);
    }

    static class TestBinaryDocValues extends BinaryDocValues {
        private final byte[] value;
        public TestBinaryDocValues(byte[] value) {
            this.value = value;
        }
        @Override
        public BytesRef binaryValue() {
            return new BytesRef(value);
        }
        @Override
        public boolean advanceExact(int target) { return true; }
        @Override
        public int docID() { return 0; }
        @Override
        public int nextDoc() { return NO_MORE_DOCS; }
        @Override
        public int advance(int target) { return NO_MORE_DOCS; }
        @Override
        public long cost() { return 1; }
    }
}

