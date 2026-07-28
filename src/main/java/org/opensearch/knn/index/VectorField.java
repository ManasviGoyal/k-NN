/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;

public class VectorField extends Field {

    public VectorField(String name, float[] value, IndexableFieldType type) {
        this(name, value, type, VectorDataType.FLOAT);
    }

    public VectorField(String name, float[] value, IndexableFieldType type, VectorDataType vectorDataType) {
        super(name, new BytesRef(), type);
        try {
            if (vectorDataType == VectorDataType.HALF_FLOAT) {
                byte[] output = new byte[value.length * 2];
                KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArrayFallback(value, output, value.length);
                this.setBytesValue(output);
            } else {
                final KNNVectorSerializer vectorSerializer = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE;
                final byte[] floatToByte = vectorSerializer.floatToByteArray(value);
                this.setBytesValue(floatToByte);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @param name FieldType name
     * @param value an array of byte vector values
     * @param type FieldType to build DocValues
     */
    public VectorField(String name, byte[] value, IndexableFieldType type) {
        super(name, new BytesRef(), type);
        try {
            this.setBytesValue(value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }
}
