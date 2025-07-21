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

public class VectorField extends Field {

    /**
     * @param name FieldType name
     * @param value an array of float vector values
     * @param type FieldType to build DocValues
     * @param dataType VectorDataType (FLOAT or HALF_FLOAT)
     */
    public VectorField(String name, float[] value, IndexableFieldType type, VectorDataType dataType) {
        super(name, new BytesRef(), type);
        try {
            final byte[] floatToByte;
            if (dataType == VectorDataType.HALF_FLOAT) {
                KNNVectorAsCollectionOfHalfFloatsSerializer vectorSerializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(
                    value.length
                );
                floatToByte = vectorSerializer.floatToByteArray(value);
            } else {
                floatToByte = KNNVectorAsCollectionOfFloatsSerializer.INSTANCE.floatToByteArray(value);
            }
            setBytesValue(floatToByte);
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
