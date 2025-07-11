/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;

import org.opensearch.knn.jni.JNICommons;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.stream.IntStream;

/**
 * Class implements KNNVectorSerializer based on serialization/deserialization of float array
 * as a collection of individual half-precision values
 */
public class KNNVectorAsCollectionOfHalfFloatsSerializer implements KNNVectorSerializer {
    private static final int BYTES_IN_HALF_FLOAT = 2;

    public static final KNNVectorAsCollectionOfHalfFloatsSerializer INSTANCE = new KNNVectorAsCollectionOfHalfFloatsSerializer();

    /**
     * Converts a float array to a byte array using half-precision encoding.
     *
     * @param input the float[] to be serialized into half-precision format
     * @return a byte[] containing the float16-encoded data
     */
    @Override
    public byte[] floatToByteArray(float[] input) {
        final ByteBuffer bb = ByteBuffer.allocate(input.length * BYTES_IN_HALF_FLOAT).order(ByteOrder.BIG_ENDIAN);
        IntStream.range(0, input.length).forEach(index -> bb.putShort(Float.floatToFloat16(input[index])));
        byte[] bytes = new byte[bb.flip().limit()];
        bb.get(bytes);
        return bytes;
    }

    /**
     * Converts a BytesRef-wrapped byte array (encoded as float16) back into a float array.
     *
     * @param bytesRef the BytesRef containing float16-encoded vector data
     * @return a float containing the decoded float32 values
     */
    @Override
    public float[] byteToFloatArray(BytesRef bytesRef) {
        if (bytesRef == null || bytesRef.length % BYTES_IN_HALF_FLOAT != 0) {
            throw new IllegalArgumentException("Byte stream cannot be deserialized to array of half-floats");
        }
        byte[] halfFloatBytes = new byte[bytesRef.length];
        System.arraycopy(bytesRef.bytes, bytesRef.offset, halfFloatBytes, 0, bytesRef.length);
        return JNICommons.simdFp16ToFp32(halfFloatBytes);
    }

    /**
     * Deserializes all bytes from the stream to array of floats
     *
     * @param bytesRef bytes that will be used for deserialization to array of floats
     * @param floats array of floats to fill with deserialized values
     */
    @Override
    public void byteToFloatArray(BytesRef bytesRef, float[] floats) {
        if (bytesRef == null || bytesRef.length % BYTES_IN_HALF_FLOAT != 0) {
            throw new IllegalArgumentException("Byte stream cannot be deserialized to array of half-floats");
        }
        final int sizeOfFloatArray = bytesRef.length / BYTES_IN_HALF_FLOAT;

        ShortBuffer sb = ByteBuffer.wrap(bytesRef.bytes, bytesRef.offset, bytesRef.length).order(ByteOrder.BIG_ENDIAN).asShortBuffer();

        for (int i = 0; i < sizeOfFloatArray; i++) {
            floats[i] = Float.float16ToFloat(sb.get());
        }
    }
}
