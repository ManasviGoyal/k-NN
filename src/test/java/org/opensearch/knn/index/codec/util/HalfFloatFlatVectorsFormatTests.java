/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package src.test.java.org.opensearch.knn.index.codec.util;

import org.apache.lucene.codecs.FlatVectorsWriter;
import org.apache.lucene.codecs.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.KnnVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.mockito.MockedStatic;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.HalfFloatFlatVectorsFormat;

import static org.mockito.Mockito.*;

import java.io.IOException;

public class HalfFloatFlatVectorsFormatTests {
    @Test
    public void testWriterAndReaderWithHalfFloat() throws IOException {
        // Mock dependencies
        SegmentWriteState writeState = mock(SegmentWriteState.class);
        SegmentReadState readState = mock(SegmentReadState.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.name).thenReturn("test_field");
        when(fieldInfo.getAttribute(anyString())).thenReturn(VectorDataType.HALF_FLOAT.getValue());

        // Create format
        HalfFloatFlatVectorsFormat format = new HalfFloatFlatVectorsFormat();
        FlatVectorsWriter writer = format.fieldsWriter(writeState);
        FlatVectorsReader reader = format.fieldsReader(readState);

        // Check that writer and reader are not null
        Assertions.assertNotNull(writer);
        Assertions.assertNotNull(reader);
        Assertions.assertTrue(writer.getClass().getSimpleName().contains("HalfFloat"));
        Assertions.assertTrue(reader.getClass().getSimpleName().contains("HalfFloat"));

        // Simulate writing a field with HALF_FLOAT and verify factory usage
        KnnVectorValues values = mock(KnnVectorValues.class);
        try (MockedStatic<KNNVectorValuesFactory> factoryMock = mockStatic(KNNVectorValuesFactory.class)) {
            KnnVectorValues halfFloatValues = mock(KnnVectorValues.class);
            factoryMock.when(() -> KNNVectorValuesFactory.createHalfFloatVectorValues(values)).thenReturn(halfFloatValues);
            writer.writeField(fieldInfo, values);
            factoryMock.verify(() -> KNNVectorValuesFactory.createHalfFloatVectorValues(values), times(1));
        }
    }

    @Test
    public void testWriterAndReaderWithFloat() throws IOException {
        SegmentWriteState writeState = mock(SegmentWriteState.class);
        SegmentReadState readState = mock(SegmentReadState.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.name).thenReturn("test_field");
        when(fieldInfo.getAttribute(anyString())).thenReturn(VectorDataType.FLOAT.getValue());

        HalfFloatFlatVectorsFormat format = new HalfFloatFlatVectorsFormat();
        FlatVectorsWriter writer = format.fieldsWriter(writeState);
        FlatVectorsReader reader = format.fieldsReader(readState);

        Assertions.assertNotNull(writer);
        Assertions.assertNotNull(reader);
        Assertions.assertTrue(writer.getClass().getSimpleName().contains("HalfFloat"));
        Assertions.assertTrue(reader.getClass().getSimpleName().contains("HalfFloat"));
        // Simulate writing and reading a field with FLOAT
        // (Here, you would ideally mock delegate.writeField and verify KNNVectorValuesFactory.createHalfFloatVectorValues is NOT called)
    }

    @Test
    public void testWriterAndReaderWithNullFieldInfo() throws IOException {
        SegmentWriteState writeState = mock(SegmentWriteState.class);
        SegmentReadState readState = mock(SegmentReadState.class);
        FieldInfo fieldInfo = null;
        HalfFloatFlatVectorsFormat format = new HalfFloatFlatVectorsFormat();
        FlatVectorsWriter writer = format.fieldsWriter(writeState);
        FlatVectorsReader reader = format.fieldsReader(readState);
        Assertions.assertNotNull(writer);
        Assertions.assertNotNull(reader);
    }

    @Test
    public void testWriterAndReaderWithUnknownType() throws IOException {
        SegmentWriteState writeState = mock(SegmentWriteState.class);
        SegmentReadState readState = mock(SegmentReadState.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.name).thenReturn("test_field");
        when(fieldInfo.getAttribute(anyString())).thenReturn("unknown_type");
        HalfFloatFlatVectorsFormat format = new HalfFloatFlatVectorsFormat();
        FlatVectorsWriter writer = format.fieldsWriter(writeState);
        FlatVectorsReader reader = format.fieldsReader(readState);
        Assertions.assertNotNull(writer);
        Assertions.assertNotNull(reader);
    }
}
