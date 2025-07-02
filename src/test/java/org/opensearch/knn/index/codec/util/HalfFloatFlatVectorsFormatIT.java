/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package src.test.java.org.opensearch.knn.index.codec.util;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene99.Lucene99Codec;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.HalfFloatFlatVectorsFormat;

import java.io.IOException;

public class HalfFloatFlatVectorsFormatIT {
    @Test
    public void testIndexAndRetrieveHalfFloatVector() throws IOException {
        Directory dir = new RAMDirectory();
        IndexWriterConfig config = new IndexWriterConfig();
        // Set codec to use our HalfFloatFlatVectorsFormat
        Codec codec = new Lucene99Codec() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new HalfFloatFlatVectorsFormat();
            }
        };
        config.setCodec(codec);

        int dim = 4;
        FieldType type = VectorDataType.HALF_FLOAT.createKnnVectorFieldType(dim, null);
        float[] vector = new float[] {1.1f, 2.2f, 3.3f, 4.4f};
        KnnFloatVectorField vectorField = new KnnFloatVectorField("vec", vector, type);
        Document doc = new Document();
        doc.add(vectorField);
        IndexWriter writer = new IndexWriter(dir, config);
        writer.addDocument(doc);
        writer.commit();
        writer.close();

        DirectoryReader reader = DirectoryReader.open(dir);
        LeafReader leaf = reader.leaves().get(0).reader();
        KnnVectorValues values = leaf.getVectorValues("vec");
        Assertions.assertNotNull(values);
        Assertions.assertTrue(values.nextDoc() != DocIdSetIterator.NO_MORE_DOCS);
        float[] readVec = values.vectorValue();
        Assertions.assertEquals(vector.length, readVec.length);
        for (int i = 0; i < vector.length; i++) {
            Assertions.assertEquals(vector[i], readVec[i], 1e-2f, "Mismatch at dim " + i);
        }
        reader.close();
        dir.close();
    }
}
