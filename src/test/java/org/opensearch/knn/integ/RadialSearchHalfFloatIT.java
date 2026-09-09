/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;

import java.util.List;

public class RadialSearchHalfFloatIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "half_float_radial_test_index";
    private static final String FIELD_NAME = "test_vector";
    private static final int DIMENSION = 4;

    // doc1: squaredDist=1 -> score=0.5, doc2: squaredDist=4 -> score=0.2, doc3: squaredDist=25 -> score=0.0385
    private static final Float[] QUERY_VECTOR_ORIGIN = { 0.0f, 0.0f, 0.0f, 0.0f };

    @SneakyThrows
    public void testRadialSearch_whenHalfFloatFlat_thenReturnsExpectedResults() {
        createKnnIndex(INDEX_NAME, buildHalfFloatFlatMapping());
        indexRadialTestDocs();

        // Only doc1 (score=0.5) clears a 0.3 threshold; doc2 (0.2) and doc3 (0.0385) do not.
        Response response = searchKNNIndex(INDEX_NAME, buildMinScoreQuery(0.3f), 10);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(1, results.size());
        assertEquals("1", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testRadialSearch_whenHalfFloatHnsw_thenReturnsExpectedResults() {
        createKnnIndex(INDEX_NAME, buildHalfFloatHnswMapping());
        indexRadialTestDocs();

        // doc1 (0.5) and doc2 (0.2) clear a 0.1 threshold; doc3 (0.0385) does not.
        Response response = searchKNNIndex(INDEX_NAME, buildMinScoreQuery(0.1f), 10);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(2, results.size());
    }

    @SneakyThrows
    public void testRadialSearch_whenHalfFloatHnswSq1Bit_thenReturnsExpectedResults() {
        createKnnIndex(INDEX_NAME, buildHalfFloatHnswSq1BitMapping());
        indexRadialTestDocs();

        Response response = searchKNNIndex(INDEX_NAME, buildMinScoreQuery(0.1f), 10);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(2, results.size());
    }

    private void indexRadialTestDocs() throws Exception {
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 0.0f, 0.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 2.0f, 0.0f, 0.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 5.0f, 0.0f, 0.0f, 0.0f });
    }

    private String buildHalfFloatFlatMapping() throws Exception {
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .compressionLevel("1x")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("flat").spaceType("l2").build())
            .build()
            .getIndexMapping();
    }

    private String buildHalfFloatHnswMapping() throws Exception {
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("hnsw").engine("lucene").spaceType("l2").build())
            .build()
            .getIndexMapping();
    }

    private String buildHalfFloatHnswSq1BitMapping() throws Exception {
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .compressionLevel("16x")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("hnsw").engine("lucene").spaceType("l2").build())
            .build()
            .getIndexMapping();
    }

    private XContentBuilder buildMinScoreQuery(float minScore) throws Exception {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", QUERY_VECTOR_ORIGIN)
            .field("min_score", minScore)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
    }
}
