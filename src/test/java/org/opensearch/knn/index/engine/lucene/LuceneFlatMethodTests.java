/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.NormalizeVectorTransformer;
import org.opensearch.knn.index.mapper.VectorTransformer;
import org.opensearch.knn.index.mapper.VectorTransformerFactory;

public class LuceneFlatMethodTests extends KNNTestCase {

    /**
     * Cosine on the flat FP16 format is scored by the native FP16_COSINE kernel as (1 + dot) / 2 clamped
     * to [0, 1], which only equals cosine for unit-length vectors, so writes must be normalized.
     */
    public void testGetVectorTransformer_cosine_returnsNormalizer() {
        VectorTransformer transformer = new LuceneFlatMethod().getVectorTransformer(SpaceType.COSINESIMIL);
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    /**
     * Only cosine normalizes. L2 and inner product are computed directly on the stored vectors, where
     * magnitude is meaningful -- normalizing them would silently change what those spaces measure.
     */
    public void testGetVectorTransformer_nonCosineSpaces_returnNoop() {
        LuceneFlatMethod method = new LuceneFlatMethod();
        for (SpaceType spaceType : LuceneFlatMethod.SUPPORTED_SPACES) {
            if (spaceType == SpaceType.COSINESIMIL) {
                continue;
            }
            assertSame(
                "expected no-op transformer for space type " + spaceType,
                VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER,
                method.getVectorTransformer(spaceType)
            );
        }
    }

    /**
     * Guards the L2 case specifically, since it is the space type most likely to be swept up by a broader
     * normalize-on-write change.
     */
    public void testGetVectorTransformer_l2_returnsNoop() {
        VectorTransformer transformer = new LuceneFlatMethod().getVectorTransformer(SpaceType.L2);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
        assertFalse(transformer instanceof NormalizeVectorTransformer);
    }
}
