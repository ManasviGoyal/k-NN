/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.codecs;

import org.apache.lucene.index.FloatVectorValues;

/**
 * Exposes which source segment and ordinal {@link KnnVectorsWriter.MergedVectorValues}' merged float
 * view is currently positioned on, so a codec can read that segment's vectors in their stored form
 * instead of the float it hands out - which for an FP16 codec means decoding bytes it is
 * about to re-encode unchanged.
 *
 * <p>In Lucene's package because both pieces are package-private: {@code MergedFloat32VectorValues}
 * and its {@code current} field. The alternative - reimplementing the merged view over the public
 * {@link org.apache.lucene.index.DocIDMerger} - duplicates Lucene's merge ordering.
 */
public final class MergedVectorValuesAccessor {

    private MergedVectorValuesAccessor() {}

    /**
     * The source segment's own values at the merged view's current position, or null if it has not
     * been advanced onto a document yet. Throws if {@code mergedValues} did not come from
     * {@code mergeFloatVectorValues}.
     */
    public static FloatVectorValues currentSubValues(FloatVectorValues mergedValues) {
        KnnVectorsWriter.FloatVectorValuesSub sub = currentSub(mergedValues);
        return sub == null ? null : sub.values;
    }

    /**
     * The ordinal, within the segment {@link #currentSubValues} returns, of the vector at the merged
     * view's current position. Meaningless unless that call returned non-null.
     */
    public static int currentSubOrd(FloatVectorValues mergedValues) {
        KnnVectorsWriter.FloatVectorValuesSub sub = currentSub(mergedValues);
        if (sub == null) {
            throw new IllegalStateException("Merged vector values are not positioned on a document");
        }
        return sub.index();
    }

    private static KnnVectorsWriter.FloatVectorValuesSub currentSub(FloatVectorValues mergedValues) {
        if (mergedValues instanceof KnnVectorsWriter.MergedVectorValues.MergedFloat32VectorValues merged) {
            return merged.current;
        }
        // Distinct from a null `current`: that means "not positioned on a doc yet", this means the
        // value didn't come from mergeFloatVectorValues at all. Conflating them would report an
        // upstream change as an iteration bug.
        throw new IllegalArgumentException("Expected Lucene's merged float vector values, got " + mergedValues.getClass().getName());
    }
}
