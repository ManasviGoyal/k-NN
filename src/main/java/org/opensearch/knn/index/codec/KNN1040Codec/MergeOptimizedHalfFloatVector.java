/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.MergedVectorValuesAccessor;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.search.VectorScorer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;

import java.io.IOException;

/**
 * Merge-time view that hands {@link KNN1040HalfFloatFlatVectorsWriter} each vector's stored FP16
 * bytes instead of a decoded {@code float[]}, removing one decode and one re-encode per vector.
 *
 * <p>Wraps {@code MergedVectorValues#mergeFloatVectorValues}, so doc order, deletions and index
 * sorting stay Lucene's; only data access differs. The merged view exposes only {@code float[]}, so
 * {@link MergedVectorValuesAccessor} supplies the source segment and ordinal to read bytes from.
 *
 * <p>The round trip it replaces is exact - FP16 widening is lossless and narrowing recovers every
 * finite value's bit pattern; NaN is rejected at index time. Segments that don't unwrap to FP16 fall
 * back to decode-then-encode: unreachable today, and it costs only that segment the fast path.
 */
class MergeOptimizedHalfFloatVector extends FloatVectorValues {

    private final FloatVectorValues mergedValues;
    private final int dimension;

    // Cached per segment: the merged view returns a fresh reference each call, so without this the
    // unwrap would run per vector.
    private FloatVectorValues cachedSubValues;
    private KNN1040HalfFloatFlatVectorsValues cachedHalfFloatValues;

    static MergeOptimizedHalfFloatVector create(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        return new MergeOptimizedHalfFloatVector(
            KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState),
            fieldInfo.getVectorDimension()
        );
    }

    private MergeOptimizedHalfFloatVector(FloatVectorValues mergedValues, int dimension) {
        this.mergedValues = mergedValues;
        this.dimension = dimension;
    }

    /**
     * Fills {@code dest} (exactly {@code dimension * Short.BYTES} bytes) with the current vector's
     * FP16 encoding - copied verbatim from the source segment, or encoded from {@code float[]} if that
     * segment isn't FP16-backed. Valid only after {@code nextDoc} has returned a real doc.
     */
    void currentVectorBytes(byte[] dest) throws IOException {
        final FloatVectorValues subValues = MergedVectorValuesAccessor.currentSubValues(mergedValues);
        if (subValues == null) {
            throw new IllegalStateException("nextDoc must be called before reading a vector");
        }
        final int subOrd = MergedVectorValuesAccessor.currentSubOrd(mergedValues);

        final KNN1040HalfFloatFlatVectorsValues halfFloatValues = halfFloatValuesFor(subValues);
        if (halfFloatValues != null) {
            halfFloatValues.readRawVectorBytes(subOrd, dest, 0);
            return;
        }
        KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(subValues.vectorValue(subOrd), dest, dimension);
    }

    private KNN1040HalfFloatFlatVectorsValues halfFloatValuesFor(FloatVectorValues subValues) {
        if (subValues != cachedSubValues) {
            cachedSubValues = subValues;
            cachedHalfFloatValues = unwrapHalfFloatValues(subValues, dimension);
        }
        return cachedHalfFloatValues;
    }

    @Override
    public DocIndexIterator iterator() {
        return mergedValues.iterator();
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
        return mergedValues.vectorValue(ord);
    }

    @Override
    public int size() {
        return mergedValues.size();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int ordToDoc(int ord) {
        return mergedValues.ordToDoc(ord);
    }

    @Override
    public VectorScorer scorer(float[] target) throws IOException {
        return mergedValues.scorer(target);
    }

    @Override
    public FloatVectorValues copy() throws IOException {
        return mergedValues.copy();
    }

    /**
     * Unwraps the layers a reader may add - {@link WrappedFloatVectorValues} and the
     * {@link MMapFloatVectorValues} used for mmap-backed segments - to reach the FP16 values, or null
     * if this segment isn't FP16-backed. The dimension check guards the caller's buffer: a mismatch
     * would have {@code readRawVectorBytes} write more bytes than it holds.
     */
    private static KNN1040HalfFloatFlatVectorsValues unwrapHalfFloatValues(FloatVectorValues values, int dimension) {
        FloatVectorValues candidate = values;
        while (candidate != null) {
            if (candidate instanceof KNN1040HalfFloatFlatVectorsValues halfFloatValues) {
                return halfFloatValues.dimension() == dimension ? halfFloatValues : null;
            }
            if (candidate instanceof MMapFloatVectorValues mmapValues) {
                candidate = mmapValues.getDelegate();
                continue;
            }
            final FloatVectorValues unwrapped = WrappedFloatVectorValues.getBottomFloatVectorValues(candidate);
            // Returns its argument unchanged when nothing is left to unwrap - that's the exit.
            if (unwrapped == candidate) {
                return null;
            }
            candidate = unwrapped;
        }
        return null;
    }
}
