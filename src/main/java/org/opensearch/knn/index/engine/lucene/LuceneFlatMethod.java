/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.VectorTransformer;
import org.opensearch.knn.index.mapper.VectorTransformerFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;

/**
 * Lucene Flat implementation
 */
public class LuceneFlatMethod extends AbstractKNNMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT, VectorDataType.HALF_FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.L2,
        SpaceType.COSINESIMIL,
        SpaceType.INNER_PRODUCT
    );

    final static MethodComponent FLAT_METHOD_COMPONENT = initMethodComponent();

    /**
     * Identifies this method to {@link VectorTransformerFactory} as {@code METHOD_FLAT}. The factory only
     * reads the component name for the flat case, so no parameters are needed.
     */
    private final static MethodComponentContext FLAT_METHOD_COMPONENT_CONTEXT = new MethodComponentContext(
        METHOD_FLAT,
        Collections.emptyMap()
    );

    /**
     * Constructor for LuceneFlatMethod
     *
     * @see AbstractKNNMethod
     */
    public LuceneFlatMethod() {
        super(FLAT_METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new LuceneFlatSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_FLAT).addSupportedDataTypes(SUPPORTED_DATA_TYPES).build();
    }

    /**
     * Supplies the normalizing transformer for cosine so that {@code half_float} vectors are unit length
     * before they are written. {@code KNN1040HalfFloatFlatVectorsFormat} scores cosine with the native
     * FP16_COSINE kernel, which computes {@code (1 + dot) / 2} clamped to {@code [0, 1]}; that equals
     * cosine only when the magnitudes are 1, and otherwise saturates every score at 1.0. The query vector
     * is already normalized on this path by {@code KNNVectorFieldType#transformQueryVector}, so this is
     * the write side catching up, mirroring what {@code AbstractFaissMethod} already does for Faiss cosine.
     *
     * <p>The factory returns a no-op for every non-cosine space, and {@code EngineFieldMapper} only honors
     * this for {@code HALF_FLOAT}, so {@code float32} flat -- which resolves to the scalar-quantized
     * format -- keeps its existing unnormalized behavior.</p>
     */
    @Override
    protected VectorTransformer getVectorTransformer(SpaceType spaceType) {
        return VectorTransformerFactory.getVectorTransformer(KNNEngine.LUCENE, spaceType, FLAT_METHOD_COMPONENT_CONTEXT);
    }

}
