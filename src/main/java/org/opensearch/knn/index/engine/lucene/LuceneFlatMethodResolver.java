/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.index.engine.lucene.LuceneFlatMethod.FLAT_METHOD_COMPONENT;

/**
 * Resolves method configuration for the Lucene flat method. For FLOAT vectors, the flat method uses SQ
 * (1-bit quantization) without an HNSW graph, supporting only {@link org.opensearch.knn.index.mapper.CompressionLevel#x32}
 * compression. HALF_FLOAT vectors default to {@link org.opensearch.knn.index.mapper.CompressionLevel#x2}
 * (no SQ, exact FP16 storage, reflecting their actual on-disk footprint), but may also opt into
 * {@link org.opensearch.knn.index.mapper.CompressionLevel#x32} for SQ 1-bit with an FP16 (instead of FP32)
 * rescoring copy. Neither data type supports {@link org.opensearch.knn.index.mapper.Mode}.
 */
public class LuceneFlatMethodResolver extends AbstractMethodResolver {

    static final CompressionLevel DEFAULT_COMPRESSION = CompressionLevel.x32;
    static final CompressionLevel DEFAULT_COMPRESSION_HALF_FLOAT = CompressionLevel.x2;

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        validateNotTrainingContext(shouldRequireTraining, knnMethodConfigContext);
        validateParameters(knnMethodContext);
        validateMode(knnMethodConfigContext);

        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.LUCENE,
            spaceType,
            METHOD_FLAT
        );
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, FLAT_METHOD_COMPONENT);

        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(validateAndResolveCompressionLevel(knnMethodConfigContext))
            .build();
    }

    private void validateNotTrainingContext(boolean shouldRequireTraining, KNNMethodConfigContext knnMethodConfigContext) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, KNNEngine.LUCENE, null);
        if (validationException != null) {
            throw validationException;
        }
    }

    private void validateParameters(KNNMethodContext knnMethodContext) {
        Map<String, Object> parameters = knnMethodContext.getMethodComponentContext().getParameters();
        if (parameters != null && !parameters.isEmpty()) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(Locale.ROOT, "Parameters are not supported for the \"%s\" method", METHOD_FLAT)
            );
            throw validationException;
        }
    }

    private void validateMode(KNNMethodConfigContext knnMethodConfigContext) {
        if (Mode.isConfigured(knnMethodConfigContext.getMode())) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(Locale.ROOT, "\"%s\" is not supported for the \"%s\" method", MODE_PARAMETER, METHOD_FLAT)
            );
            throw validationException;
        }
    }

    private CompressionLevel validateAndResolveCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        final boolean isHalfFloat = VectorDataType.HALF_FLOAT == knnMethodConfigContext.getVectorDataType();
        // HALF_FLOAT isn't SQ by default, so it gets its own default instead of x32's rescore-triggering one.
        final CompressionLevel defaultCompression = isHalfFloat ? DEFAULT_COMPRESSION_HALF_FLOAT : DEFAULT_COMPRESSION;

        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        if (CompressionLevel.isConfigured(compressionLevel)) {
            // HALF_FLOAT may additionally opt into x32 (SQ 1-bit with an FP16 rescoring copy instead of FP32).
            final boolean isValid = compressionLevel == defaultCompression || (isHalfFloat && compressionLevel == CompressionLevel.x32);
            if (isValid == false) {
                ValidationException validationException = new ValidationException();
                final String supportedCompressionLevels = isHalfFloat
                    ? String.format(Locale.ROOT, "\"%s\" or \"%s\"", DEFAULT_COMPRESSION_HALF_FLOAT.getName(), CompressionLevel.x32.getName())
                    : String.format(Locale.ROOT, "\"%s\"", defaultCompression.getName());
                validationException.addValidationError(
                    String.format(
                        Locale.ROOT,
                        "\"%s\" method only supports %s compression",
                        METHOD_FLAT,
                        supportedCompressionLevels
                    )
                );
                throw validationException;
            }
            return compressionLevel;
        }
        return defaultCompression;
    }
}
