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
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.index.engine.lucene.LuceneFlatMethod.FLAT_METHOD_COMPONENT;

/**
 * Resolves method configuration for the Lucene flat method. For FLOAT vectors, the flat method uses SQ
 * (1-bit quantization) without an HNSW graph, supporting only {@link org.opensearch.knn.index.mapper.CompressionLevel#x32}
 * compression. HALF_FLOAT vectors default to {@link org.opensearch.knn.index.mapper.CompressionLevel#x16}
 * (SQ 1-bit - 16x for HALF_FLOAT's 16-bit storage, vs 32x for FLOAT's 32-bit storage), but may also opt
 * into {@link org.opensearch.knn.index.mapper.CompressionLevel#x1} for exact FP16 storage with no
 * further reduction. Neither data type supports {@link org.opensearch.knn.index.mapper.Mode}.
 */
public class LuceneFlatMethodResolver extends AbstractMethodResolver {

    static final CompressionLevel DEFAULT_COMPRESSION = CompressionLevel.x32;
    static final CompressionLevel DEFAULT_COMPRESSION_HALF_FLOAT = CompressionLevel.x16;
    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS_HALF_FLOAT = Set.of(CompressionLevel.x1, CompressionLevel.x16);

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
        final CompressionLevel defaultCompression = isHalfFloat ? DEFAULT_COMPRESSION_HALF_FLOAT : DEFAULT_COMPRESSION;

        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        if (CompressionLevel.isConfigured(compressionLevel)) {
            // HALF_FLOAT supports x1 (exact FP16) and x16 (SQ 1-bit); FLOAT supports only x32 (SQ 1-bit) for now.
            final boolean isValid = isHalfFloat
                ? SUPPORTED_COMPRESSION_LEVELS_HALF_FLOAT.contains(compressionLevel)
                : compressionLevel == defaultCompression;
            if (isValid == false) {
                ValidationException validationException = new ValidationException();
                final String supportedCompressionLevels = isHalfFloat
                    ? String.format(Locale.ROOT, "\"%s\" or \"%s\"", CompressionLevel.x1.getName(), CompressionLevel.x16.getName())
                    : String.format(Locale.ROOT, "\"%s\"", defaultCompression.getName());
                validationException.addValidationError(
                    String.format(Locale.ROOT, "\"%s\" method only supports %s compression", METHOD_FLAT, supportedCompressionLevels)
                );
                throw validationException;
            }
            return compressionLevel;
        }
        return defaultCompression;
    }
}
