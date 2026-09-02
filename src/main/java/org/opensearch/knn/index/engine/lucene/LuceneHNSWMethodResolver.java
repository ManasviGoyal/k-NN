/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SCALAR_QUANTIZER_DEFAULT_BITS_AFTER_V360;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.engine.lucene.LuceneHNSWMethod.HNSW_METHOD_COMPONENT;
import static org.opensearch.knn.index.engine.lucene.LuceneHNSWMethod.SUPPORTED_ENCODERS;

/**
 * Resolves method configuration for the Lucene HNSW method. Supports optional scalar quantization
 * encoding and compression-level-based resolution, with supported compression levels of x1, x4, and x32
 * for FLOAT. HALF_FLOAT vectors default to {@link org.opensearch.knn.index.mapper.CompressionLevel#x1}
 * (plain FP16 graph, no SQ; mirrors {@link LuceneFlatMethodResolver}), or may opt into x16 (SQ 1-bit,
 * {@code bits=1} - 16x for HALF_FLOAT's 16-bit storage, vs 32x for FLOAT's 32-bit storage).
 */
public class LuceneHNSWMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x4,
        CompressionLevel.x32
    );
    static final CompressionLevel DEFAULT_COMPRESSION_HALF_FLOAT = CompressionLevel.x1;
    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS_HALF_FLOAT = Set.of(
        DEFAULT_COMPRESSION_HALF_FLOAT,
        CompressionLevel.x16
    );

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        if (VectorDataType.HALF_FLOAT == knnMethodConfigContext.getVectorDataType()) {
            return resolveHalfFloatMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
        }

        validateConfig(knnMethodConfigContext, shouldRequireTraining);
        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.LUCENE,
            spaceType,
            METHOD_HNSW
        );
        resolveEncoder(resolvedKNNMethodContext, knnMethodConfigContext);
        resolveEncoderBitsAndValidate(knnMethodContext, resolvedKNNMethodContext, knnMethodConfigContext);
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, HNSW_METHOD_COMPONENT);
        CompressionLevel resolvedCompressionLevel = resolveCompressionLevelFromMethodContext(
            resolvedKNNMethodContext,
            knnMethodConfigContext,
            LuceneHNSWMethod.SUPPORTED_ENCODERS
        );
        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompressionLevel);
        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(resolvedCompressionLevel)
            .build();
    }

    private ResolvedMethodContext resolveHalfFloatMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, KNNEngine.LUCENE, null);
        validationException = validateCompressionSupported(
            knnMethodConfigContext.getCompressionLevel(),
            SUPPORTED_COMPRESSION_LEVELS_HALF_FLOAT,
            KNNEngine.LUCENE,
            validationException
        );
        // half_float's only encoder configuration (SQ 1-bit) is fully determined by compression_level
        // (x16) and auto-resolved internally - there's no tunable parameter surface to expose, so
        // users configure it via compression_level, not by writing an encoder block themselves.
        if (isEncoderSpecified(knnMethodContext)) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "\"%s\" parameter is not supported for \"%s\" data type; use \"%s\" instead.",
                    METHOD_ENCODER_PARAMETER,
                    VectorDataType.HALF_FLOAT.getValue(),
                    COMPRESSION_LEVEL_PARAMETER
                )
            );
        }
        // Same contradiction FLOAT's validateConfig rejects: asking for disk-optimized storage and then
        // opting out of compression.
        validationException = validateCompressionNotx1WhenOnDisk(knnMethodConfigContext, validationException);
        if (validationException != null) {
            throw validationException;
        }

        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.LUCENE,
            spaceType,
            METHOD_HNSW
        );
        resolveEncoder(resolvedKNNMethodContext, knnMethodConfigContext);
        resolveEncoderBitsAndValidate(knnMethodContext, resolvedKNNMethodContext, knnMethodConfigContext);
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, HNSW_METHOD_COMPONENT);

        // The CompressionLevel enum constant here (x1) is the same one FLOAT's own default resolves
        // to, but "x1" just means "no additional quantization beyond this data type's own native
        // representation" - it's data-type-relative, not a shared format. HALF_FLOAT still resolves
        // to KNN1040HnswHalfFloatVectorsFormat (FP16), never the FLOAT32 Lucene99HnswVectorsFormat
        // FLOAT's x1 resolves to. This dedicated dispatch exists for the encoder-resolution and
        // validation differences above, not because the two data types share any storage at x1.
        CompressionLevel resolvedCompressionLevel = isEncoderSpecified(resolvedKNNMethodContext)
            ? resolveCompressionLevelFromMethodContext(
                resolvedKNNMethodContext,
                knnMethodConfigContext,
                LuceneHNSWMethod.SUPPORTED_ENCODERS
            )
            : getDataTypeAwareDefaultCompressionLevel(knnMethodConfigContext);
        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompressionLevel);
        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(resolvedCompressionLevel)
            .build();
    }

    // AbstractMethodResolver.shouldEncoderBeResolved() only auto-resolves an encoder for FLOAT. Lucene
    // HNSW also supports SQ 1-bit for HALF_FLOAT, so widen just that data-type check here rather than
    // in the shared base class, which Faiss's resolver also uses. Checks for x16 specifically (rather
    // than reusing the "anything but x1" shortcut below), since x16 is half_float's only SQ-triggering
    // compression level - unlike FLOAT, where any configured level besides x1 (or ON_DISK mode with
    // no level configured) resolves an encoder.
    @Override
    protected boolean shouldEncoderBeResolved(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (isEncoderSpecified(knnMethodContext)) {
            return false;
        }

        if (knnMethodConfigContext.getVectorDataType() == VectorDataType.HALF_FLOAT) {
            // x16 is half_float's only SQ-triggering level, and ON_DISK with nothing configured defaults
            // to it - the half_float counterpart of FLOAT's ON_DISK -> x32.
            return getDataTypeAwareDefaultCompressionLevel(knnMethodConfigContext) == CompressionLevel.x16;
        }

        if (knnMethodConfigContext.getCompressionLevel() == CompressionLevel.x1) {
            return false;
        }

        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel()) == false
            && Mode.ON_DISK != knnMethodConfigContext.getMode()) {
            return false;
        }

        return knnMethodConfigContext.getVectorDataType() == VectorDataType.FLOAT;
    }

    protected void resolveEncoder(KNNMethodContext resolvedKNNMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (shouldEncoderBeResolved(resolvedKNNMethodContext, knnMethodConfigContext) == false) {
            return;
        }

        CompressionLevel resolvedCompressionLevel = getDataTypeAwareDefaultCompressionLevel(knnMethodConfigContext);
        if (resolvedCompressionLevel == CompressionLevel.x1) {
            return;
        }

        MethodComponentContext methodComponentContext = resolvedKNNMethodContext.getMethodComponentContext();

        String encoderName;
        MethodComponent encoderComponent;

        encoderName = LuceneHNSWMethod.SQ_ENCODER.getName();
        encoderComponent = LuceneHNSWMethod.SQ_ENCODER.getMethodComponent();

        MethodComponentContext encoderComponentContext = new MethodComponentContext(encoderName, new HashMap<>());
        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            encoderComponent,
            knnMethodConfigContext
        );

        encoderComponentContext.getParameters().putAll(resolvedParams);
        methodComponentContext.getParameters().put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
    }

    // if encoder gets resolved, determine if default bits need to be added and validate encoder config makes sense
    private void resolveEncoderBitsAndValidate(
        KNNMethodContext originalMethodContext,
        KNNMethodContext resolvedKNNMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        if (!isEncoderSpecified(resolvedKNNMethodContext)) {
            return;
        }
        boolean didUserSpecifyEncoder = isEncoderSpecified(originalMethodContext);
        boolean isV360OrLater = knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0);

        MethodComponentContext encoderComponentContext = getEncoderComponentContext(resolvedKNNMethodContext);
        if (encoderComponentContext == null) {
            return;
        }

        boolean bitsAlreadySet = encoderComponentContext.getParameters().containsKey(LUCENE_SQ_BITS);
        boolean skipAutoResolve = isV360OrLater && didUserSpecifyEncoder;

        if (bitsAlreadySet == false && skipAutoResolve == false) {
            CompressionLevel effectiveCompression = CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())
                ? knnMethodConfigContext.getCompressionLevel()
                : getDataTypeAwareDefaultCompressionLevel(knnMethodConfigContext);
            boolean useNewDefault = isV360OrLater
                && LuceneSQEncoder.Bits.fromValue(LUCENE_SCALAR_QUANTIZER_DEFAULT_BITS_AFTER_V360)
                    .getCompressionLevel(knnMethodConfigContext.getVectorDataType()) == effectiveCompression;
            encoderComponentContext.getParameters()
                .put(LUCENE_SQ_BITS, useNewDefault ? LUCENE_SCALAR_QUANTIZER_DEFAULT_BITS_AFTER_V360 : LUCENE_SQ_DEFAULT_BITS);
        }
        String encoderName = encoderComponentContext.getName();
        Encoder encoder = SUPPORTED_ENCODERS.get(encoderName);

        if (encoder == null) {
            return;
        }
        encoder.validate(resolvedKNNMethodContext, knnMethodConfigContext);
    }

    // Method validates for explicit contradictions in the config
    private void validateConfig(KNNMethodConfigContext knnMethodConfigContext, boolean shouldRequireTraining) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, KNNEngine.LUCENE, null);
        validationException = validateCompressionSupported(
            knnMethodConfigContext.getCompressionLevel(),
            SUPPORTED_COMPRESSION_LEVELS,
            KNNEngine.LUCENE,
            validationException
        );
        validationException = validateCompressionNotx1WhenOnDisk(knnMethodConfigContext, validationException);
        if (validationException != null) {
            throw validationException;
        }
    }

    private CompressionLevel getDefaultCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        return getDefaultCompressionLevel(knnMethodConfigContext, CompressionLevel.x4);
    }

    /**
     * Default compression for {@code knnMethodConfigContext}, expressed in that data type's own terms.
     * The shared default is measured against FLOAT: ON_DISK resolves to x32, which half_float does not
     * support - its equivalent "quantize as far as this data type goes" level is x16 (SQ 1-bit on 16-bit
     * storage). Used everywhere a default is needed, so encoder resolution, bit-width resolution and the
     * reported compression level cannot disagree.
     */
    private CompressionLevel getDataTypeAwareDefaultCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext.getVectorDataType() != VectorDataType.HALF_FLOAT) {
            return getDefaultCompressionLevel(knnMethodConfigContext);
        }
        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())) {
            return knnMethodConfigContext.getCompressionLevel();
        }
        return Mode.ON_DISK == knnMethodConfigContext.getMode() ? CompressionLevel.x16 : DEFAULT_COMPRESSION_HALF_FLOAT;
    }
}
