/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.index.engine.faiss.FaissHNSWMethod.HNSW_COMPONENT;
import static org.opensearch.knn.index.engine.faiss.FaissIVFMethod.IVF_COMPONENT;

public class FaissMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x2,
        CompressionLevel.x8,
        CompressionLevel.x16,
        CompressionLevel.x32
    );

    // x1 stores raw fp16 with no encoder; everything above it is whatever FaissSQEncoder can quantize
    // half_float to. Derived from that one table rather than listed again here, so a width added or
    // removed there cannot leave this set behind. x2 and x32 stay unreachable - 8-bit SQ is not a width
    // Faiss exposes, and x32 would need half a bit per dimension.
    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS_HALF_FLOAT = Stream.concat(
        Stream.of(CompressionLevel.x1),
        FaissSQEncoder.halfFloatSQCompressionLevels().stream()
    ).collect(Collectors.toUnmodifiableSet());

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        // Initial validation to ensure that there are no contradictions in provided parameters
        validateEncoderNotSpecifiedForHalfFloat(knnMethodContext, knnMethodConfigContext);
        validateConfig(knnMethodConfigContext);

        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.FAISS,
            spaceType,
            shouldRequireTraining ? METHOD_IVF : METHOD_HNSW
        );
        MethodComponent method = METHOD_HNSW.equals(resolvedKNNMethodContext.getMethodComponentContext().getName()) == false
            ? IVF_COMPONENT
            : HNSW_COMPONENT;
        Map<String, Encoder> encoderMap = method == HNSW_COMPONENT ? FaissHNSWMethod.SUPPORTED_ENCODERS : FaissIVFMethod.SUPPORTED_ENCODERS;

        // Fill in parameters for the encoder and then the method.
        resolveEncoder(resolvedKNNMethodContext, knnMethodConfigContext, encoderMap);
        // From the resolved method context, get the compression level and validate it against the passed in
        // configuration
        CompressionLevel resolvedCompressionLevel = resolveCompressionLevelFromMethodContext(
            resolvedKNNMethodContext,
            knnMethodConfigContext,
            encoderMap
        );

        // Validate encoder parameters
        validateEncoderConfig(resolvedKNNMethodContext, knnMethodConfigContext, encoderMap);

        // Validate that resolved compression doesnt have any conflicts
        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompressionLevel);
        knnMethodConfigContext.setCompressionLevel(resolvedCompressionLevel);
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, method);

        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(resolvedCompressionLevel)
            .build();
    }

    // AbstractMethodResolver.shouldEncoderBeResolved() only auto-resolves an encoder for FLOAT, so
    // half_float would fall through to the flat encoder and resolve to x1 - conflicting with a
    // configured quantized level. Widen just that data-type check here, the same way
    // LuceneHNSWMethodResolver does. Unlike FLOAT, where any configured level besides x1 resolves an
    // encoder, half_float resolves one exactly for the levels HALF_FLOAT_SQ_BITS covers.
    @Override
    protected boolean shouldEncoderBeResolved(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (isEncoderSpecified(knnMethodContext)) {
            return false;
        }

        if (knnMethodConfigContext.getVectorDataType() == VectorDataType.HALF_FLOAT) {
            return FaissSQEncoder.halfFloatBitsFor(halfFloatCompressionLevel(knnMethodConfigContext)) != null;
        }

        return super.shouldEncoderBeResolved(knnMethodContext, knnMethodConfigContext);
    }

    private static CompressionLevel halfFloatCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())) {
            return knnMethodConfigContext.getCompressionLevel();
        }
        return Mode.ON_DISK == knnMethodConfigContext.getMode() ? CompressionLevel.x16 : CompressionLevel.x1;
    }

    private void resolveEncoder(
        KNNMethodContext resolvedKNNMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        Map<String, Encoder> encoderMap
    ) {
        if (shouldEncoderBeResolved(resolvedKNNMethodContext, knnMethodConfigContext) == false) {
            return;
        }

        // Same shape as the FLOAT blocks below - build the encoder context, put the parameters, hand it to
        // applyEncoder - but taken before them. Those are written against FLOAT's 32 bits, where each level
        // picks a different encoder; half_float's levels are against its own 16 and always land on sq,
        // differing only in bit width, so the level-to-encoder decision collapses to a single lookup.
        if (VectorDataType.HALF_FLOAT == knnMethodConfigContext.getVectorDataType()) {
            Encoder.QuantizationBits halfFloatBits = FaissSQEncoder.halfFloatBitsFor(halfFloatCompressionLevel(knnMethodConfigContext));
            if (halfFloatBits == null) {
                return;
            }
            MethodComponentContext encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
            Encoder encoder = encoderMap.get(ENCODER_SQ);
            encoderComponentContext.getParameters().put(SQ_BITS, halfFloatBits.getValue());
            applyEncoder(resolvedKNNMethodContext, knnMethodConfigContext, encoderComponentContext, encoder);
            return;
        }

        CompressionLevel resolvedCompressionLevel = getDefaultCompressionLevel(knnMethodConfigContext);
        if (resolvedCompressionLevel == CompressionLevel.x1) {
            return;
        }

        // TODO: This chain of if-blocks mapping compression levels to encoder configs is too complex.
        // Need to refactor it into a strategy or registry pattern where each CompressionLevel declares
        // its own encoder factory, e.g. CompressionLevel.x2.createEncoder(context, encoderMap). That
        // would make it easier to add new compression level resolutions.
        MethodComponentContext encoderComponentContext = new MethodComponentContext(ENCODER_FLAT, new HashMap<>());
        Encoder encoder = encoderMap.get(ENCODER_FLAT);
        if (CompressionLevel.x2 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
            encoder = encoderMap.get(ENCODER_SQ);
            encoderComponentContext.getParameters().put(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16);
            // On 3.6.0+, also set bits for consistency with the new bits-based validation
            if (knnMethodConfigContext.getVersionCreated() != null
                && knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0)) {
                encoderComponentContext.getParameters().put(SQ_BITS, Encoder.QuantizationBits.SIXTEEN.getValue());
            }
        }

        if (CompressionLevel.x8 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
            encoder = encoderMap.get(QFrameBitEncoder.NAME);
            encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x8.numBitsForFloat32());
        }

        if (CompressionLevel.x16 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
            encoder = encoderMap.get(QFrameBitEncoder.NAME);
            encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x16.numBitsForFloat32());
        }

        if (CompressionLevel.x32 == resolvedCompressionLevel) {
            if (shouldUseSQOneBitForX32(knnMethodConfigContext, encoderMap)) {
                encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
                encoder = encoderMap.get(ENCODER_SQ);
                encoderComponentContext.getParameters().put(SQ_BITS, Encoder.QuantizationBits.ONE.getValue());
            } else {
                encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
                encoder = encoderMap.get(QFrameBitEncoder.NAME);
                encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x32.numBitsForFloat32());
            }
        }

        applyEncoder(resolvedKNNMethodContext, knnMethodConfigContext, encoderComponentContext, encoder);
    }

    private static void applyEncoder(
        KNNMethodContext resolvedKNNMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        MethodComponentContext encoderComponentContext,
        Encoder encoder
    ) {
        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            encoder.getMethodComponent(),
            knnMethodConfigContext
        );
        encoderComponentContext.getParameters().putAll(resolvedParams);

        // When auto-resolved to a coded bit width (1, 2 or 4), remove the type and clip defaults that
        // were injected — those paths don't use them, and validateEncoderConfig rejects them for any
        // width other than 16. Keyed off isSQCodedBits rather than a single width so the two stay in
        // step: every width that path accepts is a width validate refuses type and clip for.
        if (encoderComponentContext.getParameters().get(SQ_BITS) instanceof Integer bitsVal && FaissSQEncoder.isSQCodedBits(bitsVal)) {
            encoderComponentContext.getParameters().remove(FAISS_SQ_TYPE);
            encoderComponentContext.getParameters().remove(FAISS_SQ_CLIP);
        }

        resolvedKNNMethodContext.getMethodComponentContext().getParameters().put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
    }

    /**
     * half_float exposes exactly one knob - {@code compression_level}, x1 / x4 / x8 / x16 - so naming an
     * encoder is rejected rather than silently accepted. Checked against the user's own method context,
     * before resolution injects the matching {@code sq bits=N}: that injected encoder is internal and
     * must still work.
     */
    private void validateEncoderNotSpecifiedForHalfFloat(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext.getVectorDataType() != VectorDataType.HALF_FLOAT || isEncoderSpecified(knnMethodContext) == false) {
            return;
        }
        ValidationException validationException = new ValidationException();
        validationException.addValidationError(
            String.format(
                Locale.ROOT,
                "\"%s\" parameter is not supported for \"%s\" data type; use \"%s\" instead.",
                METHOD_ENCODER_PARAMETER,
                VectorDataType.HALF_FLOAT.getValue(),
                COMPRESSION_LEVEL_PARAMETER
            )
        );
        throw validationException;
    }

    // Method validates for explicit contradictions in the config
    private void validateConfig(KNNMethodConfigContext knnMethodConfigContext) {
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        ValidationException validationException = validateCompressionSupported(
            compressionLevel,
            supportedCompressionLevels(knnMethodConfigContext),
            KNNEngine.FAISS,
            knnMethodConfigContext.getVectorDataType(),
            null
        );
        if (validationException != null) {
            throw validationException;
        }
    }

    private static Set<CompressionLevel> supportedCompressionLevels(KNNMethodConfigContext knnMethodConfigContext) {
        return knnMethodConfigContext.getVectorDataType() == VectorDataType.HALF_FLOAT
            ? SUPPORTED_COMPRESSION_LEVELS_HALF_FLOAT
            : SUPPORTED_COMPRESSION_LEVELS;
    }

    protected void validateEncoderConfig(
        KNNMethodContext resolvedKnnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        Map<String, Encoder> encoderMap
    ) {
        if (isEncoderSpecified(resolvedKnnMethodContext) == false) {
            return;
        }
        Encoder encoder = encoderMap.get(getEncoderName(resolvedKnnMethodContext));
        if (encoder == null) {
            return;
        }

        encoder.validate(resolvedKnnMethodContext, knnMethodConfigContext);
    }

    private CompressionLevel getDefaultCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        return getDefaultCompressionLevel(knnMethodConfigContext, CompressionLevel.x32);
    }

    /**
     * Starting 3.6.0, x32 compression can use sq(bits=1) instead of the older QFrameBitEncoder (binary).
     * 1-bit quantization delegates to Lucene's flat format rather than the k-NN quantization
     * framework, which gives better recall. The encoderMap guard is needed because IVF doesn't
     * register the sq encoder — only HNSW does.
     *
     * Currently disabled — the SQ writer pipeline is not yet fully stable for auto-resolved
     * indices. Users can still explicitly specify sq(bits=1) to opt in. This will be enabled
     * as the default in Part 2.
     * TODO: Enable once the Faiss1040ScalarQuantizedKnnVectorsWriter pipeline is validated end-to-end.
     */
    private static boolean shouldUseSQOneBitForX32(KNNMethodConfigContext knnMethodConfigContext, Map<String, Encoder> encoderMap) {
        return knnMethodConfigContext.getVersionCreated() != null
            && knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0)
            && encoderMap.containsKey(ENCODER_SQ);
    }
}
