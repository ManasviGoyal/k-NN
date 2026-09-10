/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableSet;

import lombok.Getter;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DYNAMIC_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.MAXIMUM_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MINIMUM_CONFIDENCE_INTERVAL;

/**
 * Lucene scalar quantization encoder
 */
public class LuceneSQEncoder implements Encoder {
    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT, VectorDataType.HALF_FLOAT);
    static final Set<Integer> LUCENE_SQ_BITS_SUPPORTED = Arrays.stream(Bits.values())
        .map(Bits::getValue)
        .collect(Collectors.toUnmodifiableSet());
    static final Bits LUCENE_PRE_360_SUPPORTED_SQ_BITS = Bits.SEVEN;

    /**
     * Supported bit widths for SQ quantization. Each maps to a specific quantization strategy, and the
     * compression a width achieves depends on the pre-quantization width of the data type: FLOAT starts
     * from 32 bits, HALF_FLOAT from 16, so every coded width saves exactly half as much on HALF_FLOAT.
     * 1-bit is x32 on FLOAT but x16 on HALF_FLOAT, 2-bit is x16 / x8, 4-bit is x8 / x4.
     *
     * <p>Not every width is valid for every data type - see {@link #FLOAT_BITS} and
     * {@link #HALF_FLOAT_BITS_BY_COMPRESSION}. {@code bits=7} is FLOAT-only (16/7 is not a power of two,
     * so it has no {@code Nx} name for HALF_FLOAT), and {@code bits=2}/{@code bits=4} are HALF_FLOAT-only
     * because FLOAT at those widths is served by Faiss, not by this encoder.</p>
     */
    @Getter
    public enum Bits {
        ONE(1),
        TWO(2),
        FOUR(4),
        SEVEN(7);

        private final int value;

        Bits(int value) {
            this.value = value;
        }

        /**
         * @param vectorDataType the vector data type being quantized, needed because the compression a
         *                       width achieves depends on the original (pre-quantization) bit width
         * @return the compression level {@code bits} quantization achieves for that data type
         */
        public CompressionLevel getCompressionLevel(VectorDataType vectorDataType) {
            final boolean isHalfFloat = vectorDataType == VectorDataType.HALF_FLOAT;
            return switch (this) {
                case ONE -> isHalfFloat ? CompressionLevel.x16 : CompressionLevel.x32;
                case TWO -> isHalfFloat ? CompressionLevel.x8 : CompressionLevel.x16;
                case FOUR -> isHalfFloat ? CompressionLevel.x4 : CompressionLevel.x8;
                case SEVEN -> CompressionLevel.x4;
            };
        }

        public static Bits fromValue(int value) {
            for (Bits b : values()) {
                if (b.value == value) return b;
            }
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Unsupported bits value: %d", value));
        }
    }

    private static final Set<Bits> FLOAT_BITS = Set.of(Bits.ONE, Bits.SEVEN);

    private static final Map<CompressionLevel, Bits> HALF_FLOAT_BITS_BY_COMPRESSION = Map.of(
        CompressionLevel.x4,
        Bits.FOUR,
        CompressionLevel.x8,
        Bits.TWO,
        CompressionLevel.x16,
        Bits.ONE
    );

    /** SQ width achieving {@code compressionLevel} on half_float, or null when the level resolves no encoder. */
    public static Bits halfFloatBitsFor(final CompressionLevel compressionLevel) {
        return HALF_FLOAT_BITS_BY_COMPRESSION.get(compressionLevel);
    }

    /** Compression levels half_float can reach through Lucene SQ. Excludes x1, which resolves no encoder. */
    public static Set<CompressionLevel> halfFloatSQCompressionLevels() {
        return HALF_FLOAT_BITS_BY_COMPRESSION.keySet();
    }

    /** Whether {@code bits} is a coded width that stores integer SQ codes rather than Lucene's stock byte SQ. */
    public static boolean isCodedBits(final int bits) {
        return bits == Bits.ONE.getValue() || bits == Bits.TWO.getValue() || bits == Bits.FOUR.getValue();
    }

    /**
     * Whether {@code bits} is accepted for the data type in {@code context}. Data-type aware on purpose:
     * {@link #LUCENE_SQ_BITS_SUPPORTED} is the union across data types, so checking membership in it alone
     * would let FLOAT through at 2 and 4 - widths this engine cannot write for FLOAT, since the codec sends
     * anything but a coded width to Lucene's stock byte-SQ format, which has no 2-bit variant.
     *
     * <p>A null context (or null data type) is treated as FLOAT, matching the field's default.</p>
     */
    private static Boolean isSupportedBitsForDataType(final Integer bits, final KNNMethodConfigContext context) {
        if (bits == null) {
            return false;
        }
        final VectorDataType vectorDataType = context == null ? null : context.getVectorDataType();
        if (vectorDataType == VectorDataType.HALF_FLOAT) {
            return isCodedBits(bits);
        }
        return FLOAT_BITS.stream().anyMatch(b -> b.getValue() == bits);
    }

    // Lucene SQ supports compression to 1 bit only in indices with version >= 3.6.0
    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_SQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            LUCENE_SQ_CONFIDENCE_INTERVAL,
            new Parameter.DoubleParameter(
                LUCENE_SQ_CONFIDENCE_INTERVAL,
                null,
                (v, context) -> v == DYNAMIC_CONFIDENCE_INTERVAL || (v >= MINIMUM_CONFIDENCE_INTERVAL && v <= MAXIMUM_CONFIDENCE_INTERVAL)
            )
        )
        .addParameter(
            LUCENE_SQ_BITS,
            // Making default value null - it should be passed in from LuceneHNSWMethodResolver
            new Parameter.IntegerParameter(LUCENE_SQ_BITS, null, LuceneSQEncoder::isSupportedBitsForDataType)
        )
        .build();

    /**
     * Validates the SQ encoder configuration on the resolved method context. Checks performed:
     * <ul>
     *     <li>The {@code bits} parameter is required on indices created with version 3.6.0 or later and must be a
     *     supported value (see {@link #LUCENE_SQ_BITS_SUPPORTED}); {@code bits=1} is rejected on earlier versions.</li>
     *     <li>The {@code bits} value must be compatible with any explicitly configured compression level
     *     (e.g. {@code bits=1} requires x32 compression, {@code bits=7} requires x4).</li>
     *     <li>Non-bit parameters (e.g. {@code confidence_interval}) are rejected when {@code bits=1}, since the
     *     1-bit scalar quantization path does not use them.</li>
     * </ul>
     * Returns silently without validation if either the method context or the config context is null.
     *
     * @param resolvedMethodContext the resolved method context containing the encoder parameters
     * @param configContext the config context containing index version and compression level
     * @throws ValidationException if any of the above checks fail
     */
    @Override
    public void validate(KNNMethodContext resolvedMethodContext, KNNMethodConfigContext configContext) {
        if (resolvedMethodContext == null || configContext == null) {
            return;
        }

        MethodComponentContext encoderContext = (MethodComponentContext) resolvedMethodContext.getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        if (encoderContext == null) {
            return;
        }

        Map<String, Object> encoderParams = encoderContext.getParameters();
        Version version = configContext.getVersionCreated();
        boolean isV360OrLater = version != null && version.onOrAfter(Version.V_3_6_0);
        Object bitsObj = encoderParams.get(LUCENE_SQ_BITS);

        ValidationException validationException = new ValidationException();

        if (isV360OrLater && bitsObj == null) {
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Parameter [%s] is required for encoder [%s] on indices created with version 3.6.0 or later. " + "Supported values: %s",
                    LUCENE_SQ_BITS,
                    ENCODER_SQ,
                    LUCENE_SQ_BITS_SUPPORTED
                )
            );
            throw validationException;
        }

        if (bitsObj instanceof Integer bits) {
            // Widths are per data type: half_float takes the coded widths (1, 2, 4) against its 16 bits,
            // FLOAT takes 1 and 7.
            if (configContext.getVectorDataType() == VectorDataType.HALF_FLOAT) {
                if (isCodedBits(bits) == false) {
                    validationException.addValidationError(
                        String.format(
                            Locale.ROOT,
                            "[%s] data type supports [%s] values %s for encoder [%s].",
                            VectorDataType.HALF_FLOAT.getValue(),
                            LUCENE_SQ_BITS,
                            HALF_FLOAT_BITS_BY_COMPRESSION.values().stream().map(Bits::getValue).sorted().toList(),
                            ENCODER_SQ
                        )
                    );
                    throw validationException;
                }
            } else if (FLOAT_BITS.contains(Bits.fromValue(bits)) == false) {
                validationException.addValidationError(
                    String.format(
                        Locale.ROOT,
                        "[%s] data type supports [%s] values %s for encoder [%s].",
                        configContext.getVectorDataType() == null
                            ? VectorDataType.FLOAT.getValue()
                            : configContext.getVectorDataType().getValue(),
                        LUCENE_SQ_BITS,
                        FLOAT_BITS.stream().map(Bits::getValue).sorted().toList(),
                        ENCODER_SQ
                    )
                );
                throw validationException;
            }

            if (isCodedBits(bits)) {
                Set<String> nonBitParameters = encoderParams.keySet()
                    .stream()
                    .filter(k -> !k.equals(LUCENE_SQ_BITS))
                    .collect(Collectors.toSet());
                if (!nonBitParameters.isEmpty()) {
                    validationException.addValidationError(
                        String.format(
                            Locale.ROOT,
                            "Parameters [%s] are not supported when [%s=%d] for encoder [%s]. "
                                + "The coded scalar quantization path does not use additional parameters.",
                            nonBitParameters,
                            LUCENE_SQ_BITS,
                            bits,
                            ENCODER_SQ
                        )
                    );
                    throw validationException;
                }
                if (!isV360OrLater) {
                    validationException.addValidationError(
                        String.format(
                            Locale.ROOT,
                            "Parameter [%s=%d] is only supported for indices created with version 3.6.0 or later. "
                                + "Supported values: %s",
                            LUCENE_SQ_BITS,
                            bits,
                            LUCENE_PRE_360_SUPPORTED_SQ_BITS
                        )
                    );
                    throw validationException;
                }
            }

            CompressionLevel configuredCompression = configContext.getCompressionLevel();
            if (CompressionLevel.isConfigured(configuredCompression)) {
                CompressionLevel expectedCompression = Bits.fromValue(bits).getCompressionLevel(configContext.getVectorDataType());
                if (configuredCompression != expectedCompression) {
                    validationException.addValidationError(
                        String.format(
                            Locale.ROOT,
                            "Compression level [%s] is incompatible with [%s=%d] for encoder [%s]. " + "Expected compression level: [%s]",
                            configuredCompression.getName(),
                            LUCENE_SQ_BITS,
                            bits,
                            ENCODER_SQ,
                            expectedCompression.getName()
                        )
                    );
                    throw validationException;
                }
            }
        }
    }

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        if (knnMethodConfigContext == null) {
            return CompressionLevel.x4;
        }

        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())) {
            return knnMethodConfigContext.getCompressionLevel();
        }

        // resolve compression level based on bits if its specified - the two must be equivalent
        if (methodComponentContext != null && methodComponentContext.getParameters() != null) {
            Object bitsObj = methodComponentContext.getParameters().get(LUCENE_SQ_BITS);
            if (bitsObj instanceof Integer) {
                return Bits.fromValue((Integer) bitsObj).getCompressionLevel(knnMethodConfigContext.getVectorDataType());
            }
        }

        // For indices after version 3.6.0, we want to default to 1-bit SQ's compression level.
        if (knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0)) {
            return Bits.ONE.getCompressionLevel(knnMethodConfigContext.getVectorDataType());
        }
        return CompressionLevel.x4;
    }

    @Override
    public EncoderType getEncoderType() {
        return EncoderType.SQ;
    }

    @Override
    public Set<QuantizationBits> getSupportedBits() {
        return EnumSet.of(QuantizationBits.ONE, QuantizationBits.SEVEN);
    }
}
