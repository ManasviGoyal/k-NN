/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.backward_codecs.lucene99.Lucene99RWHnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import com.google.common.annotations.VisibleForTesting;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;

import org.opensearch.common.collect.Tuple;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN1040BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KnnVectorsFormatContext;
import org.opensearch.knn.index.codec.LuceneVectorsFormatType;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120HnswBinaryVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.faiss.FaissCodecFormatResolver;
import org.opensearch.knn.index.engine.lucene.LuceneCodecFormatResolver;
import org.opensearch.knn.index.engine.lucene.LuceneFlatMethodResolver;
import org.opensearch.knn.index.engine.lucene.LuceneSQEncoder;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;

/**
 * Per-field KNN vectors format for the KNN1040 codec. Uses {@link Lucene99HnswVectorsFormat}
 * for HNSW, {@link Lucene99RWHnswScalarQuantizedVectorsFormat} for scalar quantization (to
 * preserve the {@code confidenceInterval} parameter), and
 * {@link Lucene104ScalarQuantizedVectorsFormat} with {@code SINGLE_BIT_QUERY_NIBBLE} encoding
 * for the flat SQ method.
 */
public class KNN1040PerFieldKnnVectorsFormat extends KNN1040BasePerFieldKnnVectorsFormat {

    private static final Tuple<Integer, ExecutorService> DEFAULT_MERGE_THREAD_COUNT_AND_EXECUTOR_SERVICE = Tuple.tuple(1, null);

    public KNN1040PerFieldKnnVectorsFormat(final Optional<MapperService> mapperService) {
        this(mapperService, new NativeIndexBuildStrategyFactory());
    }

    public KNN1040PerFieldKnnVectorsFormat(
        final Optional<MapperService> mapperService,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(
            mapperService,
            Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN,
            Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
            Lucene99HnswVectorsFormat::new,
            new LuceneCodecFormatResolver(buildLuceneFormatResolvers(), mapperService.orElse(null)),
            new FaissCodecFormatResolver(mapperService.orElse(null), nativeIndexBuildStrategyFactory),
            nativeIndexBuildStrategyFactory
        );
    }

    /**
     * Maps the {@code index.knn.advanced.approximate_threshold} setting value to the
     * {@code tinySegmentsThreshold} used by Lucene HNSW writers.
     * <ul>
     *   <li>{@code approximateThreshold < 0} (e.g. {@code -1}) → {@link Integer#MAX_VALUE} (never build the graph)</li>
     *   <li>{@code approximateThreshold >= 0} → returned as-is (0 = always build, N = skip when docCount &lt; N)</li>
     * </ul>
     */
    static int toTinySegmentsThreshold(int approximateThreshold) {
        if (approximateThreshold < 0) {
            return Integer.MAX_VALUE;
        }
        return approximateThreshold;
    }

    private static Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> buildLuceneFormatResolvers() {
        return Map.of(LuceneVectorsFormatType.HNSW, ctx -> {
            final KNNVectorsFormatParams p = new KNNVectorsFormatParams(
                ctx.getParams(),
                ctx.getDefaultMaxConnections(),
                ctx.getDefaultBeamWidth(),
                ctx.getMethodContext().getSpaceType()
            );
            final Tuple<Integer, ExecutorService> merge = getMergeThreadCountAndExecutorService();
            final int threshold = toTinySegmentsThreshold(ctx.getApproximateThreshold());
            if (p.getSpaceType() == SpaceType.HAMMING) {
                return new KNN9120HnswBinaryVectorsFormat(p.getMaxConnections(), p.getBeamWidth(), merge.v1(), merge.v2(), threshold);
            }
            // TODO: This branches on data type alone. Once x16 (SQ over FP16) lands, half_float will
            // also need to select a quantized format, so this must additionally gate on compression level.
            if (ctx.getVectorDataType() == VectorDataType.HALF_FLOAT) {
                return new KNN1040HnswHalfFloatVectorsFormat(p.getMaxConnections(), p.getBeamWidth(), merge.v1(), merge.v2(), threshold);
            }
            return new Lucene99HnswVectorsFormat(p.getMaxConnections(), p.getBeamWidth(), merge.v1(), merge.v2(), threshold);
        }, LuceneVectorsFormatType.SCALAR_QUANTIZED, ctx -> {
            final KNNScalarQuantizedVectorsFormatParams p = new KNNScalarQuantizedVectorsFormatParams(
                ctx.getParams(),
                ctx.getDefaultMaxConnections(),
                ctx.getDefaultBeamWidth()
            );
            final Tuple<Integer, ExecutorService> merge = getMergeThreadCountAndExecutorService();
            final int threshold = toTinySegmentsThreshold(ctx.getApproximateThreshold());
            if (LuceneSQEncoder.isCodedBits(p.getBits())) {
                final ScalarEncoding encoding = ScalarEncodingResolver.forDocBits(p.getBits());
                if (ctx.getVectorDataType() == VectorDataType.HALF_FLOAT) {
                    return new KNN1040HnswHalfFloatScalarQuantizedVectorsFormat(
                        encoding,
                        p.getMaxConnections(),
                        p.getBeamWidth(),
                        merge.v1(),
                        merge.v2(),
                        threshold
                    );
                }
                return new KNN1040HnswScalarQuantizedVectorsFormat(
                    encoding,
                    p.getMaxConnections(),
                    p.getBeamWidth(),
                    merge.v1(),
                    merge.v2(),
                    threshold
                );
            }
            return new Lucene99RWHnswScalarQuantizedVectorsFormat(
                p.getMaxConnections(),
                p.getBeamWidth(),
                merge.v1(),
                p.getBits(),
                p.isCompressFlag(),
                p.getConfidenceInterval(),
                merge.v2(),
                threshold
            );
        }, LuceneVectorsFormatType.FLAT, ctx -> {
            if (ctx.getVectorDataType() == VectorDataType.HALF_FLOAT) {
                // x1 is the only half_float level that stores raw fp16, with no encoder at all.
                if (LuceneSQEncoder.halfFloatBitsFor(ctx.getCompressionLevel()) == null) {
                    return new KNN1040HalfFloatFlatVectorsFormat();
                }
                return new KNN1040HalfFloatScalarQuantizedVectorsFormat(
                    resolveFlatScalarEncoding(ctx.getCompressionLevel(), ctx.getVectorDataType())
                );
            }
            return new KNN1040ScalarQuantizedVectorsFormat(resolveFlatScalarEncoding(ctx.getCompressionLevel(), ctx.getVectorDataType()));
        });
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    private static Tuple<Integer, ExecutorService> getMergeThreadCountAndExecutorService() {
        int mergeThreadCount = KNNSettings.getIndexThreadQty();
        if (mergeThreadCount <= 1) {
            return DEFAULT_MERGE_THREAD_COUNT_AND_EXECUTOR_SERVICE;
        }
        return Tuple.tuple(mergeThreadCount, Executors.newFixedThreadPool(mergeThreadCount));
    }

    /**
     * Picks the {@link ScalarEncoding} for the FLAT format from a field's compression level.
     *
     * <p>Data-type aware, because a compression level names a different width for each data type: levels
     * are measured against the pre-quantization width, so FLOAT's 32 bits give x8 -> 4-bit, x16 -> 2-bit,
     * x32 -> 1-bit, while HALF_FLOAT's 16 bits give x4 -> 4-bit, x8 -> 2-bit, x16 -> 1-bit. Using the FLOAT
     * table for half_float silently scrambles the widths, so the half_float mapping is taken from
     * {@link LuceneSQEncoder}, which owns it for both engines.</p>
     *
     * <p>A level with no mapping falls back to 1-bit. For half_float that is x1, which callers resolve to
     * the raw FP16 format before reaching here; {@link LuceneFlatMethodResolver} rejects unsupported levels
     * at mapping time, so an unexpected value would indicate an upstream invariant violation.</p>
     */
    @VisibleForTesting
    static ScalarEncoding resolveFlatScalarEncoding(final CompressionLevel compressionLevel, final VectorDataType vectorDataType) {
        if (vectorDataType == VectorDataType.HALF_FLOAT) {
            final LuceneSQEncoder.Bits halfFloatBits = LuceneSQEncoder.halfFloatBitsFor(compressionLevel);
            return ScalarEncodingResolver.forDocBits(
                halfFloatBits == null ? LuceneSQEncoder.Bits.ONE.getValue() : halfFloatBits.getValue()
            );
        }
        if (compressionLevel == CompressionLevel.x8) {
            return ScalarEncodingResolver.forDocBits(4);
        }
        if (compressionLevel == CompressionLevel.x16) {
            return ScalarEncodingResolver.forDocBits(2);
        }
        return ScalarEncodingResolver.forDocBits(1);
    }
}
