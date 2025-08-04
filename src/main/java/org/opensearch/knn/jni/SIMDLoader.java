
/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;

public class SIMDLoader {

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            try {
                if (!isAVX512SPRFusedDisabled() && PlatformUtils.isAVX512SPRSupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX512_SPR_JNI_LIBRARY_NAME);
                } else if (!isAVX512Disabled() && PlatformUtils.isAVX512SupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX512_JNI_LIBRARY_NAME);
                } else if (!isAVX2Disabled() && PlatformUtils.isAVX2SupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX2_JNI_LIBRARY_NAME);
                } else {
                    System.loadLibrary(KNNConstants.SIMD_JNI_LIBRARY_NAME);
                }
            } catch (UnsatisfiedLinkError e) {
                throw new RuntimeException("Failed to load native SIMD library", e);
            }
            return null;
        });
    }

    public static void ensureLoaded() {
        // Triggers the static block if not already loaded
    }

    private static boolean isAVX512Disabled() {
        return !Boolean.parseBoolean(System.getProperty("avx512.enabled", "true"));
    }

    private static boolean isAVX2Disabled() {
        return !Boolean.parseBoolean(System.getProperty("avx2.enabled", "true"));
    }

    private static boolean isAVX512SPRFusedDisabled() {
        return !Boolean.parseBoolean(System.getProperty("avx512_spr.enabled", "true"));
    }
}
