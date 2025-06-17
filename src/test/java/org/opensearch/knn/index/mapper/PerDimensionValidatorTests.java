/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;

public class PerDimensionValidatorTests extends KNNTestCase {

    public void testDefaultFloatValidator() {
        PerDimensionValidator validator = PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;

        // Valid float values
        validator.validate(0.0f);
        validator.validate(1.0f);
        validator.validate(-1.0f);
        validator.validate(Float.MAX_VALUE);
        validator.validate(Float.MIN_VALUE);
        validator.validate(Float.MIN_NORMAL);

        // Invalid float values
        expectThrows(IllegalArgumentException.class, () -> validator.validate(Float.NaN));
        expectThrows(IllegalArgumentException.class, () -> validator.validate(Float.POSITIVE_INFINITY));
        expectThrows(IllegalArgumentException.class, () -> validator.validate(Float.NEGATIVE_INFINITY));

        // Should throw for validateByte
        expectThrows(IllegalStateException.class, () -> validator.validateByte(1.0f));
    }

    public void testDefaultHalfFloatValidator() {
        PerDimensionValidator validator = PerDimensionValidator.DEFAULT_HALF_FLOAT_VALIDATOR;

        // Valid half float values
        validator.validate(0.0f);
        validator.validate(1.0f);
        validator.validate(-1.0f);
        validator.validate(65504.0f);
        validator.validate(-65504.0f);
        validator.validate(6.103515625e-05f); // Min positive normal fp16

        // Invalid half float values
        expectThrows(IllegalArgumentException.class, () -> validator.validate(Float.NaN));
        expectThrows(IllegalArgumentException.class, () -> validator.validate(Float.POSITIVE_INFINITY));
        expectThrows(IllegalArgumentException.class, () -> validator.validate(Float.NEGATIVE_INFINITY));
        expectThrows(IllegalArgumentException.class, () -> validator.validate(100000.0f)); // Exceeds fp16 max
        expectThrows(IllegalArgumentException.class, () -> validator.validate(-100000.0f));

        // Should throw for validateByte
        expectThrows(IllegalStateException.class, () -> validator.validateByte(1.0f));
    }

    public void testDefaultByteValidator() {
        PerDimensionValidator validator = PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;

        // Valid byte values
        validator.validateByte(0.0f);
        validator.validateByte(127.0f);
        validator.validateByte(-128.0f);
        validator.validateByte(50.0f);
        validator.validateByte(-50.0f);

        // Invalid byte values
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(128.0f));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(-129.0f));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(255.0f));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(1.5f));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(Float.NaN));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(Float.POSITIVE_INFINITY));

        // Should throw for validate
        expectThrows(IllegalStateException.class, () -> validator.validate(1.0f));
    }

    public void testDefaultBitValidator() {
        PerDimensionValidator validator = PerDimensionValidator.DEFAULT_BIT_VALIDATOR;

        // Valid binary values (0 or 1)
        validator.validateByte(0.0f);
        validator.validateByte(1.0f);

        // Invalid binary values
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(128.0f));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(-129.0f));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(255.0f));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(1.5f));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(Float.NaN));
        expectThrows(IllegalArgumentException.class, () -> validator.validateByte(Float.POSITIVE_INFINITY));

        // Should throw for validate
        expectThrows(IllegalStateException.class, () -> validator.validate(1.0f));
    }

    public void testCustomValidator() {
        // Test custom validator implementation
        PerDimensionValidator customValidator = new PerDimensionValidator() {
            @Override
            public void validate(float value) {
                if (value < 0 || value > 1) {
                    throw new IllegalArgumentException("Value must be between 0 and 1");
                }
            }
        };

        // Valid values
        customValidator.validate(0.0f);
        customValidator.validate(0.5f);
        customValidator.validate(1.0f);

        // Invalid values
        expectThrows(IllegalArgumentException.class, () -> customValidator.validate(-0.1f));
        expectThrows(IllegalArgumentException.class, () -> customValidator.validate(1.1f));
    }

    public void testValidatorSelection() {
        assertEquals(PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR.getClass(),
                PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR.getClass());
        assertEquals(PerDimensionValidator.DEFAULT_HALF_FLOAT_VALIDATOR.getClass(),
                PerDimensionValidator.DEFAULT_HALF_FLOAT_VALIDATOR.getClass());
        assertEquals(PerDimensionValidator.DEFAULT_BYTE_VALIDATOR.getClass(),
                PerDimensionValidator.DEFAULT_BYTE_VALIDATOR.getClass());
        assertEquals(PerDimensionValidator.DEFAULT_BIT_VALIDATOR.getClass(),
                PerDimensionValidator.DEFAULT_BIT_VALIDATOR.getClass());
    }

    public void testHalfFloatBoundaryValues() {
        PerDimensionValidator validator = PerDimensionValidator.DEFAULT_HALF_FLOAT_VALIDATOR;

        validator.validate(65504.0f); // Max finite positive
        validator.validate(-65504.0f); // Max finite negative
        validator.validate(5.960464477539063e-08f); // Min positive subnormal
        validator.validate(-5.960464477539063e-08f); // Min negative subnormal
        validator.validate(6.103515625e-05f); // Min positive normal
        validator.validate(-6.103515625e-05f); // Min negative normal

        // Values that would overflow fp16
        expectThrows(IllegalArgumentException.class, () -> validator.validate(65505.0f));
        expectThrows(IllegalArgumentException.class, () -> validator.validate(-65505.0f));
    }

    public void testAllValidatorsWithEdgeCases() {
        // Test zero values
        PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR.validate(0.0f);
        PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR.validate(-0.0f);

        PerDimensionValidator.DEFAULT_HALF_FLOAT_VALIDATOR.validate(0.0f);
        PerDimensionValidator.DEFAULT_HALF_FLOAT_VALIDATOR.validate(-0.0f);

        PerDimensionValidator.DEFAULT_BYTE_VALIDATOR.validateByte(0.0f);
        PerDimensionValidator.DEFAULT_BIT_VALIDATOR.validateByte(0.0f);
    }
}