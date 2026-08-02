"""Coverage for quantization spec validation and parameter edge cases."""

import numpy as np
import pytest

from tempo_dag.quantization_config import (
    FixedPointSpec,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationSpec,
    QuantizationType,
    StateQuantSpec,
    compute_quant_params,
    to_fixed_point,
)


def test_fixed_point_spec_rejects_negative_integer_bits():
    with pytest.raises(ValueError, match="integer_bits must be non-negative"):
        FixedPointSpec(integer_bits=-1, fractional_bits=4)


def test_fixed_point_spec_rejects_negative_fractional_bits():
    with pytest.raises(ValueError, match="fractional_bits must be non-negative"):
        FixedPointSpec(integer_bits=4, fractional_bits=-1)


def test_quantization_spec_requires_fixed_point_payload():
    spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        qtype=QuantizationType.FIXED_POINT,
        fixed_point=None,
    )
    with pytest.raises(ValueError, match="requires fixed_point spec"):
        spec.validate()


def test_state_quant_spec_rejects_blank_dtype():
    with pytest.raises(ValueError, match="dtype must be non-empty"):
        StateQuantSpec(dtype="   ", scale=1.0)


def test_state_quant_spec_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="scale must be positive"):
        StateQuantSpec(dtype="fixed16", scale=0.0)


def test_state_quant_spec_from_dict_rejects_non_dict_fixed_point():
    with pytest.raises(TypeError, match="fixed_point must be a dictionary"):
        StateQuantSpec.from_dict(
            {"dtype": "fixed16", "scale": 1.0, "fixed_point": [8, 8]}
        )


def test_config_operator_override_inherits_global_fixed_point():
    config = QuantizationConfig.from_dict(
        {
            "global": {
                "bit_width": 8,
                "fixed_point": {"integer_bits": 4, "fractional_bits": 4},
            },
            "operators": {"MatMul": {"scheme": "asymmetric"}},
        }
    )
    override = config.operator_overrides["MatMul"]
    assert override.scheme is QuantizationScheme.ASYMMETRIC
    assert override.fixed_point is config.global_default.fixed_point
    assert override.fixed_point.integer_bits == 4


def test_to_fixed_point_requires_fixed_point_spec():
    spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        qtype=QuantizationType.INTEGER,
    )
    with pytest.raises(ValueError, match="must have fixed-point spec"):
        to_fixed_point(1.5, spec)


def test_compute_quant_params_symmetric_all_zero_tensor():
    spec = QuantizationSpec(bit_width=8, scheme=QuantizationScheme.SYMMETRIC)
    scale, zero_point = compute_quant_params(np.zeros(4), spec)
    assert scale == 1.0
    assert zero_point == 0


def test_compute_quant_params_asymmetric_constant_tensor():
    spec = QuantizationSpec(bit_width=8, scheme=QuantizationScheme.ASYMMETRIC)
    scale, zero_point = compute_quant_params(np.full(4, 2.5), spec)
    assert scale == 1.0
    assert zero_point == 0
