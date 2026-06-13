import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import FPGACost, Operator
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.quantization_config import (
    FixedPointSpec,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationSpec,
    apply_quantization_config,
    compute_quant_params,
    to_fixed_point,
)


class MockOp(Operator):
    OP_TYPE = "Mock"

    def validate(self, values):
        pass

    def estimate_fpga_cost(self, values):
        return FPGACost(1)

    def hls_template_path(self):
        return ""

    def hls_context(self, values):
        return {}


def create_test_graph():
    # Simple v1 -> op1 -> v2
    v1 = Value("v1", ValueType.TENSOR, "float32", [1], ["N"])
    v2 = Value("v2", ValueType.TENSOR, "float32", [1], ["N"], producer_op_id="op1")
    op1 = MockOp("op1", ["v1"], ["v2"])
    return Graph({"v1": v1, "v2": v2}, {"op1": op1}, ["v1"], ["v2"])


def test_config_parsing_defaults():
    data = {
        "global": {
            "bit_width": 16,
            "scheme": "asymmetric",
            "fixed_point": {"integer_bits": 8, "fractional_bits": 8},
        }
    }
    config = QuantizationConfig.from_dict(data)
    assert config.global_default.bit_width == 16
    assert config.global_default.scheme == QuantizationScheme.ASYMMETRIC
    assert config.global_default.fixed_point is not None
    assert config.global_default.fixed_point.integer_bits == 8


def test_config_validation_invalid_bit_width():
    spec = QuantizationSpec(bit_width=1, scheme=QuantizationScheme.SYMMETRIC)
    with pytest.raises(ValueError, match="Unsupported bit_width"):
        spec.validate()


def test_config_validation_fixed_point_split():
    spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        fixed_point=FixedPointSpec(integer_bits=3, fractional_bits=3),  # 3+3=6 != 8
    )
    with pytest.raises(ValueError, match="Fixed-point split mismatch"):
        spec.validate()


def test_to_fixed_point_conversion():
    spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        fixed_point=FixedPointSpec(
            integer_bits=3, fractional_bits=4
        ),  # 3+4=7... wait, bit-width should be total.
        # If bit_width=8, 8 bits signed is -128 to 127
        # Fractional=4 means 1.0 = 16
    )
    spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        fixed_point=FixedPointSpec(integer_bits=4, fractional_bits=4),
    )

    # Value 1.5 with 4 fractional bits = 1.5 * 16 = 24
    assert to_fixed_point(1.5, spec) == 24
    # Value -1.5 with 4 fractional bits = -1.5 * 16 = -24
    assert to_fixed_point(-1.5, spec) == -24
    # Overflow check: max pos 127/16 = 7.9375
    assert to_fixed_point(10.0, spec) == 127
    # Underflow check: max neg -128/16 = -8.0
    assert to_fixed_point(-10.0, spec) == -128


def test_compute_quant_params_symmetric():
    spec = QuantizationSpec(bit_width=8, scheme=QuantizationScheme.SYMMETRIC)
    # Signed 8-bit range: -128 to 127.
    # ONNX signed symmetric usually scales around qmax=127.
    data = [-1.0, 0.0, 2.0]
    # abs_max = 2.0. Scale = 2.0 / 127
    scale, zp = compute_quant_params(data, spec)
    assert scale == 2.0 / 127
    assert zp == 0


def test_compute_quant_params_asymmetric():
    spec = QuantizationSpec(bit_width=8, scheme=QuantizationScheme.ASYMMETRIC)
    data = [0.0, 1.0, 3.0]
    # Range [0.0, 3.0], QRange [0, 255]
    # Scale = (3.0 - 0.0) / 255 = 3/255
    # ZP = 0 - 0/scale = 0
    scale, zp = compute_quant_params(data, spec)
    assert scale == 3.0 / 255
    assert zp == 0

    # Shifted range
    data = [-1.0, 1.0]
    # Range [-1.0, 1.0], QRange [0, 255]
    # Scale = 2.0 / 255
    # ZP = 0 - (-1.0)/(2.0/255) = 1.0 / (2.0/255) = 255 / 2 = 127.5 -> 128
    scale, zp = compute_quant_params(data, spec)
    assert zp == 128


def test_apply_quantization_overrides():
    graph = create_test_graph()
    # v1 is input, v2 produced by MockOp

    config_data = {
        "global": {
            "bit_width": 8,
            "scheme": "symmetric",
            "fixed_point": {"integer_bits": 4, "fractional_bits": 4},
        },
        "operators": {
            "Mock": {
                "bit_width": 12,
                "fixed_point": {"integer_bits": 4, "fractional_bits": 8},
            }
        },
        "tensors": {
            "v1": {
                "bit_width": 16,
                "fixed_point": {"integer_bits": 8, "fractional_bits": 8},
            }
        },
    }
    config = QuantizationConfig.from_dict(config_data)
    apply_quantization_config(graph, config)

    # v1 should have tensor override (16-bit)
    assert (
        graph.values["v1"].quant is not None
    ), "v1.quant was not set by apply_quantization_config"
    assert graph.values["v1"].quant["bit_width"] == 16
    # v2 produced by MockOp should have operator override (12-bit)
    assert (
        graph.values["v2"].quant is not None
    ), "v2.quant was not set by apply_quantization_config"
    assert graph.values["v2"].quant["bit_width"] == 12


def test_priority_tensor_over_operator():
    v1 = Value("v1", ValueType.TENSOR, "float32", [1], ["N"], producer_op_id="op1")
    op1 = MockOp("op1", [], ["v1"])
    graph = Graph({"v1": v1}, {"op1": op1}, [], ["v1"])

    config_data = {
        "global": {
            "bit_width": 8,
            "fixed_point": {"integer_bits": 4, "fractional_bits": 4},
        },
        "operators": {
            "Mock": {
                "bit_width": 10,
                "fixed_point": {"integer_bits": 4, "fractional_bits": 6},
            }
        },
        "tensors": {
            "v1": {
                "bit_width": 12,
                "fixed_point": {"integer_bits": 4, "fractional_bits": 8},
            }
        },
    }
    config = QuantizationConfig.from_dict(config_data)
    apply_quantization_config(graph, config)

    # v1 should be 12-bit, NOT 10-bit from operator
    assert (
        graph.values["v1"].quant is not None
    ), "v1.quant was not set by apply_quantization_config"
    assert graph.values["v1"].quant["bit_width"] == 12
