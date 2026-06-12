from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from edge_lstm.ir.graph import Graph


class QuantizationScheme(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class QuantizationType(Enum):
    FIXED_POINT = "fixed-point"
    INTEGER = "integer"


@dataclass(frozen=True)
class FixedPointSpec:
    integer_bits: int
    fractional_bits: int

    def __post_init__(self):
        if self.integer_bits < 0:
            raise ValueError("integer_bits must be non-negative")
        if self.fractional_bits < 0:
            raise ValueError("fractional_bits must be non-negative")


@dataclass
class QuantizationSpec:
    bit_width: int
    scheme: QuantizationScheme
    qtype: QuantizationType = QuantizationType.FIXED_POINT
    fixed_point: FixedPointSpec | None = None

    # Resolved parameters (populated during compute_quant_params or manually)
    scale: float | None = None
    zero_point: int | None = None

    def validate(self):
        if self.bit_width < 2 or self.bit_width > 32:
            raise ValueError(
                f"Unsupported bit_width: {self.bit_width}. Must be [2, 32]."
            )

        if self.qtype == QuantizationType.FIXED_POINT:
            if self.fixed_point is None:
                raise ValueError("Fixed-point quantization requires fixed_point spec.")
            if (
                self.fixed_point.integer_bits + self.fixed_point.fractional_bits
                != self.bit_width
            ):
                raise ValueError(
                    f"Fixed-point split mismatch: {self.fixed_point.integer_bits} + "
                    f"{self.fixed_point.fractional_bits} != {self.bit_width}"
                )


@dataclass
class QuantizationConfig:
    global_default: QuantizationSpec
    operator_overrides: dict[str, QuantizationSpec] = field(default_factory=dict)
    tensor_overrides: dict[str, QuantizationSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizationConfig:
        def parse_spec(
            d: dict[str, Any], default: QuantizationSpec | None = None
        ) -> QuantizationSpec:
            bw = d.get("bit_width", default.bit_width if default else 8)
            scheme_str = d.get(
                "scheme", default.scheme.value if default else "symmetric"
            )
            scheme = QuantizationScheme(scheme_str)
            qtype_str = d.get("type", default.qtype.value if default else "fixed-point")
            qtype = QuantizationType(qtype_str)

            fp_spec = None
            fp_data = d.get("fixed_point")
            if fp_data:
                fp_spec = FixedPointSpec(
                    integer_bits=fp_data["integer_bits"],
                    fractional_bits=fp_data["fractional_bits"],
                )
            elif default and default.fixed_point:
                # If bit_width changed but fp_data didn't, we might have a
                # mismatch. Simplified: only inherit if bit_width matches.
                fp_spec = default.fixed_point

            spec = QuantizationSpec(
                bit_width=bw, scheme=scheme, qtype=qtype, fixed_point=fp_spec
            )
            spec.validate()
            return spec

        global_spec = parse_spec(data.get("global", {}))

        ops = {}
        for op_type, op_data in data.get("operators", {}).items():
            ops[op_type] = parse_spec(op_data, default=global_spec)

        tensors = {}
        for t_name, t_data in data.get("tensors", {}).items():
            tensors[t_name] = parse_spec(t_data, default=global_spec)

        return cls(
            global_default=global_spec, operator_overrides=ops, tensor_overrides=tensors
        )


def to_fixed_point(value: float, spec: QuantizationSpec) -> int:
    """Convert a float to its fixed-point integer representation."""
    if spec.fixed_point is None:
        raise ValueError("Operator/Tensor must have fixed-point spec for conversion.")

    # Value = Mantissa * 2^(-fractional_bits)
    # Mantissa = Value * 2^(fractional_bits)
    scaled = value * (2**spec.fixed_point.fractional_bits)

    # Clip to bit-width
    qmin = -(2 ** (spec.bit_width - 1))
    qmax = (2 ** (spec.bit_width - 1)) - 1

    return int(max(qmin, min(qmax, round(scaled))))


def compute_quant_params(tensor_data: Any, spec: QuantizationSpec) -> tuple[float, int]:
    """Compute scale and zero_point for a given tensor and spec."""
    data = np.array(tensor_data)
    min_val, max_val = data.min(), data.max()

    qmin = 0
    qmax = (2**spec.bit_width) - 1

    if spec.scheme == QuantizationScheme.SYMMETRIC:
        # Symmetric: range is [-max(abs), max(abs)]
        abs_max = max(abs(min_val), abs(max_val))
        if abs_max == 0:
            return 1.0, 0
        # For symmetric signed, qmin = -2^(b-1), qmax = 2^(b-1)-1
        # But ONNX-style symmetric usually means symmetric around 0 in float space.
        # Scale = abs_max / (2^(b-1) - 1)
        qmax_signed = (2 ** (spec.bit_width - 1)) - 1
        scale = abs_max / qmax_signed if abs_max > 0 else 1.0
        return float(scale), 0
    else:
        # Asymmetric
        if max_val == min_val:
            return 1.0, 0
        scale = (max_val - min_val) / (qmax - qmin)
        initial_zero_point = qmin - min_val / scale
        zero_point = max(qmin, min(qmax, round(initial_zero_point)))
        return float(scale), int(zero_point)


def apply_quantization_config(graph: Graph, config: QuantizationConfig) -> None:
    """Resolve and attach quantization specs to all values in the graph."""
    for val_id, val in graph.values.items():
        # Resolve priority: Tensor > Operator > Global
        spec = config.global_default

        # Check operator override
        if val.producer_op_id and val.producer_op_id in graph.ops:
            op = graph.ops[val.producer_op_id]
            if op.op_type in config.operator_overrides:
                spec = config.operator_overrides[op.op_type]

        # Check tensor override
        if val_id in config.tensor_overrides:
            spec = config.tensor_overrides[val_id]

        # Attach to value.quant as a dictionary for serialization
        val.quant = {
            "bit_width": spec.bit_width,
            "scheme": spec.scheme.value,
            "type": spec.qtype.value,
            "integer_bits": spec.fixed_point.integer_bits if spec.fixed_point else None,
            "fractional_bits": spec.fixed_point.fractional_bits
            if spec.fixed_point
            else None,
            "scale": spec.scale,
            "zero_point": spec.zero_point,
        }
