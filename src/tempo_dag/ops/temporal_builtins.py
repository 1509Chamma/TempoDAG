from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from tempo_dag.ir.op import FPGACost, InvalidOperatorInstanceError
from tempo_dag.ir.registry import OperatorRegistry
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ops.builtins import Add, BuiltinOperator, MatMul, _shape_product


@dataclass(frozen=True)
class FixedPointRange:
    """Fixed-point range hint attached before full quantization is resolved."""

    minimum: float
    maximum: float
    signed: bool = True

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("minimum must be <= maximum")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "signed": self.signed,
        }


@dataclass(frozen=True)
class TemporalMetadata:
    """State-threading metadata consumed by temporal lowering passes."""

    op_id: str
    op_type: str
    stateful: bool
    state_reads: tuple[str, ...] = ()
    state_writes: tuple[str, ...] = ()
    buffers: tuple[str, ...] = ()
    lag_cycles: int = 0
    window_size: int | None = None
    fixed_point_ranges: dict[str, FixedPointRange] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "stateful": self.stateful,
            "state_reads": list(self.state_reads),
            "state_writes": list(self.state_writes),
            "buffers": list(self.buffers),
            "lag_cycles": self.lag_cycles,
            "window_size": self.window_size,
            "fixed_point_ranges": {
                value_id: value_range.to_dict()
                for value_id, value_range in sorted(self.fixed_point_ranges.items())
            },
        }


class TemporalOperator(BuiltinOperator):
    """Base class for operators that expose temporal lowering metadata."""

    OP_TYPE: ClassVar[str | None] = "_TemporalOperatorBase"

    def temporal_metadata(self, values: Mapping[str, Value]) -> TemporalMetadata:
        self.validate(values)
        return TemporalMetadata(
            op_id=self.op_id,
            op_type=self.op_type,
            stateful=False,
            fixed_point_ranges=_fixed_point_ranges(self.attrs, self.outputs),
        )

    def _buffer_id(self, *, default: str) -> str:
        return _string_attr(self.attrs, "buffer_id", default)

    def _state_id(self, *, default: str) -> str:
        return _string_attr(self.attrs, "state_id", default)

    def _state_ids(self) -> list[str]:
        return _string_sequence_attr(self.attrs, "state_ids")


class TemporalAdd(Add, TemporalOperator):
    """Temporal-aware stateless Add bridge."""


class TemporalMatMul(MatMul, TemporalOperator):
    """Temporal-aware stateless MatMul bridge."""


class Delay(TemporalOperator):
    """Causal delay line that returns input from N timesteps ago."""

    OP_TYPE = "Delay"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        lag_cycles = self._lag_cycles()
        input_value = self._input_values(values)[0]
        self._require_scalar_or_tensor(input_value, "input[0]")
        output_type = (
            ValueType.SCALAR
            if input_value.vtype is ValueType.SCALAR
            else ValueType.TENSOR
        )
        self._match_output(
            values,
            shape=input_value.shape,
            axes=input_value.axes,
            dtype=input_value.dtype,
            vtype=output_type,
        )
        if lag_cycles < 1:
            raise InvalidOperatorInstanceError("Delay requires lag_cycles >= 1")

    def temporal_metadata(self, values: Mapping[str, Value]) -> TemporalMetadata:
        self.validate(values)
        buffer_id = self._buffer_id(default=f"{self.op_id}_buffer")
        return TemporalMetadata(
            op_id=self.op_id,
            op_type=self.op_type,
            stateful=True,
            state_reads=(buffer_id,),
            state_writes=(buffer_id,),
            buffers=(buffer_id,),
            lag_cycles=self._lag_cycles(),
            fixed_point_ranges=_fixed_point_ranges(self.attrs, self.outputs),
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        input_value = self._input_values(values)[0]
        work = max(1, _shape_product(input_value.shape))
        return FPGACost(
            latency_cycles=1,
            initiation_interval=1,
            bram=max(1, (work * self._lag_cycles() + 255) // 256),
            lut=max(1, work // 2),
            ff=max(1, work),
            metadata={"heuristic": "temporal_delay", "lag_cycles": self._lag_cycles()},
        )

    def _lag_cycles(self) -> int:
        return self._optional_int_attr("lag_cycles", 1)


class RollingWindow(TemporalOperator):
    """Causal rolling window view backed by bounded history state."""

    OP_TYPE = "RollingWindow"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        window_size = self._window_size()
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")
        self._match_output(
            values,
            shape=[window_size, *input_value.shape],
            axes=["window", *input_value.axes],
            dtype=input_value.dtype,
            vtype=ValueType.TENSOR,
        )

    def temporal_metadata(self, values: Mapping[str, Value]) -> TemporalMetadata:
        self.validate(values)
        buffer_id = self._buffer_id(default=f"{self.op_id}_buffer")
        return TemporalMetadata(
            op_id=self.op_id,
            op_type=self.op_type,
            stateful=True,
            state_reads=(buffer_id,),
            state_writes=(buffer_id,),
            buffers=(buffer_id,),
            lag_cycles=1,
            window_size=self._window_size(),
            fixed_point_ranges=_fixed_point_ranges(self.attrs, self.outputs),
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        input_value = self._input_values(values)[0]
        work = max(1, _shape_product(input_value.shape))
        window_size = self._window_size()
        return FPGACost(
            latency_cycles=window_size,
            initiation_interval=1,
            bram=max(1, (work * window_size + 255) // 256),
            lut=max(1, work),
            ff=max(1, work),
            metadata={"heuristic": "rolling_window", "window_size": window_size},
        )

    def _window_size(self) -> int:
        window_size = self._optional_int_attr("window_size", 1)
        if window_size < 1:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires window_size >= 1"
            )
        return window_size


class _RollingStat(TemporalOperator):
    OP_TYPE: ClassVar[str | None] = "_RollingStatBase"
    STAT_NAME: ClassVar[str] = "stat"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        window_size = self._window_size()
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")
        self._match_output(
            values,
            shape=input_value.shape,
            axes=input_value.axes,
            dtype=input_value.dtype,
            vtype=ValueType.TENSOR,
        )
        if window_size < 1:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires window_size >= 1"
            )

    def temporal_metadata(self, values: Mapping[str, Value]) -> TemporalMetadata:
        self.validate(values)
        state_id = self._state_id(default=f"{self.op_id}_{self.STAT_NAME}")
        buffer_id = self._buffer_id(default=f"{self.op_id}_buffer")
        output_id = self.outputs[0]
        return TemporalMetadata(
            op_id=self.op_id,
            op_type=self.op_type,
            stateful=True,
            state_reads=(state_id, buffer_id),
            state_writes=(state_id, buffer_id),
            buffers=(buffer_id,),
            lag_cycles=1,
            window_size=self._window_size(),
            fixed_point_ranges=_fixed_point_ranges(self.attrs, (output_id, state_id)),
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = max(1, _shape_product(self._output_value(values).shape))
        window_size = self._window_size()
        return FPGACost(
            latency_cycles=max(1, work * 2),
            initiation_interval=1,
            dsp=max(1, work),
            bram=max(1, (work * window_size + 255) // 256),
            lut=max(2, work * 2),
            ff=max(2, work * 2),
            metadata={"heuristic": self.STAT_NAME, "window_size": window_size},
        )

    def _window_size(self) -> int:
        return self._optional_int_attr("window_size", 1)


class RollingMean(_RollingStat):
    """Streaming rolling mean over a causal window."""

    OP_TYPE = "RollingMean"
    STAT_NAME = "rolling_mean"


class RollingVar(_RollingStat):
    """Streaming rolling variance over a causal window."""

    OP_TYPE = "RollingVar"
    STAT_NAME = "rolling_var"

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        base = super().estimate_fpga_cost(values)
        return FPGACost(
            latency_cycles=base.latency_cycles + self._output_work(values),
            initiation_interval=base.initiation_interval,
            dsp=base.dsp + max(1, self._output_work(values)),
            bram=base.bram,
            lut=base.lut + max(1, self._output_work(values)),
            ff=base.ff + max(1, self._output_work(values)),
            metadata={"heuristic": self.STAT_NAME, "window_size": self._window_size()},
        )


class ScanCell(TemporalOperator):
    """Placeholder bridge for lowered scan/loop bodies."""

    OP_TYPE = "ScanCell"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(range(1, 65))
        self._require_output_count(range(1, 65))
        for input_value in self._input_values(values):
            self._require_scalar_or_tensor(input_value, "input")
        for output_id in self.outputs:
            output_value = self._lookup_value(values, output_id, "output")
            self._require_scalar_or_tensor(output_value, "output")

    def temporal_metadata(self, values: Mapping[str, Value]) -> TemporalMetadata:
        self.validate(values)
        state_ids = tuple(self._state_ids())
        return TemporalMetadata(
            op_id=self.op_id,
            op_type=self.op_type,
            stateful=bool(state_ids),
            state_reads=state_ids,
            state_writes=state_ids,
            lag_cycles=1 if state_ids else 0,
            fixed_point_ranges=_fixed_point_ranges(self.attrs, self.outputs),
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = sum(
            max(
                1,
                _shape_product(self._lookup_value(values, output_id, "output").shape),
            )
            for output_id in self.outputs
        )
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            lut=max(1, work),
            ff=max(1, work),
            metadata={"heuristic": "scan_cell"},
        )


def _fixed_point_ranges(
    attrs: Mapping[str, object], value_ids: Sequence[str]
) -> dict[str, FixedPointRange]:
    ranges: dict[str, FixedPointRange] = {}
    raw_ranges = attrs.get("fixed_point_ranges", {})
    if not isinstance(raw_ranges, Mapping):
        raise InvalidOperatorInstanceError(
            "fixed_point_ranges must be a mapping from value id to range metadata"
        )

    for value_id in value_ids:
        range_data = raw_ranges.get(value_id)
        if isinstance(range_data, Mapping):
            minimum = range_data.get("minimum")
            maximum = range_data.get("maximum")
            signed = range_data.get("signed", True)
            if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
                ranges[value_id] = FixedPointRange(
                    minimum=float(minimum),
                    maximum=float(maximum),
                    signed=bool(signed),
                )
    return ranges


def _string_attr(attrs: Mapping[str, object], name: str, default: str) -> str:
    value = attrs.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise InvalidOperatorInstanceError(f"{name} must be a non-empty string")
    return value


def _string_sequence_attr(attrs: Mapping[str, object], name: str) -> list[str]:
    value = attrs.get(name, ())
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise InvalidOperatorInstanceError(f"{name} must be a sequence of strings")
    result = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise InvalidOperatorInstanceError(
                f"{name}[{idx}] must be a non-empty string"
            )
        result.append(item)
    return result


def register_temporal_builtin_operators(registry: OperatorRegistry) -> None:
    for operator_cls in TEMPORAL_BUILTIN_OPERATORS:
        registry.register(operator_cls)


TEMPORAL_BUILTIN_OPERATORS = [
    TemporalAdd,
    TemporalMatMul,
    Delay,
    RollingWindow,
    RollingMean,
    RollingVar,
    ScanCell,
]

TEMPORAL_BUILTIN_OPERATOR_TYPES = [
    operator_cls.operator_type() for operator_cls in TEMPORAL_BUILTIN_OPERATORS
]


__all__ = [
    "Delay",
    "FixedPointRange",
    "RollingMean",
    "RollingVar",
    "RollingWindow",
    "ScanCell",
    "TEMPORAL_BUILTIN_OPERATORS",
    "TEMPORAL_BUILTIN_OPERATOR_TYPES",
    "TemporalAdd",
    "TemporalMatMul",
    "TemporalMetadata",
    "TemporalOperator",
    "register_temporal_builtin_operators",
]
